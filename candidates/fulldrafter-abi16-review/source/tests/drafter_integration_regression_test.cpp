#include <cstdint>
#include <cstddef>
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
#include "sched/advance_impl.h"
#include "diag.h"
#include "text/tokenizer.h"

#define private public
#include "sched/scheduler.h"
#include "sched/batch.h"
#include "engine.h"
#undef private
#include "engine_priv.h"

#include "rad_core.h"


#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)
using namespace rad;
using namespace radiance;
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
    int rollback_calls = 0;
    int64_t last_valid = -1;
    int rollback(uint64_t, int64_t valid) override {
        ++rollback_calls; last_valid = valid; return RAD_OK;
    }
    int64_t first_block_pos(uint64_t, int32_t) const override { return 0; }
    int64_t take_table_changes(uint64_t, int32_t) const override { return 0; }
};


static void test_block_anchor() {
    MinimalFakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    sched.cfg_.max_tok = 128;
    CHECK(sched.draft_.configure({12, false}) == RAD_OK);
    auto req = std::make_unique<Request>();
    req->id = 7101; req->prompt = {17, 18}; req->output = {42, 0};
    req->n_computed = 137; req->state = ReqState::Running; req->may_speculate = true;
    CHECK(sched.add(std::move(req)) == RAD_OK);
    const int slot = sched.by_id_.at(7101);
    auto& s = *sched.reqs_[slot];
    s.running = true; s.req->state = ReqState::Running;
    StepEntry e{}; e.seq = 7101; e.slot = slot; e.card = 0;
    e.ctx_len = 0; e.n_tokens = 1; e.produces_token = true;
    sched.plan_.e.push_back(e);
    Scheduler::BlockRow row;
    CHECK(row.anchor_token == -1 && row.root_pos == -1);
    CHECK(sched.block_begin(13, 12, 0, 99, &row, 1) == 1);
    CHECK(row.anchor_token == 0 && row.root_pos == 137 && row.seq == 7101);
    s.req->output.back() = 123;
    s.req->n_computed = 149;
    CHECK(sched.block_begin(13, 12, 0, 99, &row, 1) == 1);
    CHECK(row.anchor_token == 123 && row.root_pos == 149);
    // Multi-stage block drafter (2 x 6 = 12, block=7, depth=12)
    CHECK(sched.block_begin(7, 12, 0, 99, &row, 1) == 1);
    CHECK(row.anchor_token == 123 && row.root_pos == 149);
    // Invalid multi-stage drafter (13 % 6 != 0)
    CHECK(sched.block_begin(7, 13, 0, 99, &row, 1) == 0);
    // Invalid block < 2
    CHECK(sched.block_begin(1, 12, 0, 99, &row, 1) == 0);
    // Invalid geometry with depth < block: block=10, depth=3
    CHECK(sched.block_begin(10, 3, 0, 99, &row, 1) == 0);
    s.req->output.clear();
    CHECK(sched.block_begin(13, 12, 0, 99, &row, 1) == 0);
}
static void test_tree_publication() {
    Scheduler sched;
    auto req = std::make_unique<Request>(); req->id = 7102;
    req->prompt = {5}; req->may_speculate = true; req->max_tokens = 3;
    CHECK(sched.add(std::move(req)) == RAD_OK);
    const int slot = sched.by_id_.at(7102);
    auto& s = *sched.reqs_[slot];
    uint32_t drafts[12]; for (int i=0;i<12;++i) drafts[i]=100+i;
    CandidateTree tree; tree.build_linear(0, drafts, 12);
    CHECK(sched.set_draft_tree(7102, tree, 137) == RAD_OK);
    ActiveTreeRecord rec;
    CHECK(TreeRegistry::instance().get(7102, &rec));
    CHECK(s.n_draft == 2 && rec.tree.node_count() == 3);
    CHECK(s.draft[0] == 100 && s.draft[1] == 101);
    CHECK(rec.tree[0].token_id == 0 && rec.positions[0] == 137 && rec.positions[2] == 139);
    s.req->max_tokens = 1;
    CHECK(sched.set_draft_tree(7102, tree, 137) == RAD_OK);
    CHECK(!TreeRegistry::instance().get(7102, &rec));
    CHECK(s.n_draft == 0 && !s.has_draft_tree && s.req->draft == nullptr);
    s.req->max_tokens = 20;
    CHECK(sched.set_draft_tree(7102, tree, 137) == RAD_OK);
    s.req->may_speculate = false;
    CHECK(sched.set_draft_tree(7102, tree, 137) == RAD_OK);
    CHECK(!TreeRegistry::instance().get(7102, &rec));
    s.req->may_speculate = true;
    tree.build_linear(0, nullptr, 0);
    CHECK(sched.set_draft_tree(7102, tree, 137) == RAD_OK);
    CHECK(!TreeRegistry::instance().get(7102, &rec));
    CHECK(sched.set_draft_tree(999999, tree, 137) == RAD_E_NOTFOUND);
}
static void test_token_ids_are_not_eos() {
    for (uint32_t token : {0u, 151643u, 151645u}) {
        uint32_t draft[] = {token, 72};
        CandidateTree tree; tree.build_linear(19, draft, 2);
        int32_t preds[] = {static_cast<int32_t>(token), 72, 81};
        int32_t out[4] = {-1,-1,-1,-1}, accepted=-1, bonus=-1;
        CHECK(accept_tree_greedy(tree, preds, 3, out, &accepted, &bonus) == RAD_OK);
        CHECK(accepted == 2 && out[0] == static_cast<int32_t>(token));
        CHECK(out[1] == 72 && out[2] == 81 && bonus == 81);
    }
}
static void test_short_candidate_buffers() {
    for (size_t n=0;n<=15;++n) {
        std::vector<uint32_t> candidates(n); for(size_t i=0;i<n;++i)candidates[i]=100+i;
        for(int mode=0;mode<3;++mode) {
            CandidateTree tree;
            if(mode<2)tree.build_top2(0,candidates.data(),mode==1,n);
            else tree.build_top4(0,candidates.data(),n);
            CHECK(tree.node_count()==n+1);
            for(size_t i=1;i<=n;++i)CHECK(tree[i].token_id==candidates[i-1]);
        }
    }
    // Fixed topologies still end at their16-entry tables even when the host can hold65 nodes.
    uint32_t wide_candidates[64];
    for (size_t i = 0; i < 64; ++i) wide_candidates[i] = static_cast<uint32_t>(100 + i);
    for (int mode = 0; mode < 3; ++mode) {
        CandidateTree tree;
        if (mode < 2) tree.build_top2(0, wide_candidates, mode == 1, 64);
        else tree.build_top4(0, wide_candidates, 64);
        CHECK(tree.node_count() == 16);
        for (int i = 1; i < 16; ++i) CHECK(tree[i].token_id == wide_candidates[i - 1]);
    }
}
static void test_unwired_sidecar_rejected() {
    const char* prev = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
    const bool had = prev != nullptr; const std::string saved = prev ? prev : "";
    setenv("RADIANCE_FULLDRAFTER_SIDECAR", "/does/not/read/checkpoint.safetensors", 1);
    { Config cfg; Engine engine(cfg); CHECK(engine.step_state_init() == RAD_E_INVAL); }
    if(had)setenv("RADIANCE_FULLDRAFTER_SIDECAR",saved.c_str(),1);
    else unsetenv("RADIANCE_FULLDRAFTER_SIDECAR");
}

// Exercise real collect_step -> commit, not a simulated acceptance loop.
static void test_rollback_owner_and_emission() {
    for (bool tree_mode : {false, true})
    for (int accepted : {0, 1, 3})
    for (int pipeline : {0, 1, 2}) {
        Program program; Config cfg; Engine engine(cfg);
        auto owned = std::make_unique<MinimalFakeKV>();
        auto* kv = owned.get(); engine.kv_iface_ = std::move(owned);
        auto& sched = engine.sched_; sched.kv_ = kv; sched.prog_ = &program;
        auto req = std::make_unique<Request>(); req->id = 7201;
        req->prompt = {8, 9}; req->output = {19}; req->n_computed = 17;
        req->may_speculate = true; req->max_tokens = 100;
        CHECK(sched.add(std::move(req)) == RAD_OK);
        const int slot = sched.by_id_.at(7201); auto& s = *sched.reqs_[slot];
        s.running = true; s.req->state = ReqState::Running; s.entry = 0;
        s.entry_next = pipeline == 2 ? 0 : -1;
        sched.next_open_ = pipeline != 0; sched.step_open_ = true;
        uint32_t drafts[] = {31,32,33}; CandidateTree tree;
        tree.build_linear(19, drafts, 3);
        CHECK(sched.set_draft_tree(7201, tree, 17) == RAD_OK);
        if (!tree_mode) TreeRegistry::instance().clear(7201);
        StepEntry e{}; e.seq = 7201; e.slot = slot; e.card = 0;
        e.ctx_len = 17; e.n_tokens = 4; e.n_spec = 3; e.n_draft = 3;
        e.req = s.req.get(); e.draft = s.draft; e.produces_token = true;
        sched.plan_.e.push_back(e);
        int32_t sampled[] = {31,32,33,77}; sampled[accepted] = 77;
        Engine::StepOut so; so.n_rows = 4; so.rb = sampled; so.first_row = {0};
        RadBatch batch{}; batch.n_seq = 1; batch.n_spec = 3;
        CHECK(engine.collect_step(&batch, so) == RAD_OK);
        CHECK(kv->rollback_calls == 0); // Collector must never free pipelined KV.
        CHECK(so.accepted[0] == accepted && so.token[0] == 77 && so.eos[0] == 0);
        Scheduler::StepResult result{}; result.n_seq = 1;
        result.n_accepted = so.accepted.data(); result.token = so.token.data(); result.eos = so.eos.data();
        CHECK(sched.commit(result) == RAD_OK);
        const bool rollback = accepted < 3 && pipeline != 2;
        CHECK(kv->rollback_calls == (rollback ? 1 : 0));
        if (rollback) CHECK(kv->last_valid == 18 + accepted);
        CHECK(s.req->n_computed == 18 + accepted);
        CHECK(s.req->output.size() == static_cast<size_t>(accepted + 2));
        CHECK(s.req->output[0] == 19 && s.req->output.back() == 77);
        for(int i=0;i<accepted;++i) CHECK(s.req->output[i+1] == static_cast<int32_t>(drafts[i]));
        sched.next_open_ = false; s.entry_next = -1;
    }
}
static void test_linear_proposal_contract() {
    uint32_t drafts[] = {31,32,33};
    CandidateTree tree; tree.build_linear(19, drafts, 3);
    uint32_t tokens[RAD_TREE_MAX_NODES]{}, positions[RAD_TREE_MAX_NODES]{};
    TreeMask masks[RAD_TREE_MAX_NODES]{};
    tree.populate_tensors(tokens, masks, positions, 137);
    for (int i=0;i<4;++i) {
        CHECK(tree[i].parent_id == i-1 && tree[i].depth == i);
        CHECK(positions[i] == static_cast<uint32_t>(137+i));
        CHECK(masks[i] == (1u << (i+1))-1);
    }
    int32_t preds[] = {31,32,33,77}, out[4]{}, accepted=-1, bonus=-1;
    CHECK(accept_tree_greedy(tree,preds,4,out,&accepted,&bonus) == RAD_OK);
    CHECK(accepted == 3 && bonus == 77);
}

// Real collect_step -> scheduler commit, for a full chain and a noncontiguous high-node path.
static void test_wide_tree_collection_commit() {
    for (bool chain : {false, true}) {
        Program program; program.ctx.max_spec = 64;
        Config cfg; Engine engine(cfg);
        auto owned = std::make_unique<MinimalFakeKV>();
        auto* kv = owned.get(); engine.kv_iface_ = std::move(owned);
        auto& sched = engine.sched_; sched.kv_ = kv; sched.prog_ = &program;
        auto req = std::make_unique<Request>(); req->id = 8501;
        req->prompt = {8, 9}; req->output = {19}; req->n_computed = 17;
        req->may_speculate = true; req->max_tokens = 200;
        CHECK(sched.add(std::move(req)) == RAD_OK);
        const int slot = sched.by_id_.at(8501); auto& s = *sched.reqs_[slot];
        s.running = true; s.req->state = ReqState::Running; s.entry = 0;
        sched.step_open_ = true;

        CandidateTree tree;
        CHECK(tree.add_node(0, -1, 19, 0));
        for (int i = 1; i < 65; ++i) {
            const int p = chain ? i - 1 : i == 31 ? 15 : i == 63 ? 31 : i == 64 ? 63 : 0;
            CHECK(tree.add_node(i, p, static_cast<uint32_t>(100 + i), tree[p].depth + 1));
        }
        CHECK(tree.node_count() == 65);
        for (int node : {15, 31, 63, 64}) {
            CHECK(tree[node].ancestor_bitmask.test(node));
            CHECK(tree[node].ancestor_bitmask.test(0));
            CHECK(tree.is_ancestor(node, 64));
        }
        CHECK(tree[64].ancestor_bitmask.count() == (chain ? 65u : 5u));
        CHECK(!tree.add_node(65, 64, 165, tree[64].depth + 1));
        CHECK(sched.set_draft_tree(8501, tree, 17) == RAD_OK);
        CHECK(s.n_draft == 64 && s.draft[63] == 164);
        CandidateTree invalid = tree; invalid.count = 66;
        CHECK(sched.set_draft_tree(8501, invalid, 17) == RAD_E_INVAL);
        ActiveTreeRecord rec;
        CHECK(TreeRegistry::instance().get(8501, &rec));
        CHECK(rec.tree.node_count() == 65 && rec.parents[64] == tree[64].parent_id);
        CHECK(rec.positions[64] == static_cast<uint32_t>(17 + tree[64].depth));

        std::vector<int32_t> expected = chain ? std::vector<int32_t>{} : std::vector<int32_t>{0,15,31,63,64};
        if (chain) for (int i = 0; i < 65; ++i) expected.push_back(i);
        const int accepted = static_cast<int>(expected.size()) - 1;
        int32_t sampled[65]; std::fill(std::begin(sampled), std::end(sampled), 777);
        for (size_t i = 1; i < expected.size(); ++i)
            sampled[expected[i-1]] = static_cast<int32_t>(tree[expected[i]].token_id);
        StepEntry e{}; e.seq = 8501; e.slot = slot; e.card = 0;
        e.ctx_len = 17; e.n_tokens = 65; e.n_spec = e.n_draft = 64;
        e.req = s.req.get(); e.draft = s.draft; e.produces_token = true;
        sched.plan_.e.push_back(e);
        // Fence both ends of the mapped layout. The test supplies storage without a device.
        std::vector<int32_t> staging(139, -909);
        sched.builder_.tree_paths_ = staging.data() + 1;
        sched.builder_.tree_path_pitch_ = 65;
        sched.builder_.span_at_ = 135;
        Engine::StepOut so; so.n_rows = 65; so.rb = sampled; so.first_row = {0};
        RadBatch batch{}; batch.n_seq = 1; batch.n_spec = 64;
        CHECK(engine.collect_step(&batch, so) == RAD_OK);
        const int32_t* paths = sched.tree_paths();
        const int32_t* lengths = paths + 65 + 2;
        const int32_t* starts = lengths + 1;
        const int32_t* absolute = starts + 1;
        CHECK(sched.tree_path_pitch() == 65 && paths[65] == 0 && paths[66] == 1);
        CHECK(lengths[0] == accepted + 1 && starts[0] == 0);
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(paths[i] == expected[i] && absolute[i] == expected[i]);
            if (!chain) CHECK(s.accepted_rows[i] == expected[i]);
        }
        if (!chain) CHECK(paths[5] == -1 && absolute[5] == -1 && s.accepted_rows_n == 5);
        CHECK(staging.front() == -909 && staging.back() == -909);
        CHECK(so.accepted[0] == accepted && so.token[0] == 777 && kv->rollback_calls == 0);
        Scheduler::StepResult result{}; result.n_seq = 1;
        result.n_accepted = so.accepted.data(); result.token = so.token.data(); result.eos = so.eos.data();
        CHECK(sched.commit(result) == RAD_OK);
        CHECK(s.req->n_computed == 18 + accepted);
        CHECK(s.req->output.size() == static_cast<size_t>(accepted + 2));
        CHECK(s.req->output.back() == 777);
        for (int i = 0; i < accepted; ++i)
            CHECK(s.req->output[i + 1] == static_cast<int32_t>(tree[expected[i + 1]].token_id));
        CHECK(!TreeRegistry::instance().get(8501, nullptr));
        sched.builder_.tree_paths_ = nullptr; // vector-owned, never pass it to device free
    }
}

static void test_recurrent_tree_guard() {
    for (int kind : {RAD_KV_LINEAR, RAD_KV_CONV, RAD_KV_FULL}) {
        Program program;
        KVGroupInfo group{}; group.decl.kind = kind;
        program.kv_groups.push_back(group);
        program.branched_tree_supported = kind != RAD_KV_FULL;
        Scheduler sched; sched.prog_ = &program;
        auto request = std::make_unique<Request>();
        request->id = 7301; request->prompt = {1}; request->may_speculate = true;
        CHECK(sched.add(std::move(request)) == RAD_OK);
        uint32_t tokens[3] = {2,3,4}; CandidateTree tree;
        tree.build_linear(1,tokens,3);
        CHECK(sched.set_draft_tree(7301,tree,10) == RAD_OK);
        CHECK(TreeRegistry::instance().get(7301,nullptr));
        tree.build_top2(1,tokens,false,3);
        if (kind == RAD_KV_FULL) {
            program.branched_tree_supported = true;
            CHECK(sched.set_draft_tree(7301,tree,10) == RAD_OK);
            CHECK(TreeRegistry::instance().get(7301,nullptr));
            program.branched_tree_supported = false;
        }
        CHECK(sched.set_draft_tree(7301,tree,10) == RAD_E_UNSUPPORTED);
        CHECK(!TreeRegistry::instance().get(7301,nullptr));
        const auto& state = *sched.reqs_[sched.by_id_.at(7301)];
        CHECK(!state.has_draft_tree && state.n_draft == 0 && state.req->draft == nullptr);
        tree.build_linear(1,tokens,3);
        CHECK(sched.set_draft_tree(7301,tree,10) == RAD_OK);
        TreeRegistry::instance().clear(7301);
    }
}

static void test_concurrent_batch_deterministic_indexing() {
    Program program;
    MinimalFakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    sched.prog_ = &program;
    sched.hit_holes_draft_ = false;
    sched.cfg_.max_tok = 512;
    CHECK(sched.draft_.configure({12, false}) == RAD_OK);

    // Create 3 requests in concurrent execution:
    // Req 1: id=8001, ctx_len=10, K=3, accepted=1 (a < K, partial rejection)
    // Req 2: id=8002, ctx_len=20, K=2, accepted=2 (a = K, full acceptance)
    // Req 3: id=8003, ctx_len=30, K=4, accepted=0 (a = 0, complete rejection)
    auto r1 = std::make_unique<Request>();
    r1->id = 8001; r1->prompt = {1, 2}; r1->output = {100}; r1->n_computed = 10;
    r1->may_speculate = true; r1->state = ReqState::Running;
    CHECK(sched.add(std::move(r1)) == RAD_OK);
    const int slot1 = sched.by_id_.at(8001);
    auto& s1 = *sched.reqs_[slot1]; s1.running = true; s1.req->state = ReqState::Running; s1.entry = 0;

    auto r2 = std::make_unique<Request>();
    r2->id = 8002; r2->prompt = {3, 4}; r2->output = {200}; r2->n_computed = 20;
    r2->may_speculate = true;
    CHECK(sched.add(std::move(r2)) == RAD_OK);
    const int slot2 = sched.by_id_.at(8002);
    auto& s2 = *sched.reqs_[slot2]; s2.running = true; s2.req->state = ReqState::Running; s2.entry = 1;

    auto r3 = std::make_unique<Request>();
    r3->id = 8003; r3->prompt = {5, 6}; r3->output = {300}; r3->n_computed = 30;
    r3->may_speculate = true;
    CHECK(sched.add(std::move(r3)) == RAD_OK);
    const int slot3 = sched.by_id_.at(8003);
    auto& s3 = *sched.reqs_[slot3]; s3.running = true; s3.req->state = ReqState::Running; s3.entry = 2;

    int32_t draft1[] = {101, 102, 103};
    int32_t draft2[] = {201, 202};
    int32_t draft3[] = {301, 302, 303, 304};
    CHECK(sched.set_draft(8001, draft1, 3) == RAD_OK);
    CHECK(sched.set_draft(8002, draft2, 2) == RAD_OK);
    CHECK(sched.set_draft(8003, draft3, 4) == RAD_OK);

    StepEntry e1{}; e1.seq = 8001; e1.slot = slot1; e1.card = 0;
    e1.ctx_len = 10; e1.n_tokens = 4; e1.n_spec = 3; e1.n_draft = 3;
    e1.req = s1.req.get(); e1.draft = s1.draft; e1.produces_token = true;

    StepEntry e2{}; e2.seq = 8002; e2.slot = slot2; e2.card = 1;
    e2.ctx_len = 20; e2.n_tokens = 3; e2.n_spec = 2; e2.n_draft = 2;
    e2.req = s2.req.get(); e2.draft = s2.draft; e2.produces_token = true;

    StepEntry e3{}; e3.seq = 8003; e3.slot = slot3; e3.card = 2;
    e3.ctx_len = 30; e3.n_tokens = 5; e3.n_spec = 4; e3.n_draft = 4;
    e3.req = s3.req.get(); e3.draft = s3.draft; e3.produces_token = true;

    sched.plan_.e = {e1, e2, e3};
    sched.running_ = {slot1, slot2, slot3};
    sched.step_open_ = true;

    int32_t accepted[] = {1, 2, 0};
    int32_t tokens[] = {999, 888, 777};
    uint8_t eos[] = {0, 0, 0};
    Scheduler::StepResult res{};
    res.n_seq = 3;
    res.n_accepted = accepted;
    res.token = tokens;
    res.eos = eos;

    CHECK(sched.commit(res) == RAD_OK);

    CHECK(s1.req->n_accepted == 1);
    CHECK(s1.req->n_computed == 12);
    CHECK(s1.req->output.back() == 999);

    CHECK(s2.req->n_accepted == 2);
    CHECK(s2.req->n_computed == 23);
    CHECK(s2.req->output.back() == 888);

    CHECK(s3.req->n_accepted == 0);
    CHECK(s3.req->n_computed == 31);
    CHECK(s3.req->output.back() == 777);

    Scheduler::BlockRow rows[3];
    int n_rows = sched.block_begin(13, 12, 0, 99, rows, 3);
    CHECK(n_rows == 3);

    CHECK(rows[0].seq == 8001);
    CHECK(rows[0].tap_row == 1);
    CHECK(rows[0].anchor_token == 999);
    CHECK(rows[0].root_pos == 12);

    CHECK(rows[1].seq == 8002);
    CHECK(rows[1].tap_row == 6);
    CHECK(rows[1].anchor_token == 888);
    CHECK(rows[1].root_pos == 23);

    CHECK(rows[2].seq == 8003);
    CHECK(rows[2].tap_row == 7);
    CHECK(rows[2].anchor_token == 777);
    CHECK(rows[2].root_pos == 31);

    // The query pass reads these through StepEntry::src_row -> RadBatch::draft_out_ids.
    CHECK(sched.block_.size() == 3);
    CHECK(sched.block_[0].tap_row == 1);
    CHECK(sched.block_[1].tap_row == 6);
    CHECK(sched.block_[2].tap_row == 7);
}

static void test_block_device_row_patches() {
    struct Slab {
        int32_t tok[24], pos[24], ctx[3], dout[3], sid[3], cu[3], q[3], rows[12];
    } slab{};
    std::fill(slab.tok, slab.tok + 24, 99);
    std::fill(slab.pos, slab.pos + 24, 137);
    std::fill(slab.ctx, slab.ctx + 3, 137);
    std::fill(slab.dout, slab.dout + 3, -1);
    slab.sid[0] = 2; slab.sid[1] = 0; slab.sid[2] = 1;
    slab.cu[0] = 2; slab.cu[1] = 9; slab.cu[2] = 16;
    std::fill(slab.q, slab.q + 3, 7);
    slab.rows[0] = 1; slab.rows[1] = 1; slab.rows[2] = 0;
    slab.rows[3] = 0; slab.rows[4] = 4; slab.rows[5] = 7;
    StepSlot slots[3]{};
    slots[2].acc = 0; slots[2].tok[0] = 0;
    slots[0].acc = 2; slots[0].tok[0] = 123;
    slots[1].acc = 4; slots[1].tok[0] = 456;
    const Slab before = slab;
    StepSlot slots_before[3];
    std::memcpy(slots_before, slots, sizeof(slots));
    AdvanceArgs args{};
    args.mode = ADV_BLOCK; args.n_seq = 3; args.row_pitch = 3;
    args.slab = reinterpret_cast<char*>(&slab); args.st = slots;
    args.tok_off = offsetof(Slab, tok); args.pos_off = offsetof(Slab, pos);
    args.ctx_off = offsetof(Slab, ctx); args.dout_off = offsetof(Slab, dout);
    args.sid_off = offsetof(Slab, sid); args.cu_off = offsetof(Slab, cu);
    args.q_off = offsetof(Slab, q); args.row_off = offsetof(Slab, rows);
    CHECK(ADV_BLOCK == 6);
    for (int i = 0; i < 3; ++i) adv_row(args, i);
    CHECK(slab.tok[2] == 0 && slab.tok[9] == 123);
    CHECK(slab.dout[0] == 0 && slab.dout[1] == 6 && slab.dout[2] == -1);
    Slab expected = before;
    expected.tok[2] = 0; expected.tok[9] = 123;
    expected.dout[0] = 0; expected.dout[1] = 6;
    CHECK(std::memcmp(&slab, &expected, sizeof(slab)) == 0);
    CHECK(std::memcmp(slots, slots_before, sizeof(slots)) == 0);
}

static void test_block_precommit_and_committed_anchor() {
    struct RecordingKV : MinimalFakeKV {
        std::vector<std::pair<uint64_t, int64_t>> reservations;
        int ensure(uint64_t seq, int64_t end) override {
            reservations.emplace_back(seq, end); return RAD_OK;
        }
    } kv;
    Scheduler sched;
    sched.kv_ = &kv; sched.hit_holes_draft_ = false; sched.cfg_.max_tok = 128;
    CHECK(sched.draft_.configure({12, false}) == RAD_OK);
    for (int i = 0; i < 2; ++i) {
        auto req = std::make_unique<Request>();
        req->id = 8101 + i; req->prompt = {17}; req->n_computed = 20 + 10 * i;
        req->may_speculate = true;
        CHECK(sched.add(std::move(req)) == RAD_OK);
        const int slot = sched.by_id_.at(8101 + i);
        auto& s = *sched.reqs_[slot];
        s.running = true; s.req->state = ReqState::Running;
        StepEntry e{}; e.seq = 8101 + i; e.slot = slot;
        e.ctx_len = 20 + 10 * i; e.n_spec = i == 0 ? 3 : 2;
        e.n_tokens = 1 + e.n_spec; e.produces_token = true;
        sched.plan_.e.push_back(e);
    }
    sched.plan_.phase = RAD_PHASE_DECODE;
    Scheduler::BlockRow rows[2];
    CHECK(sched.block_begin(7, 12, 0, 99, rows, 2, true) == 2);
    CHECK(kv.reservations.size() == 2);
    CHECK(kv.reservations[0].first == 8101 && kv.reservations[0].second == 36);
    CHECK(kv.reservations[1].first == 8102 && kv.reservations[1].second == 45);
    CHECK(sched.block_[0].dev && sched.block_[0].dev_tok == 0);
    CHECK(sched.block_[1].dev && sched.block_[1].dev_tok == 4);
    for (const auto& row : rows)
        CHECK(row.anchor_token == -1 && row.root_pos == -1 && row.tap_row == -1);
    kv.reservations.clear();
    CHECK(sched.block_begin(13, 12, 0, 99, rows, 2, true) == 2);
    CHECK(kv.reservations.size() == 2);
    CHECK(kv.reservations[0].second == 37 && kv.reservations[1].second == 46);
    kv.reservations.clear();
    sched.ctx_limit_ = 44;
    CHECK(sched.block_begin(7, 12, 0, 99, rows, 2, true) == 1);
    CHECK(kv.reservations.size() == 1 && kv.reservations[0].second == 36);
    sched.ctx_limit_ = 0;
    kv.reservations.clear();
    sched.plan_.phase = RAD_PHASE_PREFILL;
    CHECK(sched.block_begin(7, 12, 0, 99, rows, 2, true) == 0);
    sched.plan_.phase = RAD_PHASE_DECODE;
    sched.plan_.e[1].is_prefill = true;
    CHECK(sched.block_begin(7, 12, 0, 99, rows, 2, true) == 0);
    sched.plan_.e[1].is_prefill = false;
    sched.plan_.e[1].produces_token = false;
    CHECK(sched.block_begin(7, 12, 0, 99, rows, 2, true) == 0);
    CHECK(kv.reservations.empty() && sched.block_.empty());
    const int slot = sched.by_id_.at(8101);
    auto& req = *sched.reqs_[slot]->req;
    int32_t token = -1; int64_t pos = -1;
    CHECK(!sched.block_committed_anchor(8101, slot, &token, &pos));
    req.output = {99, 0}; req.n_computed = 24;
    CHECK(sched.block_committed_anchor(8101, slot, &token, &pos));
    CHECK(token == 0 && pos == 24);
    req.state = ReqState::Finished;
    CHECK(!sched.block_committed_anchor(8101, slot, &token, &pos));
    req.state = ReqState::Running; req.id = 8199;
    CHECK(!sched.block_committed_anchor(8101, slot, &token, &pos));
    CHECK(sched.block_committed_anchor(8199, slot, &token, &pos));
    CHECK(!sched.block_committed_anchor(8199, -1, &token, &pos));
    req.id = 8101;
}

static void test_block_depth_rejection_invariance() {
    for (bool tree : {false, true}) {
        SchedReq req;
        req.has_draft_tree = tree;
        const int cap = tree ? 5 : 12;
        for (int accepted : {12, 0, 0, 0, 0, 0}) {
            req.observe_draft(12, accepted);
            CHECK(req.suggest_depth(12, true, false, cap) == 8);
            CHECK(req.suggest_depth(12, true, true, cap) == 4);
            CHECK(req.suggest_depth(3, true, false, cap) == 3);
            CHECK(req.suggest_depth(3, true, true, cap) == 3);
            CHECK(req.suggest_depth(12, false, false, cap) == 8);
            CHECK(req.suggest_depth(12, false, true, cap) == 4);
        }
    }
    SchedReq small;
    CHECK(small.suggest_depth(3, false, false, 5) == 3);
    CHECK(small.suggest_depth(3, false, true, 5) == 3);
}

static void test_serial_history_keeps_the_verified_root_after_its_lead() {
    MinimalFakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    CHECK(sched.draft_.configure({3, false}) == RAD_OK);
    const int32_t paths[2][3] = {{0, 3, -1}, {0, 1, 4}};
    for (int i = 0; i < 2; ++i) {
        auto req = std::make_unique<Request>();
        req->id = 8201 + i;
        req->prompt = {17, 18};
        req->n_computed = i == 0 ? 14 : 23;
        req->may_speculate = true;
        CHECK(sched.add(std::move(req)) == RAD_OK);
        const int slot = sched.by_id_.at(8201 + i);
        auto& s = *sched.reqs_[slot];
        s.running = true;
        s.req->state = ReqState::Running;
        sched.running_.push_back(slot);
        StepEntry e{};
        e.seq = 8201 + i; e.slot = slot; e.card = 1 - i;
        e.ctx_len = i == 0 ? 10 : 20;
        e.n_tokens = i == 0 ? 7 : 5;
        e.n_spec = 4; e.produces_token = true; e.dev_out = i * 5;
        sched.plan_.e.push_back(e);
        sched.accepted_leaf(e.seq, i == 0 ? 3 : 4, paths[i], i + 2);
    }
    int rounds = 3;
    Scheduler::SerialRow rows[2];
    CHECK(sched.serial_begin(&rounds, rows, 2, true) == 2);
    CHECK(sched.serial_[0].tap_rows == std::vector<int32_t>({0, 1, 2, 5}));
    CHECK(sched.serial_[1].tap_rows == std::vector<int32_t>({7, 8, 11}));
    CHECK(sched.serial_[0].hist_pos == 10 && sched.serial_[0].hist_n == 4);
    CHECK(sched.serial_[0].base_pos == 13 && !sched.serial_[0].dev);
    CHECK(sched.serial_[1].base_pos == 22 && !sched.serial_[1].dev);

    /* Pre-commit history must ignore those host paths and gather the current card path once. */
    CHECK(sched.serial_begin(&rounds, rows, 2) == 2);
    CHECK(sched.serial_[0].tap_rows.empty() && sched.serial_[1].tap_rows.empty());
    struct Slab {
        int32_t tok[12], pos[12], out[12], ctx[2], sid[2], cu[3], q[2], rows[8];
    } slab{};
    slab.sid[0] = 1; slab.sid[1] = 0;
    slab.cu[1] = 7; slab.cu[2] = 12;
    slab.q[0] = 7; slab.q[1] = 5;
    slab.rows[0] = slab.rows[1] = 1;
    slab.rows[4] = 0; slab.rows[5] = 5;
    slab.rows[6] = 2;
    slab.tok[0] = 101; slab.tok[1] = 102;
    for (int i = 0; i < 12; ++i) slab.out[i] = i;
    StepSlot slots[2]{};
    slots[1].head = 13; slots[1].acc = 1;
    slots[0].head = 22; slots[0].acc = 2;
    for (int i = 0; i < 3; ++i) {
        slots[1].path[i] = paths[0][i];
        slots[0].path[i] = paths[1][i];
    }
    const int32_t sampled[] = {31, -1, -1, 41, -1, 21, 32, -1, -1, 42};
    AdvanceArgs args{};
    args.mode = ADV_HIST; args.n_seq = 2; args.row_pitch = 2;
    args.slab = reinterpret_cast<char*>(&slab); args.st = slots; args.src = sampled;
    args.tok_off = offsetof(Slab, tok); args.pos_off = offsetof(Slab, pos);
    args.out_off = offsetof(Slab, out); args.ctx_off = offsetof(Slab, ctx);
    args.sid_off = offsetof(Slab, sid); args.cu_off = offsetof(Slab, cu);
    args.q_off = offsetof(Slab, q); args.row_off = offsetof(Slab, rows);
    for (int i = 0; i < 2; ++i) adv_row(args, i);
    const int32_t expected_tokens[] = {101, 102, 31, 41, 41, 41, 41, 21, 32, 42, 42, 42};
    const int32_t expected_taps[] = {0, 1, 2, 5, 5, 5, 5, 7, 8, 11, 11, 11};
    const int32_t expected_positions[] = {10, 11, 12, 13, 13, 13, 13, 20, 21, 22, 22, 22};
    for (int i = 0; i < 12; ++i) {
        CHECK(slab.tok[i] == expected_tokens[i]);
        CHECK(slab.out[i] == expected_taps[i]);
        CHECK(slab.pos[i] == expected_positions[i]);
    }
}

static void test_serial_tree_imports_the_device_topology() {
    Engine engine(Config{});
    engine.ranks_.push_back(std::make_unique<Engine::Rank>());
    auto& rank = *engine.ranks_[0];
    rank.program.drafter_tree_buf = 1;
    rank.program.drafter_tree_capacity = 8;
    engine.drafter_.name = "mtp";
    engine.drafter_.kind = RAD_DRAFT_SERIAL;
    engine.drafter_.depth = 3;
    engine.meta_.n_vocab = 256;
    engine.serial_tree_ = true;
    engine.serial_rounds_ = engine.serial_done_ = 3;
    engine.serial_rows_ = 1;
    auto& sched = engine.sched_;
    CHECK(sched.draft_.configure({7, false}) == RAD_OK);
    auto req = std::make_unique<Request>();
    req->id = 8301; req->prompt = {8}; req->output = {9}; req->n_computed = 20;
    req->may_speculate = true; req->max_tokens = 100;
    CHECK(sched.add(std::move(req)) == RAD_OK);
    const int slot = sched.by_id_.at(8301);
    auto& s = *sched.reqs_[slot];
    s.running = true; s.req->state = ReqState::Running;
    Scheduler::SerialRow sr{};
    sr.seq = 8301; sr.slot = slot; sr.card = 0;
    engine.serial_row_.push_back(sr);
    StepSlot slots[1]{};
    slots[0].tok[0] = 9;
    slots[0].tok[1] = 10; slots[0].tok[2] = 20; slots[0].tok[3] = 30;
    engine.slots_host_ = slots;
    int32_t packed[] = {8, 9, 10, 20, 30, 11, 12, 21, 22, -1, 0, 1, 2, 0, 0, 1, 1};
    engine.readback_.p = packed;
    engine.readback_.n = 17;
    Engine::StepOut so;
    CHECK(engine.finish_serial_drafter(so) == RAD_OK);
    ActiveTreeRecord rec;
    CHECK(TreeRegistry::instance().get(8301, &rec));
    CHECK(rec.tree.node_count() == 8 && rec.positions[3] == 23);
    for (int i = 0; i < 8; ++i) {
        CHECK(rec.tree[i].token_id == static_cast<uint32_t>(packed[1 + i]));
        CHECK(rec.tree[i].parent_id == packed[9 + i]);
    }
    engine.tree_exact_to_ = 23;
    CHECK(engine.finish_serial_drafter(so) == RAD_OK);
    CHECK(TreeRegistry::instance().get(8301, &rec) && rec.tree.node_count() == 4);
    CHECK(s.n_draft == 3 && s.draft[0] == 10 && s.draft[1] == 20 && s.draft[2] == 30);

    /* A shorter next step ignores stale tail entries; malformed live data fails, never rebuilds. */
    engine.serial_done_ = 1;
    packed[0] = 2;
    CHECK(engine.finish_serial_drafter(so) == RAD_OK);
    CHECK(TreeRegistry::instance().get(8301, &rec) && rec.tree.node_count() == 2);
    packed[1] = 99;
    CHECK(engine.finish_serial_drafter(so) == RAD_E_STATE);
    packed[1] = 9; packed[10] = 2;
    CHECK(engine.finish_serial_drafter(so) == RAD_E_STATE);
    packed[10] = 0; packed[0] = 9;
    CHECK(engine.finish_serial_drafter(so) == RAD_E_STATE);
    engine.readback_.p = nullptr;
    engine.readback_.n = 0;
    engine.slots_host_ = nullptr;
    engine.ranks_.clear();
    TreeRegistry::instance().clear(8301);
}

// The verifier's coupled root draw is supplied here; sampler math lives in sample_test.
static void test_spechub_accept_commit_and_terminal_ordinary_path() {
    for (int scenario = 0; scenario < 4; ++scenario) {
        // Winner row 2; EOG at root; EOG at bonus; root draw names a truncated sibling.
        Program program; program.branched_tree_supported = true;
        Engine engine(Config{});
        engine.spechub_ = true; engine.meta_.n_vocab = 100;
        auto owned = std::make_unique<MinimalFakeKV>();
        auto* kv = owned.get(); engine.kv_iface_ = std::move(owned);
        auto& sched = engine.sched_; sched.kv_ = kv; sched.prog_ = &program;
        CHECK(sched.draft_.configure({3, false}) == RAD_OK);
        if (scenario == 1 || scenario == 2) {
            VocabBuild build;
            for (int i = 0; i < 100; ++i) build.tokens.push_back(std::to_string(i));
            build.eos = scenario == 1 ? 32 : 77;
            auto vocab = std::make_shared<Vocab>();
            CHECK(vocab->load(std::move(build)) == RAD_OK);
            engine.vocab_ = std::move(vocab);
        }
        auto req = std::make_unique<Request>();
        req->id = 8401; req->prompt = {8, 9}; req->output = {19};
        req->n_computed = 17; req->may_speculate = true;
        req->max_tokens = scenario == 3 ? 3 : 4;
        CHECK(sched.add(std::move(req)) == RAD_OK);
        const int slot = sched.by_id_.at(8401);
        auto& s = *sched.reqs_[slot];
        s.running = true; s.req->state = ReqState::Running; s.entry = 0;
        sched.running_.push_back(slot); sched.step_open_ = true;
        const uint32_t pair[] = {31, 32};
        CandidateTree tree; tree.build_top2(19, pair, false, 2);
        CHECK(sched.set_draft_tree(8401, tree, 17) == RAD_OK);
        const int n_spec = scenario == 3 ? 1 : 2;
        CHECK(s.n_draft == n_spec);
        ActiveTreeRecord retained;
        CHECK(TreeRegistry::instance().get(8401, &retained));
        CHECK(retained.tree.node_count() == static_cast<size_t>(n_spec + 1));
        CHECK(s.draft[0] == 31);
        StepEntry e{}; e.seq = 8401; e.slot = slot; e.card = 0;
        e.ctx_len = 17; e.n_tokens = n_spec + 1; e.n_spec = n_spec;
        e.n_draft = n_spec; e.req = s.req.get(); e.draft = s.draft;
        e.produces_token = true; sched.plan_.e.push_back(e);
        struct Slab {
            int32_t tok[3], pos[3], ctx[1], sid[1], cu[2], q[1], rows[4];
        } slab{};
        slab.tok[0] = 19; slab.tok[1] = 31; slab.tok[2] = 32;
        slab.pos[0] = 17; slab.pos[1] = slab.pos[2] = 18;
        slab.ctx[0] = 17; slab.cu[1] = slab.q[0] = n_spec + 1;
        slab.rows[0] = 1; slab.rows[3] = n_spec;
        const int32_t parents[] = {-1, 0, 0};
        int32_t sampled[] = {32, 66, 77}; // Rejected row 1 must never supply the bonus.
        // Scenario 3 supplies the full-pair verifier's draw even though row 2 was truncated.
        StepSlot slots[1]{};
        std::fill(std::begin(slots[0].path), std::end(slots[0].path), -1);
        AdvanceArgs args{};
        args.mode = ADV_ACCEPT; args.n_seq = 1; args.row_pitch = 1;
        args.slab = reinterpret_cast<char*>(&slab); args.st = slots;
        args.src = sampled; args.spec_parent = parents;
        args.tok_off = offsetof(Slab, tok); args.pos_off = offsetof(Slab, pos);
        args.ctx_off = offsetof(Slab, ctx); args.sid_off = offsetof(Slab, sid);
        args.cu_off = offsetof(Slab, cu); args.q_off = offsetof(Slab, q);
        args.row_off = offsetof(Slab, rows);
        if (engine.vocab_) {
            args.eog = engine.vocab_->eog().data();
            args.n_eog = static_cast<int32_t>(engine.vocab_->eog().size());
        }
        adv_row(args, 0);
        const int accepted = scenario == 1 || scenario == 3 ? 0 : 1;
        const int bonus = accepted == 0 ? 32 : 77;
        CHECK(slots[0].acc == accepted && slots[0].n_acc == accepted + 1);
        CHECK(slots[0].head == (accepted == 0 ? 17 : 18));
        CHECK(slots[0].ctx == (accepted == 0 ? 18 : 19));
        CHECK(slots[0].tok[0] == bonus && slots[0].path[0] == 0);
        CHECK(slots[0].path[1] == (accepted == 0 ? -1 : 2));
        CHECK(slots[0].path[2] == -1);
        Engine::StepOut so;
        so.n_rows = n_spec + 1; so.rb = sampled; so.first_row = {0};
        RadBatch batch{}; batch.n_seq = 1; batch.n_spec = n_spec;
        CHECK(engine.collect_step(&batch, so) == RAD_OK);
        CHECK(so.accepted == std::vector<int32_t>({accepted}));
        CHECK(so.token == std::vector<int32_t>({bonus}));
        CHECK(so.eos[0] == (scenario == 1 || scenario == 2 ? 1 : 0));
        CHECK(kv->rollback_calls == 0);
        CHECK(s.accepted_rows_n == (scenario == 3 ? 0 : accepted + 1));
        if (scenario != 3) {
            CHECK(s.accepted_rows[0] == 0);
            if (accepted) CHECK(s.accepted_rows[1] == 2);
        }
        Scheduler::StepResult result{};
        result.n_seq = 1; result.n_accepted = so.accepted.data();
        result.token = so.token.data(); result.eos = so.eos.data();
        CHECK(sched.commit(result) == RAD_OK);
        CHECK(s.req->output == (accepted == 0 ? std::vector<int32_t>({19, 32})
                                             : std::vector<int32_t>({19, 32, 77})));
        CHECK(s.req->n_computed == slots[0].ctx && s.req->n_accepted == accepted);
        CHECK(kv->rollback_calls == 1 && kv->last_valid == slots[0].ctx);
        CHECK(!TreeRegistry::instance().get(8401, nullptr));
        if (scenario == 1 || scenario == 2)
            CHECK(s.req->state == ReqState::Finished);
        if (scenario != 0) continue;

        // Exactly one output remains: retain history, but propose no draft.
        CHECK(s.req->state == ReqState::Running && s.req->n_computed == 19);
        int rounds = 3;
        Scheduler::SerialRow row{};
        s.req->max_tokens = 5;
        CHECK(sched.serial_begin(&rounds, &row, 1, true) == 1);
        s.req->max_tokens = 4;
        CHECK(sched.serial_begin(&rounds, &row, 1, true) == 0);
        CHECK(sched.serial_.size() == 1 && sched.serial_[0].draft_idx == -1);
        CHECK(sched.serial_[0].hist_pos == 17 && sched.serial_[0].hist_n == 2);
        CHECK(sched.serial_[0].base_pos == 18 && !sched.serial_[0].dev);
        CHECK(sched.serial_[0].tap_rows == std::vector<int32_t>({0, 2}));
        sched.serial_.clear();
        e.ctx_len = 19; e.n_tokens = 1; e.n_spec = e.n_draft = 0;
        e.draft = nullptr; sched.plan_.e = {e};
        sched.step_open_ = true; s.entry = 0;
        int32_t ordinary = 88;
        Engine::StepOut last;
        last.n_rows = 1; last.rb = &ordinary; last.first_row = {0};
        batch.n_spec = 0;
        CHECK(engine.collect_step(&batch, last) == RAD_OK);
        CHECK(last.accepted[0] == 0 && last.token[0] == 88 && last.eos[0] == 0);
        result.n_accepted = last.accepted.data();
        result.token = last.token.data(); result.eos = last.eos.data();
        CHECK(sched.commit(result) == RAD_OK);
        CHECK(s.req->output == std::vector<int32_t>({19, 32, 77, 88}));
        CHECK(s.req->n_computed == 20 && s.req->state == ReqState::Finished);
        CHECK(kv->rollback_calls == 1);
    }
}

// RADIANCE_MTP_COUPLED's engine half: rank 0's state readback is bound to the committed verify
// positions, and every broken proposal path refuses the step instead of verifying a Q that did
// not produce the draft. Sampler binding and the math live in sample_test and spechub_test.
static void test_mtp_couple_binding_after_commit() {
    Engine engine(Config{});
    engine.mtp_couple_ = true; engine.meta_.n_vocab = 100; engine.couple_rows_ = 2;
    RadMtpCoupleDraw draws[2] = {{7, 40, 8501, 3, 1}, {9, 12, 8502, 5, 0}};
    RadMtpCoupleState plane[3 * 2] = {};   // [rounds][rows], as the 2D readback lays it out
    StepSlot S{};
    S.acc = 2; S.n_acc = 3;
    const int32_t toks[4] = {11, 21, 31, 41};   // bonus, then each round's card-side draft
    for (int i = 0; i < 4; ++i) S.tok[i] = toks[i];
    for (int rd = 1; rd <= 3; ++rd) {
        RadMtpCoupleState& s = plane[(rd - 1) * 2];
        s.draw = draws[0]; s.status = 1; s.round = rd; s.x = S.tok[rd]; s.count = 2;
        for (int k = 0; k < 8; ++k) s.ids[k] = -1;
        s.ids[0] = S.tok[rd]; s.ids[1] = 60; s.q[0] = 0.75; s.q[1] = 0.25;
        RadMtpCoupleState& g = plane[(rd - 1) * 2 + 1];
        g.draw = draws[1]; g.status = 0; g.round = rd; g.x = S.tok[rd];   // greedy keeps column 0
        for (int k = 0; k < 8; ++k) g.ids[k] = -1;
    }
    engine.couple_draw_stage_ = draws; engine.couple_readback_ = plane;
    CHECK(engine.bind_couple(0, 3, S) == RAD_OK);
    for (int rd = 1; rd <= 3; ++rd) {
        // Committed output size is base + n_acc; round rd verifies rd - 1 positions after it.
        CHECK(engine.couple_bound_[rd - 1].pos == 40 + 3 + (uint64_t)(rd - 1));
        CHECK(engine.couple_bound_[rd - 1].status == 1 && engine.couple_bound_[rd - 1].x == S.tok[rd]);
    }
    CHECK(engine.bind_couple(1, 3, S) == RAD_OK && engine.couple_bound_[0].status == 0);
    for (int which = 0; which < 6; ++which) {
        RadMtpCoupleState saved[6];
        std::copy(plane, plane + 6, saved);
        int row = 0;
        if (which == 0) plane[2].x = 99;              // round 2's x is not the card's draft
        if (which == 1) plane[0].draw.base = 39;      // a previous step's state
        if (which == 2) plane[4].status = 0;          // a stochastic row bypassed the proposal op
        if (which == 3) plane[0].status = -1;         // malformed shortlist or row identity
        if (which == 4) plane[2].round = 3;
        if (which == 5) { plane[1].status = 1; row = 1; }   // a greedy row claims a sample
        CHECK(engine.bind_couple(row, 3, S) == RAD_E_STATE);
        std::copy(saved, saved + 6, plane);
    }
    engine.couple_draw_stage_ = nullptr; engine.couple_readback_ = nullptr;
    engine.mtp_couple_ = false;
}

int main() {
    test_wide_tree_collection_commit();
    test_spechub_accept_commit_and_terminal_ordinary_path();
    test_mtp_couple_binding_after_commit();
    test_serial_history_keeps_the_verified_root_after_its_lead();
    test_serial_tree_imports_the_device_topology();
    test_block_device_row_patches();
    test_block_precommit_and_committed_anchor();
    test_block_depth_rejection_invariance();
    test_concurrent_batch_deterministic_indexing();
    test_recurrent_tree_guard();
    test_rollback_owner_and_emission(); test_linear_proposal_contract();
    test_block_anchor(); test_tree_publication(); test_token_ids_are_not_eos();
    test_short_candidate_buffers(); test_unwired_sidecar_rejected();
    std::puts("PASS: 16 drafter CPU regression groups (no device initialization)");
}
