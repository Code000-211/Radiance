/* tree_sampling_path_test.cpp -- per-node sampler conditioning over a speculative tree.
 *
 * THE CONTRACT UNDER TEST. When Request carries tree parents/depths (root included), the row the
 * sampler stages for node j is conditioned on the ANCESTOR PATH of j: its history is the
 * committed prompt+output plus that path's tokens, its RNG coordinate is n_out + depth(j), its
 * reasoning budget is walked over the path. The flattened row index is not a conditioning -- for
 * two siblings it names the OTHER branch's history. The oracle for every row is host_sample over
 * the same serial continuation: prompt, output, path -- what a non-speculative decode conditioned
 * on that prefix would draw -- and the real chain (Sampler::build_step + run on libref) must
 * return it token for token. A linear topology must stage exactly what the flat draft always
 * staged, and the scheduler must publish, validate and retire the topology through the same
 * lifecycle the draft window has.
 */
#include "rad_test.h"

#include "sample/accept.h"
#include "sample/grammar.h"
#include "sample/host_ref.h"
#include "sample/sampler.h"
#include "sample/vocab_view.h"

#include "build/rad_build.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "rad_sample.h"

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

/* The std prelude + diag.h BEFORE the define: scheduler.h pulls json.hpp under the access
 * rewrite otherwise, and the poisoned redeclaration of std internals is the wall of "redeclared
 * with different access" the dartree tests solved the same way. */
#include <sstream>
#include <any>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "diag.h"
#define private public
#include "sched/scheduler.h"
#undef private

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;
using radiance::CandidateTree;

/* ================================================================== the host chain
 *
 * The real sampler declared through the real builder and run by libref on the host domain, the
 * same harness sample_test.cpp uses: only the staging AND the issued chain together prove which
 * token a row produces, and a defect in either is invisible to the other. */
namespace {

std::string chain_home() {
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}

struct HostChain {
    Registry reg;
    Program  prog;
    Ctx      ctx;
    Sampler  s;
    rad_buf  logits = RAD_NULL_HANDLE, tokens = RAD_NULL_HANDLE;
    int64_t  nv = 0;
    bool     ok = false;
    int      st = RAD_OK;

    static inline HostChain* cur = nullptr;
    static void trampoline(RadCtx* c, const RadBatch*) {
        if (cur) cur->st = cur->s.run(c, cur->logits, cur->tokens);
    }

    HostChain(int64_t n_vocab, int64_t max_seqs, int max_spec, int64_t max_hist = 64,
              const VocabView* vocab = nullptr)
        : nv(n_vocab) {
        if (reg.load_plugin(chain_home() + "/kernels/libref.so", 0) < 0) return;
        RadModelMeta meta{};
        meta.arch_id = "tree_sampling";
        meta.name    = "tree_sampling";
        meta.n_vocab = n_vocab;
        RadBuildCtx bc{};
        bc.rank = 0;
        bc.world_size = 1;
        bc.max_seqs = max_seqs;
        bc.max_tok = max_seqs * (max_spec + 1);
        bc.max_ctx = 512;
        bc.max_spec = max_spec;
        bc.scope = "";
        Builder b(reg, meta, bc);

        const int64_t rows = max_seqs * (max_spec + 1);
        RadBufDecl d{};
        d.dtype = RAD_F32;
        d.rank = 2;
        d.shape[0] = rows;
        d.shape[1] = n_vocab;
        d.kind = RAD_BUF_PERSIST;
        d.domain = RAD_DOMAIN_DEVICE;
        logits = rad_decl_buffer(&b, "logits", &d);
        d.dtype = RAD_I32;
        d.shape[1] = 1;
        tokens = rad_decl_buffer(&b, "tokens", &d);

        SamplerConfig cfg;
        cfg.n_vocab = n_vocab;
        cfg.n_vocab_local = n_vocab;
        cfg.max_seqs = max_seqs;
        cfg.max_spec = max_spec;
        cfg.max_hist = max_hist;
        cfg.max_candidates = 8;
        if (s.declare(&b, cfg) < 0) return;
        if (b.finish() < 0) return;
        prog = std::move(b.program());
        if (vocab) s.attach_vocab(vocab);

        CtxDesc cd;
        cd.program = &prog;
        cd.arch_step = &HostChain::trampoline;
        cd.record_passes = false;
        if (ctx.init(cd) < 0) return;
        for (size_t i = 1; i < prog.ops.size(); ++i)
            if (ctx.set_op_domain((rad_op)i, RAD_DOMAIN_HOST) < 0) return;
        ok = true;
    }

    int step(const std::vector<Request*>& reqs, const std::vector<float>& lg,
             std::vector<int32_t>* out) {
        for (Request* r : reqs) {
            if (s.has_request(r->id)) continue;
            std::string err;
            const int b = s.begin_request(r->id, r->sp, &err);
            if (b < 0) return b;
        }
        const int bs = s.build_step(reqs);
        if (bs < 0) return bs;
        int64_t n = 0;
        s.staged_params(&n);
        if ((int64_t)lg.size() != n * nv) return RAD_E_SHAPE;
        std::memcpy(ctx.buf_ptr(logits), lg.data(), lg.size() * sizeof(float));
        int32_t* t = (int32_t*)ctx.buf_ptr(tokens);
        for (int64_t i = 0; i < n; ++i) t[i] = -7;

        RadBatch batch{};
        batch.phase = RAD_PHASE_DECODE;
        batch.n_tok = n;
        batch.n_seq = n;
        cur = this;
        st = RAD_OK;
        const int rs = ctx.run_step(&batch);
        cur = nullptr;
        if (rs < 0) return rs;
        if (st < 0) return st;
        rad_stream_sync(ctx.stream());
        out->assign(t, t + n);
        return RAD_OK;
    }
};

#define NEED_CHAIN(ch)                                                                \
    if (!(ch).ok) {                                                                   \
        fprintf(stderr, "  SKIP %s: libref did not load from %s/kernels\n",           \
                ::radtest::current(), chain_home().c_str());                          \
        return;                                                                       \
    }

Request make_req(uint64_t id, std::vector<int32_t> prompt, const SamplingParams& sp) {
    Request r;
    r.id = id;
    r.prompt = std::move(prompt);
    r.sp = sp;
    return r;
}

/* The node arrays a published tree hands the Request: parents and depths, root included. */
struct NodeMeta {
    int32_t parent[radiance::RAD_TREE_MAX_NODES + 1] = {}; // one extra row for refused capacity
    int32_t depth[radiance::RAD_TREE_MAX_NODES + 1]  = {};
    int32_t tokens[radiance::RAD_TREE_MAX_NODES + 1] = {}; /* entry0 is the committed anchor */
    int     n = 0;
};

static NodeMeta meta_of(const CandidateTree& t, int n_draft) {
    NodeMeta m;
    m.n = n_draft + 1;
    for (int j = 0; j <= n_draft; ++j) {
        m.parent[j] = t[j].parent_id;
        m.depth[j]  = t[j].depth;
        m.tokens[j] = (int32_t)t[j].token_id;
    }
    return m;
}

/* Attach the topology to a Request the way the scheduler's publication does. */
static void attach(Request& r, const NodeMeta& m, int32_t* scratch_tok) {
    for (int j = 1; j < m.n; ++j) scratch_tok[j - 1] = m.tokens[j];
    r.n_draft = m.n - 1;
    r.draft = scratch_tok;
    r.tree_parent = m.parent;
    r.tree_depth = m.depth;
}

/* The serial-prefix oracle for node j: history = prompt ++ output ++ the ancestor path (root's
 * child down to j, j included), RNG position = n_out + depth(j). */
static int32_t oracle(const Request& r, const NodeMeta& m, int j, const float* lg, int64_t nv) {
    std::vector<int32_t> path;
    int32_t cur = (int32_t)j;
    while (m.depth[cur] > 0) {
        path.push_back(m.tokens[cur]);
        cur = m.parent[cur];
    }
    std::reverse(path.begin(), path.end());
    std::vector<int32_t> hist = r.prompt;
    hist.insert(hist.end(), r.output.begin(), r.output.end());
    hist.insert(hist.end(), path.begin(), path.end());

    HostRefInput in;
    in.history = hist.data();
    in.n_history = (int64_t)hist.size();
    in.pos = (uint64_t)r.output.size() + (uint64_t)m.depth[j];
    return host_sample(lg + (int64_t)j * nv, nv, r.sp, in);
}

}  /* namespace */

/* ================================================================== the chain draws the path's token
 *
 * A 16-node Top-2 Mode A tree (nodes 1,2 siblings; 3,4 children of 1; 5,6 of 2; ...). Logits vary
 * per row and the repetition penalty is strong enough that the token each row draws depends on
 * which tokens its own path put in the history; the seed varies so the pick coordinate matters
 * too. Every row must equal the serial-prefix oracle for ITS node. */
TEST(tree_rows_draw_their_own_paths_answer) {
    HostChain ch(6, 1, 15);
    NEED_CHAIN(ch)

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_top2(7u, cands, false, 15);
    const NodeMeta m = meta_of(tree, 15);

    int discriminated = 0;
    for (uint64_t seed : {1ull, 42ull, 777ull}) {
        SamplingParams sp;
        sp.temp = 1.0f;
        sp.top_k = 0;
        sp.seed = seed;
        sp.rep_penalty = 8.0f;
        sp.penalty_last_n = -1;
        Request a = make_req(100 + seed, { 3, 4 }, sp);
        a.output = { 5, 2 };
        int32_t toks[16];
        attach(a, m, toks);

        std::vector<float> lg(16 * 6);
        for (int row = 0; row < 16; ++row)
            for (int v = 0; v < 6; ++v)
                lg[(size_t)row * 6 + (size_t)v] = 0.4f * (float)((row * 7 + v * 3) % 6) - 1.0f;

        std::vector<int32_t> tok;
        CHECK_OK(ch.step({ &a }, lg, &tok));
        int64_t n = 0;
        const DeviceSampleParams* p = ch.s.staged_params(&n);
        CHECK_EQ(n, (int64_t)16);
        for (int j = 0; j < 16; ++j) {
            CHECK_EQ(p[j].pos, (uint64_t)(2 + (uint64_t)m.depth[j]));
            CHECK_EQ(p[j].hist_len, (int32_t)(4 + m.depth[j]));
            const int32_t want = oracle(a, m, j, lg.data(), 6);
            CHECK_EQ(tok[(size_t)j], want);
            ++discriminated;
        }
        ch.s.end_request(a.id);
    }
    CHECK(discriminated == 48);
}

/* Greedy rows too (temperature 0): the argmax runs after the penalties, so the path decides which
 * token survives. */
TEST(greedy_tree_rows_are_penalised_along_their_own_path) {
    HostChain ch(6, 1, 15);
    NEED_CHAIN(ch)

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_top4(7u, cands, 15);
    const NodeMeta m = meta_of(tree, 15);

    SamplingParams sp;
    sp.temp = 0.0f;
    sp.rep_penalty = 8.0f;
    sp.penalty_last_n = -1;
    Request a = make_req(1, { 3 }, sp);
    a.output = { 4 };
    int32_t toks[16];
    attach(a, m, toks);

    std::vector<float> lg(16 * 6, 0.0f);
    for (int row = 0; row < 16; ++row) lg[(size_t)row * 6 + 1] = 2.0f;   /* token 1 wins raw */

    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    for (int j = 0; j < 16; ++j)
        CHECK_EQ(tok[(size_t)j], oracle(a, m, j, lg.data(), 6));
}

/* A SIBLING IS NOT CONDITIONED ON ITS PHYSICAL PREDECESSOR. Star: root with children 1 and 2
 * (both depth 1). The flattened rule gives node 2 the history [committed, token(1)]; the path
 * rule gives [committed, token(2)]. Token 1 tops every row's logits and the penalty is strong, so
 * the two histories draw different tokens: the chain must draw the PATH's answer, and the test
 * proves it can tell the two apart. */
TEST(a_sibling_is_conditioned_on_its_own_path_not_the_flattened_prefix) {
    HostChain ch(4, 1, 2);
    NEED_CHAIN(ch)

    NodeMeta m;
    m.n = 3;
    m.parent[0] = -1; m.depth[0] = 0; m.tokens[0] = 9;    /* the anchor, already committed */
    m.parent[1] = 0;  m.depth[1] = 1; m.tokens[1] = 1;    /* sibling A proposes token 1 */
    m.parent[2] = 0;  m.depth[2] = 1; m.tokens[2] = 2;    /* sibling B proposes token 2 */

    SamplingParams sp;
    sp.temp = 0.0f;
    sp.rep_penalty = 10.0f;
    sp.penalty_last_n = -1;
    Request a = make_req(1, { 3 }, sp);
    int32_t toks[16];
    attach(a, m, toks);

    std::vector<float> lg(3 * 4, 0.0f);
    for (int row = 0; row < 3; ++row) {
        lg[(size_t)row * 4 + 0] = 1.0f;
        lg[(size_t)row * 4 + 1] = 3.0f;   /* token 1 wins raw */
    }

    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));

    /* oracle for node 2 with its own path [2]: token 1 unpunished -> 1 */
    std::vector<int32_t> hist2 = { 3, 2 };
    HostRefInput in2;
    in2.history = hist2.data();
    in2.n_history = 2;
    in2.pos = 1;
    const int32_t want_path = host_sample(lg.data() + 2 * 4, 4, sp, in2);
    CHECK_EQ(want_path, 1);
    CHECK_EQ(tok[2], 1);

    /* and with the FLATTENED prefix [1, 2]: token 1 is in the history, the penalty buries it */
    std::vector<int32_t> hist_flat = { 3, 1, 2 };
    HostRefInput inf;
    inf.history = hist_flat.data();
    inf.n_history = 3;
    inf.pos = 2;
    const int32_t want_flat = host_sample(lg.data() + 2 * 4, 4, sp, inf);
    CHECK(want_flat != 1);                  /* the test can discriminate the two rules */
    CHECK_EQ(tok[2], want_path);            /* ... and the chain took the path's */
}

/* A LINEAR TOPOLOGY STAGES EXACTLY THE FLAT DRAFT. Same request, same logits, two ids: one carries
 * the metadata for a chain, one carries none. Every staged field and every drawn token must
 * agree -- the metadata must be a no-op for the one shape that was already correct. */
TEST(linear_metadata_reproduces_the_flat_staging_exactly) {
    HostChain ch(6, 1, 5);
    NEED_CHAIN(ch)

    NodeMeta lin;
    lin.n = 5;                              /* root + a chain of 4 */
    lin.parent[0] = -1; lin.depth[0] = 0; lin.tokens[0] = 9;
    for (int j = 1; j < 5; ++j) {
        lin.parent[j] = j - 1;
        lin.depth[j] = j;
        lin.tokens[j] = 10 + j;
    }

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;
    sp.seed = 9;
    sp.rep_penalty = 6.0f;
    sp.penalty_last_n = -1;

    Request flat = make_req(1, { 3, 4 }, sp);
    flat.output = { 1 };
    flat.n_draft = 4;
    int32_t fd[4] = { 11, 12, 13, 14 };
    flat.draft = fd;

    Request tree = make_req(2, { 3, 4 }, sp);
    tree.output = { 1 };
    int32_t td[16];
    attach(tree, lin, td);

    std::vector<float> lg(5 * 6);
    for (size_t i = 0; i < lg.size(); ++i)
        lg[i] = 0.3f * (float)((i * 5) % 7) - 1.0f;

    std::vector<int32_t> t1, t2;
    CHECK_OK(ch.step({ &flat }, lg, &t1));
    CHECK_OK(ch.step({ &tree }, lg, &t2));
    int64_t n1 = 0, n2 = 0;
    CHECK_OK(ch.s.build_step({ &flat }));
    const DeviceSampleParams* staged_flat = ch.s.staged_params(&n1);
    REQUIRE_EQ(n1, (int64_t)5);
    /* build_step reuses the staging: keep VALUES, not a pointer the tree build overwrites. */
    DeviceSampleParams p1[5];
    std::copy_n(staged_flat, 5, p1);
    CHECK_OK(ch.s.build_step({ &tree }));
    const DeviceSampleParams* p2 = ch.s.staged_params(&n2);
    CHECK_EQ(n2, (int64_t)5);
    for (int j = 0; j < 5; ++j) {
        CHECK_EQ(p1[j].pos, p2[j].pos);
        CHECK_EQ(p1[j].hist_len, p2[j].hist_len);
        CHECK_EQ(p1[j].flags, p2[j].flags);
        CHECK_EQ(p1[j].seed, p2[j].seed);
        CHECK_EQ(p1[j].hist_off, p2[j].hist_off);
    }
    for (int j = 0; j < 5; ++j) CHECK_EQ(t1[(size_t)j], t2[(size_t)j]);
}

/* THE HISTORY TRUNCATES TO THE DECLARED WINDOW ALONG THE PATH, as it always did along the chain:
 * max_hist 8, so a deep row keeps the TAIL of its (committed ++ path) history, and the oracle over
 * that same tail is what the chain draws. */
TEST(a_deep_tree_row_truncates_the_history_window_along_its_path) {
    HostChain ch(6, 1, 15, /*max_hist=*/8);
    NEED_CHAIN(ch)

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_linear(7u, cands, 15);           /* deep chain: node 15 is at depth 15 */
    tree.add_node(0, -1, 7u, 0);                /* (rebuild guard) */
    tree.reset();
    tree.build_linear(7u, cands, 15);
    /* branch the last node off node 1 so the deepest path is not the flat prefix */
    CandidateTree branched;
    branched.add_node(0, -1, 7u, 0);
    for (int i = 1; i <= 14; ++i)
        branched.add_node(i, tree[i].parent_id, tree[i].token_id, tree[i].depth);
    branched.add_node(15, 1, cands[14], 2);     /* node 15: depth 2, path [10, 24] */
    const NodeMeta m = meta_of(branched, 15);

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.seed = 5;
    sp.rep_penalty = 4.0f;
    sp.penalty_last_n = -1;
    Request a = make_req(1, { 3, 4, 5, 6, 0, 1 }, sp);
    a.output = { 2, 3, 4, 5 };
    int32_t toks[16];
    attach(a, m, toks);

    std::vector<float> lg(16 * 6);
    for (size_t i = 0; i < lg.size(); ++i)
        lg[i] = 0.25f * (float)((i * 3) % 11) - 1.2f;

    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a }, lg, &tok));
    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    /* node 15: committed 9 tokens + path 2 = 11 > max_hist 8 -> the tail of the path's history */
    CHECK_EQ(p[15].hist_len, (int32_t)8);
    for (int j = 0; j < 16; ++j) {
        /* the oracle takes the same tail: build the full path history, keep the last 8 */
        std::vector<int32_t> path;
        int32_t cur = (int32_t)j;
        while (m.depth[cur] > 0) { path.push_back(m.tokens[cur]); cur = m.parent[cur]; }
        std::reverse(path.begin(), path.end());
        std::vector<int32_t> hist = a.prompt;
        hist.insert(hist.end(), a.output.begin(), a.output.end());
        hist.insert(hist.end(), path.begin(), path.end());
        if ((int64_t)hist.size() > 8) hist.erase(hist.begin(), hist.end() - 8);
        HostRefInput in;
        in.history = hist.data();
        in.n_history = (int64_t)hist.size();
        in.pos = (uint64_t)a.output.size() + (uint64_t)m.depth[j];
        CHECK_EQ(tok[(size_t)j], host_sample(lg.data() + (int64_t)j * 6, 6, sp, in));
    }
}

/* MALFORMED TOPOLOGY IS REFUSED BEFORE ANY ROW IS STAGED. Each case returns RAD_E_INVAL and stages
 * nothing for the request (a second, clean request in the same step proves the staging stopped at
 * zero rows rather than half-way through the bad one). */
TEST(malformed_tree_topology_is_refused_before_staging) {
    HostChain ch(6, 1, 15);
    NEED_CHAIN(ch)

    NodeMeta good;
    good.n = 3;
    good.parent[0] = -1; good.depth[0] = 0; good.tokens[0] = 9;
    good.parent[1] = 0;  good.depth[1] = 1; good.tokens[1] = 1;
    good.parent[2] = 0;  good.depth[2] = 1; good.tokens[2] = 2;

    struct Case { const char* name; NodeMeta m; };
    std::vector<Case> cases;

    { NodeMeta m = good; m.parent[0] = 0;  cases.push_back({ "root parent not -1", m }); }
    { NodeMeta m = good; m.depth[0] = 1;   cases.push_back({ "root depth not 0", m }); }
    { NodeMeta m = good; m.parent[2] = 2;  cases.push_back({ "self parent", m }); }
    { NodeMeta m = good; m.parent[2] = 5;  cases.push_back({ "parent past the window", m }); }
    { NodeMeta m = good; m.parent[2] = -1; cases.push_back({ "non-root parent -1", m }); }
    { NodeMeta m = good; m.depth[2] = 2;   cases.push_back({ "depth skips", m }); }
    { NodeMeta m = good; m.tokens[2] = 1;  cases.push_back({ "duplicate siblings", m }); }
    { NodeMeta m = good; m.n = 66;
      for (int j = 3; j < 66; ++j) { m.parent[j] = 0; m.depth[j] = 1; m.tokens[j] = 20 + j; }
      cases.push_back({ "past the65-node budget", m }); }

    for (const Case& c : cases) {
        Request a = make_req(1, { 3 }, SamplingParams());
        int32_t toks[radiance::RAD_TREE_MAX_NODES + 1];
        attach(a, c.m, toks);
        CHECK_EQ(ch.s.build_step({ &a }), RAD_E_INVAL);
        int64_t n = -1;
        ch.s.staged_params(&n);
        CHECK_EQ(n, (int64_t)0);
    }

    /* the tokens themselves absent */
    Request b = make_req(1, { 3 }, SamplingParams());
    b.n_draft = 2;
    b.draft = nullptr;
    b.tree_parent = good.parent;
    b.tree_depth = good.depth;
    CHECK_EQ(ch.s.build_step({ &b }), RAD_E_INVAL);

    /* a clean request still stages after all of that */
    Request ok_req = make_req(2, { 3 }, SamplingParams());
    CHECK_OK(ch.s.build_step({ &ok_req }));
    int64_t n = 0;
    ch.s.staged_params(&n);
    CHECK_EQ(n, (int64_t)1);
}

/* A GRAMMAR REFUSES A BRANCHED WINDOW AND WALKS A LINEAR ONE. The refusal is explicit -- nothing
 * runs, the step reports RAD_E_UNSUPPORTED -- while a linear topology under the same grammar is
 * served with the grammar's mask, exactly as the flat draft always was. */
TEST(a_grammar_refuses_a_branched_window_and_walks_a_linear_one) {
    SimpleVocab v({ "a", "b", "c", "d" });
    HostChain ch(4, 1, 2, 64, &v);
    NEED_CHAIN(ch)

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.seed = 3;
    sp.grammar = "root ::= \"a\"+";

    NodeMeta branch;
    branch.n = 3;
    branch.parent[0] = -1; branch.depth[0] = 0; branch.tokens[0] = 0;
    branch.parent[1] = 0;  branch.depth[1] = 1; branch.tokens[1] = 0;
    branch.parent[2] = 0;  branch.depth[2] = 1; branch.tokens[2] = 1;

    Request a = make_req(1, { 0 }, sp);
    int32_t toks[16];
    attach(a, branch, toks);
    CHECK_OK(ch.s.begin_request(a.id, a.sp, nullptr));
    CHECK(!ch.s.allows_branching(a.id));
    CHECK_EQ(ch.s.build_step({ &a }), RAD_E_UNSUPPORTED);
    ch.s.end_request(a.id);

    NodeMeta lin = branch;
    lin.parent[2] = 1;
    lin.depth[2] = 2;
    lin.tokens[2] = 0;
    Request b = make_req(2, { 0 }, sp);
    attach(b, lin, toks);
    CHECK_OK(ch.s.begin_request(b.id, b.sp, nullptr));
    CHECK_OK(ch.s.build_step({ &b }));
    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    CHECK_EQ(n, (int64_t)3);
    for (int j = 0; j < 3; ++j) CHECK((p[j].flags & RAD_SP_MASK) != 0);
}

/* TWO REQUESTS, ONE STEP: a tree window beside a plain decode. Each keeps its own rows -- the
 * tree's rows follow their paths, the plain row follows its own single position. */
TEST(a_tree_window_and_a_plain_row_share_a_step) {
    HostChain ch(6, 2, 15);
    NEED_CHAIN(ch)

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(20 + i);
    CandidateTree tree;
    tree.build_top2(7u, cands, false, 15);
    const NodeMeta m = meta_of(tree, 15);

    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;
    sp.seed = 31;
    sp.rep_penalty = 5.0f;
    sp.penalty_last_n = -1;

    Request a = make_req(1, { 3 }, sp);
    a.output = { 4 };
    int32_t toks[16];
    attach(a, m, toks);

    SamplingParams sp2;
    sp2.temp = 1.0f;
    sp2.seed = 77;
    Request b = make_req(2, { 5, 1 }, sp2);

    std::vector<float> lg((16 + 1) * 6);
    for (size_t i = 0; i < lg.size(); ++i)
        lg[i] = 0.35f * (float)((i * 7) % 5) - 0.8f;

    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ &a, &b }, lg, &tok));
    for (int j = 0; j < 16; ++j)
        CHECK_EQ(tok[(size_t)j], oracle(a, m, j, lg.data(), 6));
    /* the plain row's answer is its own: token 4's logit dominates every row it appears in */
    CHECK_EQ(tok[16], oracle(b, NodeMeta{}, 0, lg.data() + 16 * 6, 6));
}

/* ================================================================== the scheduler's publication
 *
 * THE SCHEDULER PUBLISHES, VALIDATES AND RETIRES THE TOPOLOGY. A branched tree reaches the Request
 * with root-included parents/depths beside the draft tokens; a linear one publishes nothing (the
 * flat path); a malformed or duplicate-sibling tree is refused with NOTHING of the request
 * touched; and every path that rewrites the window -- a linear redraft, an empty tree, a lost
 * speculation right, an unsupported branch, a commit -- drops the topology with it. */
namespace {
struct FakeKV : public rad::IKVManager {
    int add_sequence(uint64_t) override { return RAD_OK; }
    bool has_sequence(uint64_t) const override { return true; }
    void free_sequence(uint64_t) override {}
    int ensure(uint64_t, int64_t) override { return RAD_OK; }
    int adopt(uint64_t, int32_t, const std::vector<int32_t>&, int64_t) override { return RAD_OK; }
    int preempt(uint64_t) override { return RAD_OK; }
    bool can_ever_fit(int64_t) const override { return true; }
    const std::vector<int32_t>& block_table(uint64_t, int32_t) const override {
        static const std::vector<int32_t> e;
        return e;
    }
    int32_t state_slot(uint64_t, int32_t) const override { return -1; }
    int state_index_row(uint64_t, int32_t, int32_t*, int) const override { return 0; }
    int64_t state_index_width(int32_t) const override { return 0; }
    int64_t free_blocks(int32_t) const override { return 4096; }
    int64_t total_blocks(int32_t) const override { return 4096; }
    int64_t held_blocks(int32_t) const override { return 0; }
    int64_t pool_bytes(int32_t) const override { return 1 << 20; }
    int64_t pool_bytes_used(int32_t) const override { return 0; }
    int64_t pool_bytes_carved(int32_t) const override { return 1 << 20; }
    int64_t block_tokens(int32_t) const override { return 16; }
    int rollback(uint64_t, int64_t) override { return RAD_OK; }
    int64_t first_block_pos(uint64_t, int32_t) const override { return 0; }
    int64_t take_table_changes(uint64_t, int32_t) const override { return 0; }
};

struct EmptyPC : IPrefixCache {
    CacheHit lookup(const Request&, std::vector<std::vector<int32_t>>*) const override { return {}; }
    const std::vector<int32_t>& cached_groups() const override {
        static const std::vector<int32_t> empty;
        return empty;
    }
    int insert(const Request&, const std::vector<std::vector<int32_t>>&, int64_t) override {
        return RAD_OK;
    }
    int64_t evict(int64_t) override { return 0; }
    int64_t evictions() const override { return 0; }
    int64_t ckpt_evictions() const override { return 0; }
    int64_t held_tokens() const override { return 0; }
    int64_t next_checkpoint_after(int64_t) const override { return -1; }
    int reserve_checkpoint(const Request&, int64_t, int32_t*) override { return RAD_E_FULL; }
    void commit_checkpoint(const Request&, int64_t) override {}
};

/* Real admission, planning, batch upload and commit; no KV groups are needed to inspect rows. */
struct ScheduledWindow {
    FakeKV kv;
    EmptyPC pc;
    Program prog;
    Scheduler sched;
    RadStream stream = nullptr;

    ~ScheduledWindow() {
        radiance::TreeRegistry::instance().clear(5757);
        sched.fini();
        if (stream) rad_stream_destroy(stream);
    }

    int init(int depth, int max_ctx, bool tree_state = false, int sequences = 1) {
        prog.branched_tree_supported = true;
        Config cfg;
        cfg.max_seqs = sequences;
        cfg.max_tok = std::max(16, depth + 1) * sequences;
        cfg.max_ctx = max_ctx;
        cfg.n_spec = depth;
        cfg.prefix_cache = false;
        if (tree_state) {
            OpInfo commit;
            commit.op = "gdn_tree_commit";
            prog.ops.push_back(commit);
        }
        RAD_TRY(rad_stream_create(&stream, 0));
        RAD_TRY(sched.init(prog, cfg, &kv, &pc, {{0, stream, &prog}}));
        /* Legacy clamp tests pin adaptation; tree-state tests exercise real initialization. */
        return tree_state ? RAD_OK : sched.draft_.configure({depth, false});
    }

    int prefill(const SamplingParams& sp, int32_t anchor) {
        auto req = std::make_unique<Request>(make_req(5757, {anchor}, sp));
        req->max_tokens = 100;
        req->may_speculate = true;
        RAD_TRY(sched.add(std::move(req)));
        if (!sched.step()) return RAD_E_STATE;
        Scheduler::StepResult res{};
        res.n_seq = 1;
        res.token = &anchor;
        return sched.commit(res);
    }
};
}  /* namespace */

TEST(a_truncated_sibling_window_keeps_batch_sampling_and_acceptance_on_one_prefix) {
    HostChain ch(4, 1, 3);
    NEED_CHAIN(ch)
    for (bool context_clamp : {false, true}) {
        ScheduledWindow w;
        REQUIRE_OK(w.init(context_clamp ? 3 : 2, context_clamp ? 4 : 64));
        SamplingParams sp;
        sp.temp = 0.0f;
        sp.seed = 123;
        sp.rep_penalty = 10.0f;
        sp.penalty_last_n = -1;
        REQUIRE_OK(w.prefill(sp, 3));
        Request* r = w.sched.find(5757);
        REQUIRE(r);

        CandidateTree tree;
        tree.add_node(0, -1, 3u, 0);
        tree.add_node(1, 0, 1u, 1);
        tree.add_node(2, 0, 2u, 1);
        tree.add_node(3, 2, 1u, 2);  /* removed by either clamp, despite matching row 2's draw */
        REQUIRE_OK(w.sched.set_draft_tree(r->id, tree, (uint32_t)r->n_computed));
        REQUIRE_EQ(r->n_draft, 3);
        const int32_t* parents = r->tree_parent;
        const int32_t* depths = r->tree_depth;
        REQUIRE(parents && depths);

        const RadBatch* batch = w.sched.step();
        REQUIRE(batch);
        REQUIRE_OK(rad_stream_sync(w.stream));
        REQUIRE_EQ(w.sched.plan().e.size(), (size_t)1);
        const StepEntry& e = w.sched.plan().e[0];
        REQUIRE(e.req == r);
        CHECK_EQ(e.n_spec, 2);
        CHECK_EQ(e.n_tokens, 3);
        CHECK_EQ(batch->n_out, (int64_t)3);
        CHECK_EQ(r->n_draft, 2);
        CHECK(r->tree_parent == parents && r->tree_depth == depths);
        CHECK_EQ(batch->token_ids[2], 2);
        CHECK_EQ(batch->positions[1], e.ctx_len + 1);
        CHECK_EQ(batch->positions[2], e.ctx_len + 1);
        CHECK_EQ(batch->spec_parent[2], 0);
        CHECK_EQ(batch->tree_ancestors[2], (uint16_t)0b101);

        /* Root selects sibling 2; its own path leaves token 1 unpenalised. Flattening
         * siblings adds token 1 to that history and instead draws token 0. */
        std::vector<float> lg = {0, 0, 10, 0, 1, 3, 0, 0, 1, 3, 0, 0};
        std::vector<int32_t> tok;
        REQUIRE_OK(ch.step({r}, lg, &tok));
        REQUIRE_EQ(tok.size(), (size_t)3);
        CHECK_EQ(tok[0], 2);
        CHECK_EQ(tok[2], 1);
        const int32_t flat_hist[] = {3, 3, 1, 2};
        HostRefInput flat;
        flat.history = flat_hist;
        flat.n_history = 4;
        flat.pos = 3;
        CHECK_EQ(host_sample(lg.data() + 8, 4, sp, flat), 0);
        const NodeMeta m = meta_of(tree, 2);
        for (int j = 0; j < 3; ++j)
            CHECK_EQ(tok[(size_t)j], oracle(*r, m, j, lg.data(), 4));

        /* The registry may retain the proposal, but acceptance must not follow node 3,
         * which has no forwarded/sample row in this window. This is the engine's acceptor. */
        radiance::ActiveTreeRecord rec;
        REQUIRE(radiance::TreeRegistry::instance().get(e.seq, &rec));
        CHECK_EQ(rec.tree.node_count(), (size_t)4);
        int32_t out[4] = {}, nodes[4] = {}, accepted = -1, bonus = -1;
        REQUIRE_OK(accept_tree_greedy(rec.tree, tok.data(), e.n_tokens,
                                    out, &accepted, &bonus, nodes));
        CHECK_EQ(accepted, 1);
        CHECK_EQ(nodes[1], 2);
        CHECK_EQ(out[0], 2);
        CHECK_EQ(out[1], 1);
        CHECK_EQ(bonus, 1);

        r->sp.temp = 1.0f;
        for (size_t i = 0; i < lg.size(); ++i) lg[i] = 0.25f * (float)(i % 4);
        REQUIRE_OK(ch.step({r}, lg, &tok));
        int64_t n = 0;
        const DeviceSampleParams* p = ch.s.staged_params(&n);
        REQUIRE_EQ(n, (int64_t)3);
        for (int j = 0; j < 3; ++j) {
            CHECK_EQ(p[j].pos, (uint64_t)r->output.size() + (uint64_t)m.depth[j]);
            CHECK_EQ(p[j].hist_len, (int32_t)(r->prompt.size() + r->output.size()) + m.depth[j]);
            CHECK_EQ(tok[(size_t)j], oracle(*r, m, j, lg.data(), 4));
        }
        CHECK_EQ(p[1].pos, p[2].pos);  /* sibling RNG coordinates are depth, not row index */
    }
}

TEST(a_clamped_tree_vetoes_grammar_only_when_the_retained_prefix_branches) {
    SimpleVocab v({"a", "b", "c", "d"});
    HostChain ch(4, 1, 3, 64, &v);
    NEED_CHAIN(ch)
    for (int retained : {2, 1, 0}) {
        ScheduledWindow w;
        REQUIRE_OK(w.init(retained, 64));
        SamplingParams sp;
        sp.grammar = "root ::= \"a\"+";
        REQUIRE_OK(w.prefill(sp, 0));
        Request* r = w.sched.find(5757);
        REQUIRE(r);
        REQUIRE_OK(ch.s.begin_request(r->id, r->sp, nullptr));
        REQUIRE(ch.s.allows_speculation(r->id));
        CandidateTree tree;
        tree.add_node(0, -1, 0u, 0);
        tree.add_node(1, 0, 0u, 1);
        tree.add_node(2, 0, 1u, 1);
        tree.add_node(3, 2, 0u, 2);
        REQUIRE_OK(w.sched.set_draft_tree(r->id, tree, (uint32_t)r->n_computed));
        REQUIRE(r->tree_parent && r->tree_depth);
        const RadBatch* batch = w.sched.step();
        REQUIRE(batch);
        REQUIRE_EQ(w.sched.plan().e.size(), (size_t)1);
        REQUIRE(w.sched.plan().e[0].req == r);
        Request* planned = r;
        CHECK_EQ(planned->n_draft, retained);
        CHECK_EQ(batch->n_out, (int64_t)retained + 1);
        CHECK_EQ(planned->tree_parent != nullptr, retained > 1);
        CHECK_EQ(planned->tree_depth != nullptr, retained > 1);
        int64_t n = -1;
        if (retained > 1) {
            CHECK_EQ(ch.s.build_step({planned}), RAD_E_UNSUPPORTED);
            ch.s.staged_params(&n);
            CHECK_EQ(n, (int64_t)0);
        } else {
            std::vector<float> lg((size_t)(retained + 1) * 4, 0.0f);
            std::vector<int32_t> tok;
            REQUIRE_OK(ch.step({planned}, lg, &tok));
            const DeviceSampleParams* p = ch.s.staged_params(&n);
            REQUIRE_EQ(n, (int64_t)retained + 1);
            for (int j = 0; j <= retained; ++j) {
                CHECK((p[j].flags & RAD_SP_MASK) != 0);
                CHECK_EQ(tok[(size_t)j], 0);
            }
        }
        ch.s.end_request(r->id);
    }
}

TEST(a_prefill_window_drops_a_published_tree_before_sampling) {
    ScheduledWindow w;
    REQUIRE_OK(w.init(3, 64));
    SimpleVocab v({"a", "b", "c", "d"});
    HostChain ch(4, 1, 3, 64, &v);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.grammar = "root ::= \"a\"+";
    auto req = std::make_unique<Request>(make_req(5757, {0, 0, 0}, sp));
    req->max_tokens = 100;
    req->may_speculate = true;
    REQUIRE_OK(w.sched.add(std::move(req)));
    CandidateTree tree;
    tree.add_node(0, -1, 0u, 0);
    tree.add_node(1, 0, 0u, 1);
    tree.add_node(2, 0, 1u, 1);
    REQUIRE_OK(w.sched.set_draft_tree(5757, tree, 0));
    const RadBatch* batch = w.sched.step();
    REQUIRE(batch);
    REQUIRE_EQ(w.sched.plan().e.size(), (size_t)1);
    const StepEntry& e = w.sched.plan().e[0];
    Request* r = w.sched.find(e.seq);
    REQUIRE(r && e.req == r);
    CHECK(e.is_prefill);
    CHECK_EQ(e.n_spec, 0);
    CHECK_EQ(batch->n_out, (int64_t)1);
    CHECK_EQ(e.req->n_draft, 0);
    CHECK(!e.req->tree_parent && !e.req->tree_depth);
    CHECK(!radiance::TreeRegistry::instance().get(e.seq, nullptr));
    std::vector<int32_t> tok;
    REQUIRE_OK(ch.step({r}, std::vector<float>(4, 0.0f), &tok));
    REQUIRE_EQ(tok.size(), (size_t)1);
    CHECK_EQ(tok[0], 0);
}

TEST(set_draft_tree_publishes_the_node_topology_root_included) {
    FakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    auto req = std::make_unique<Request>();
    req->id = 1; req->prompt = { 5 }; req->may_speculate = true; req->max_tokens = 100;
    CHECK_OK(sched.add(std::move(req)));
    const int slot = sched.by_id_.at(1);
    auto& s = *sched.reqs_[(size_t)slot];

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_top2(5u, cands, false, 15);
    CHECK_OK(sched.set_draft_tree(1, tree, 17));

    REQUIRE(s.req->tree_parent && s.req->tree_depth);
    CHECK_EQ(s.req->tree_parent[0], -1);
    CHECK_EQ(s.req->tree_depth[0], 0);
    CHECK_EQ(s.req->n_draft, 15);
    for (int j = 1; j <= 15; ++j) {
        CHECK_EQ(s.req->tree_parent[j], tree[j].parent_id);
        CHECK_EQ(s.req->tree_depth[j], tree[j].depth);
        CHECK_EQ(s.req->draft[j - 1], (int32_t)tree[j].token_id);
    }
    CHECK(s.tree_branched);

    /* a LINEAR tree publishes no topology: the flat draft the sampler always had */
    CandidateTree lin;
    lin.build_linear(5u, cands, 15);
    CHECK_OK(sched.set_draft_tree(1, lin, 17));
    CHECK(s.req->tree_parent == nullptr && s.req->tree_depth == nullptr);
    CHECK_EQ(s.req->n_draft, 15);
    CHECK(!s.tree_branched);
}

TEST(set_draft_tree_refuses_bad_topology_without_touching_the_request) {
    FakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    auto req = std::make_unique<Request>();
    req->id = 1; req->prompt = { 5 }; req->may_speculate = true; req->max_tokens = 100;
    CHECK_OK(sched.add(std::move(req)));
    const int slot = sched.by_id_.at(1);
    auto& s = *sched.reqs_[(size_t)slot];

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree good;
    good.build_top2(5u, cands, false, 15);
    CHECK_OK(sched.set_draft_tree(1, good, 17));
    const int32_t* parent_before = s.req->tree_parent;
    const int32_t* depth_before = s.req->tree_depth;
    CHECK(s.req->tree_parent != nullptr);

    /* duplicate siblings: two children of the root proposing one token */
    CandidateTree dup;
    dup.add_node(0, -1, 5u, 0);
    dup.add_node(1, 0, 10u, 1);
    dup.add_node(2, 0, 10u, 1);
    CHECK_EQ(sched.set_draft_tree(1, dup, 17), RAD_E_INVAL);
    CHECK(s.req->tree_parent == parent_before && s.req->tree_depth == depth_before);
    CHECK_EQ(s.req->n_draft, 15);
    CHECK(sched.reqs_[(size_t)slot]->n_draft == 15);

    /* hand-written malformed nodes (TreeNode is public): a future parent, a depth skip */
    CandidateTree bad;
    bad.add_node(0, -1, 5u, 0);
    bad.add_node(1, 0, 10u, 1);
    bad.nodes[1].parent_id = 3;
    bad.nodes[1].depth = 1;
    CHECK_EQ(sched.set_draft_tree(1, bad, 17), RAD_E_INVAL);
    bad.nodes[1].parent_id = 0;
    bad.nodes[1].depth = 2;
    CHECK_EQ(sched.set_draft_tree(1, bad, 17), RAD_E_INVAL);
    bad.nodes[1].depth = 1;
    bad.nodes[0].parent_id = 0;
    CHECK_EQ(sched.set_draft_tree(1, bad, 17), RAD_E_INVAL);
    CHECK(s.req->tree_parent == parent_before);

    /* the good window still stands after all the refusals */
    CHECK(s.req->tree_parent != nullptr && s.req->n_draft == 15);
}

TEST(set_draft_tree_retires_the_topology_with_the_window) {
    FakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    DraftController::Params dp;
    dp.max_depth = 16;
    CHECK_OK(sched.draft_.configure(dp));
    auto req = std::make_unique<Request>();
    req->id = 1; req->prompt = { 5 }; req->may_speculate = true; req->max_tokens = 100;
    CHECK_OK(sched.add(std::move(req)));
    const int slot = sched.by_id_.at(1);
    auto& s = *sched.reqs_[(size_t)slot];
    Request& r = *s.req;

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_top2(5u, cands, false, 15);
    CHECK_OK(sched.set_draft_tree(1, tree, 17));
    CHECK(r.tree_parent != nullptr);

    /* a linear redraft replaces the tree */
    int32_t lin[3] = { 7, 8, 9 };
    CHECK_OK(sched.set_draft(1, lin, 3));
    CHECK(r.tree_parent == nullptr && r.tree_depth == nullptr);
    CHECK_EQ(r.n_draft, 3);

    /* re-publish, then an empty tree retires it */
    CHECK_OK(sched.set_draft_tree(1, tree, 17));
    CHECK(r.tree_parent != nullptr);
    CandidateTree empty;
    CHECK_OK(sched.set_draft_tree(1, empty, 17));
    CHECK(r.tree_parent == nullptr && r.n_draft == 0);

    /* re-publish, then losing the speculation right retires it */
    CHECK_OK(sched.set_draft_tree(1, tree, 17));
    r.may_speculate = false;
    CHECK_OK(sched.set_draft_tree(1, tree, 17));
    CHECK(r.tree_parent == nullptr);
    r.may_speculate = true;

    /* re-publish, then an unsupported program retires it (a linear/conv KV group) */
    CHECK_OK(sched.set_draft_tree(1, tree, 17));
    CHECK(r.tree_parent != nullptr);
    Program prog;
    prog.branched_tree_supported = true;
    RadKVGroupDecl gd{};
    gd.kind = RAD_KV_LINEAR;
    KVGroupInfo gi;
    gi.decl = gd;
    prog.kv_groups.push_back(gi);
    sched.prog_ = &prog;
    CandidateTree lin_tree;
    lin_tree.build_linear(5u, cands, 15);
    CHECK_OK(sched.set_draft_tree(1, lin_tree, 17));   /* linear trees pass the guard */
    CHECK(r.tree_parent == nullptr);                   /* a chain is not a branch */
    CHECK_EQ(sched.set_draft_tree(1, tree, 17), RAD_E_UNSUPPORTED);
    CHECK(r.tree_parent == nullptr && r.n_draft == 0);
    sched.prog_ = nullptr;
}

/* A COMMIT RETIRES THE WINDOW, topology included -- the next step starts from the committed
 * output and whatever draft the drafter publishes next, never from this step's description. */
TEST(a_commit_drops_the_topology_before_a_prompt_lookup_window) {
    ScheduledWindow w;
    REQUIRE_OK(w.init(3, 64));
    HostChain ch(4, 1, 3);
    NEED_CHAIN(ch)
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.rep_penalty = 10.0f;
    sp.penalty_last_n = -1;
    REQUIRE_OK(w.prefill(sp, 3));
    Request* r = w.sched.find(5757);
    REQUIRE(r);
    CandidateTree tree;
    tree.add_node(0, -1, 3u, 0);
    tree.add_node(1, 0, 1u, 1);
    tree.add_node(2, 0, 2u, 1);
    tree.add_node(3, 1, 3u, 2);
    REQUIRE_OK(w.sched.set_draft_tree(r->id, tree, (uint32_t)r->n_computed));
    REQUIRE(w.sched.step());
    REQUIRE(r->tree_parent);
    auto& s = *w.sched.reqs_[(size_t)w.sched.by_id_.at(r->id)];

    /* The collector emits path 1 -> 3 and a bonus 1, and retires its registry entry. */
    const int32_t target[] = {1, 3, 0, 1};
    int32_t out[4] = {}, accepted = -1, bonus = -1;
    REQUIRE_OK(accept_tree_greedy(tree, target, 4, out, &accepted, &bonus));
    REQUIRE_EQ(accepted, 2);
    std::copy_n(out, accepted, s.draft);
    radiance::TreeRegistry::instance().clear(r->id);
    Scheduler::StepResult res{};
    res.n_seq = 1;
    res.n_accepted = &accepted;
    res.token = &bonus;
    REQUIRE_OK(w.sched.commit(res));
    CHECK(!r->tree_parent && !r->tree_depth);
    CHECK_EQ(r->n_draft, 0);
    CHECK(!s.has_draft_tree);

    /* History is 3,3,1,3,1: prompt lookup proposes the earlier continuation 3,1.
     * It is a chain, NOT the root siblings of the retired proposal. */
    const RadBatch* batch = w.sched.step();
    REQUIRE(batch);
    REQUIRE_OK(rad_stream_sync(w.stream));
    REQUIRE_EQ(r->n_draft, 2);
    CHECK_EQ(r->draft[0], 3);
    CHECK_EQ(r->draft[1], 1);
    CHECK(!r->tree_parent && !r->tree_depth);
    CHECK_EQ(batch->positions[2], w.sched.plan().e[0].ctx_len + 2);
    CHECK_EQ(batch->spec_parent[2], 1);
    std::vector<int32_t> tok;
    const std::vector<float> lg(12, 0.25f);
    REQUIRE(w.sched.plan().e[0].req == r);
    REQUIRE_OK(ch.step({r}, lg, &tok));
    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    REQUIRE_EQ(n, (int64_t)3);
    CHECK_EQ(p[2].pos, (uint64_t)r->output.size() + 2);
    CHECK_EQ(p[2].hist_len, (int32_t)(r->prompt.size() + r->output.size()) + 2);
    const uint32_t drafts[] = {3, 1};
    CandidateTree chain;
    chain.build_linear(1u, drafts, 2);
    CHECK_EQ(tok[2], oracle(*r, meta_of(chain, 2), 2, lg.data(), 4));
}

TEST(a_pipelined_linear_publication_cannot_republish_the_previous_tree) {
    FakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    auto req = std::make_unique<Request>(make_req(5758, {3}, SamplingParams{}));
    req->max_tokens = 100;
    req->may_speculate = true;
    REQUIRE_OK(sched.add(std::move(req)));
    const int slot = sched.by_id_.at(5758);
    auto& s = *sched.reqs_[(size_t)slot];
    CandidateTree tree;
    tree.add_node(0, -1, 3u, 0);
    tree.add_node(1, 0, 1u, 1);
    tree.add_node(2, 0, 2u, 1);
    tree.add_node(3, 2, 1u, 2);
    REQUIRE_OK(sched.set_draft_tree(5758, tree, 0));
    const int32_t* parents = s.req->tree_parent;
    REQUIRE(parents);
    /* The host publication arrives for a next window already planned on the card. */
    s.entry_next = 0;
    s.draft_next = 2;
    sched.next_open_ = true;
    StepEntry e{};
    e.slot = slot; e.req = s.req.get(); e.n_spec = 2; e.n_tokens = 3;
    sched.plan_next_.e.push_back(e);
    const int32_t linear[] = {1, 2};
    CHECK_EQ(sched.set_draft(5758, linear, 1), RAD_E_STATE);
    CHECK(s.has_draft_tree && s.req->tree_parent == parents);
    REQUIRE_OK(sched.set_draft(5758, linear, 2));
    CHECK(!s.has_draft_tree);
    sched.promote_next();
    CHECK(!s.req->tree_parent && !s.req->tree_depth);
    REQUIRE_EQ(s.req->n_draft, 2);
    HostChain ch(4, 1, 3);
    NEED_CHAIN(ch)
    REQUIRE(sched.plan().e[0].req == s.req.get());
    REQUIRE_OK(ch.s.build_step({s.req.get()}));
    int64_t n = 0;
    const DeviceSampleParams* p = ch.s.staged_params(&n);
    REQUIRE_EQ(n, (int64_t)3);
    CHECK_EQ(p[2].pos, (uint64_t)2);
    radiance::TreeRegistry::instance().clear(5758);
}

/* THE PUBLISHED WINDOW IS WHAT THE SAMPLER STAGES: end to end, set_draft_tree and build_step on
 * the same Request. The chain draws each node's serial-prefix answer from the topology the
 * scheduler published -- not from a hand-attached one. */
TEST(the_published_window_stages_through_the_real_sampler) {
    FakeKV kv;
    Scheduler sched;
    sched.kv_ = &kv;
    HostChain ch(6, 1, 15);
    NEED_CHAIN(ch)

    auto req = std::make_unique<Request>();
    req->id = 4242; req->prompt = { 3, 4 }; req->may_speculate = true; req->max_tokens = 1000;
    SamplingParams sp;
    sp.temp = 1.0f;
    sp.top_k = 0;
    sp.seed = 123;
    sp.rep_penalty = 7.0f;
    sp.penalty_last_n = -1;
    req->sp = sp;
    req->output = { 5, 2 };
    CHECK_OK(sched.add(std::move(req)));
    const int slot = sched.by_id_.at(4242);
    auto& s = *sched.reqs_[(size_t)slot];

    uint32_t cands[15];
    for (int i = 0; i < 15; ++i) cands[i] = (uint32_t)(10 + i);
    CandidateTree tree;
    tree.build_top2(7u, cands, false, 15);
    CHECK_OK(sched.set_draft_tree(4242, tree, 4));
    REQUIRE(s.req->tree_parent != nullptr);

    NodeMeta m = meta_of(tree, 15);
    std::vector<float> lg(16 * 6);
    for (size_t i = 0; i < lg.size(); ++i)
        lg[i] = 0.3f * (float)((i * 11) % 6) - 0.9f;

    std::vector<int32_t> tok;
    CHECK_OK(ch.step({ s.req.get() }, lg, &tok));
    for (int j = 0; j < 16; ++j)
        CHECK_EQ(tok[(size_t)j], oracle(*s.req, m, j, lg.data(), 6));
}

TEST(dloop_verifies_the_full_single_stream_tree_without_changing_legacy_caps) {
    struct Case { bool tree_state; int sequences, configured, declared, expected; };
    for (const Case c : {Case{true, 1, 15, 15, 15}, Case{true, 2, 15, 15, 4},
                         Case{false, 1, 15, 15, 8}, Case{false, 2, 15, 15, 4},
                         Case{true, 1, 6, 15, 6}, Case{true, 1, 15, 7, 7}}) {
        ScheduledWindow w;
        w.prog.drafter.kind = RAD_DRAFT_BLOCK;
        w.prog.drafter.depth = c.declared;
        REQUIRE_OK(w.init(c.configured, 256, c.tree_state, c.sequences));
        SamplingParams sp;
        for (int i = 0; i < c.sequences; ++i) {
            auto req = std::make_unique<Request>(make_req(5757 + i, {5}, sp));
            req->max_tokens = 100;
            req->may_speculate = true;
            REQUIRE_OK(w.sched.add(std::move(req)));
        }
        REQUIRE(w.sched.step());
        const int32_t anchors[] = {5, 5};
        Scheduler::StepResult pref{};
        pref.n_seq = c.sequences;
        pref.token = anchors;
        REQUIRE_OK(w.sched.commit(pref));

        /* The fixed 15-node DLoop quota: 2 nodes at depths 1..7, one at 8. */
        CandidateTree tree;
        REQUIRE(tree.add_node(0, -1, 5, 0));
        int parent = 0;
        for (int depth = 1, node = 1; depth <= 8; ++depth) {
            const int first = node;
            const int quota = depth == 8 ? 1 : 2;
            for (int j = 0; j < quota; ++j, ++node)
                REQUIRE(tree.add_node(node, parent, static_cast<uint32_t>(10 + node), depth));
            parent = first;
        }
        int32_t flat[15];
        for (int i = 0; i < 15; ++i) flat[i] = 11 + i;
        for (int i = 0; i < c.sequences; ++i) {
            REQUIRE_OK(w.sched.set_draft(5757 + i, flat, 15));
            CHECK_EQ(w.sched.find(5757 + i)->n_draft, c.expected);
            REQUIRE_OK(w.sched.set_draft_tree(5757 + i, tree, 1));
        }
        const RadBatch* verify = w.sched.step();
        REQUIRE(verify);
        REQUIRE_EQ(w.sched.plan().e.size(), static_cast<size_t>(c.sequences));
        CHECK_EQ(verify->n_tok, static_cast<int64_t>((c.expected + 1) * c.sequences));
        for (const auto& e : w.sched.plan().e) {
            CHECK_EQ(e.n_spec, c.expected);
            CHECK_EQ(e.n_tokens, c.expected + 1);
        }
        const int32_t accepted[] = {0, 0};
        Scheduler::StepResult committed{};
        committed.n_seq = c.sequences;
        committed.n_accepted = accepted;
        committed.token = anchors;
        REQUIRE_OK(w.sched.commit(committed));
        Scheduler::BlockRow rows[2]{};
        REQUIRE_EQ(w.sched.block_begin(c.declared + 1, c.declared, 32, 0, rows, c.sequences),
                   c.sequences);
        const RadBatch* query = w.sched.block_query(2);
        REQUIRE(query);
        CHECK_EQ(query->n_spec, c.expected);
        w.sched.block_end();
        radiance::TreeRegistry::instance().clear(5758);
    }
}

TEST(tree_commit_pitch_follows_configured_capacity_not_accepted_length) {
    struct Case { int configured, declared, pitch; };
    for (const Case c : {Case{15,15,16}, Case{16,16,65}, Case{64,64,65}, Case{3,64,65}}) {
        ScheduledWindow w;
        w.prog.ctx.max_spec = c.declared;
        REQUIRE_OK(w.init(c.configured, 256, true));
        CHECK_EQ(w.sched.tree_path_pitch(), static_cast<int64_t>(c.pitch));
        CHECK_EQ(w.sched.builder_.dloop_span_at(), static_cast<int64_t>(2 * c.pitch + 5));
    }
    DraftController controller;
    REQUIRE_OK(controller.configure({64, false, true}));
    CHECK_EQ(controller.next_depth(1), 64);
    CHECK_EQ(controller.next_depth(2), 4);
}

TEST(wide_tree_stages65_rows_and_samples_the_high_node_ancestor_path) {
    HostChain ch(80, 1, 64, 256);
    NEED_CHAIN(ch)
    ScheduledWindow w;
    w.prog.ctx.max_spec = 64;
    w.prog.drafter.kind = RAD_DRAFT_SERIAL;
    w.prog.drafter.depth = 64;
    REQUIRE_OK(w.init(64, 256, true));
    SamplingParams sp; sp.temp = 0.0f; sp.rep_penalty = 10.0f; sp.penalty_last_n = -1;
    REQUIRE_OK(w.prefill(sp, 3));
    Request* r = w.sched.find(5757);
    REQUIRE(r);
    CandidateTree tree;
    REQUIRE(tree.add_node(0, -1, 3, 0));
    for (int i = 1; i < 65; ++i) {
        const int p = i == 31 ? 15 : i == 63 ? 31 : i == 64 ? 63 : 0;
        REQUIRE(tree.add_node(i, p, static_cast<uint32_t>(10 + i), tree[p].depth + 1));
    }
    REQUIRE_OK(w.sched.set_draft_tree(r->id, tree, static_cast<uint32_t>(r->n_computed)));
    const RadBatch* batch = w.sched.step();
    REQUIRE(batch);
    REQUIRE_OK(rad_stream_sync(w.stream));
    REQUIRE_EQ(batch->n_tok, int64_t{65});
    REQUIRE_EQ(batch->n_out, int64_t{65});
    REQUIRE(batch->spec_parent);
    CHECK(batch->tree_ancestors == nullptr); // no truncated uint16 masks masquerading as valid ancestry
    CHECK_EQ(w.sched.tree_path_pitch(), int64_t{65});
    for (int i = 0; i < 65; ++i) {
        CHECK_EQ(batch->spec_parent[i], tree[i].parent_id);
        CHECK_EQ(batch->positions[i], r->n_computed + tree[i].depth);
        CHECK_EQ(batch->token_ids[i], static_cast<int32_t>(tree[i].token_id));
    }
    const int32_t path[] = {0, 15, 31, 63, 64};
    std::vector<float> logits(65 * 80, -1.0f);
    for (int node = 0; node < 65; ++node) logits[node * 80 + 77] = 1.0f;
    for (int j = 0; j < 4; ++j)
        logits[path[j] * 80 + tree[path[j + 1]].token_id] = 10.0f;
    std::vector<int32_t> sampled;
    REQUIRE_OK(ch.step({r}, logits, &sampled));
    REQUIRE_EQ(sampled.size(), size_t{65});
    const NodeMeta m = meta_of(tree, 64);
    for (int i = 0; i < 65; ++i)
        CHECK_EQ(sampled[i], oracle(*r, m, i, logits.data(), 80));
    int32_t out[65]{}, nodes[65]{}, accepted = -1, bonus = -1;
    REQUIRE_OK(accept_tree_greedy(tree, sampled.data(), 65, out, &accepted, &bonus, nodes));
    REQUIRE_EQ(accepted, 4);
    CHECK_EQ(bonus, 77);
    for (int i = 0; i < 5; ++i) CHECK_EQ(nodes[i], path[i]);
    // The engine publishes this same path beside its token emission before commit.
    const StepEntry& e = w.sched.plan().e[0];
    std::copy_n(out, accepted, const_cast<int32_t*>(e.draft));
    w.sched.accepted_leaf(e.seq, nodes[accepted], nodes, accepted + 1);
    Scheduler::StepResult result{}; result.n_seq = 1;
    result.n_accepted = &accepted; result.token = &bonus;
    REQUIRE_OK(w.sched.commit(result));
    CHECK_EQ(r->output.size(), size_t{6}); // prefill's token, four drafts, bonus
    for (int i = 0; i < 4; ++i) CHECK_EQ(r->output[i + 1], out[i]);
    CHECK_EQ(r->output.back(), bonus);
}

RAD_TEST_MAIN()
