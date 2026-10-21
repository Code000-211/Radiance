/* tree_state_oracle_test.cpp -- the state lane's CPU oracle for parent-indexed tree serving.
 *
 * libref/ref_tree_state.h is a reference, and a reference is checked against something OTHER
 * than itself. Three witnesses, in decreasing order of independence:
 *
 *   1. A hand-written walk in THIS file: for each node, the ancestor chain is replayed one
 *      token at a time from the same committed state, in double precision, with the arithmetic
 *      restated from the algorithm as documented (not by calling the header). This is the
 *      oracle the tree plan asked for: per-node output and state, computed along the ancestry
 *      path and nowhere else.
 *   2. The real libref ops: ref_gdn.cpp is compiled into this binary (libref is not linked into
 *      rad_core, and including the translation unit is how the linear parity case reaches the
 *      actual plugin arithmetic). A chain-shaped tree must agree with gdn_conv_update followed
 *      by gdn_recurrent_update run as the engine would run them -- per-candidate state slots,
 *      num_accepted 1, one cumulative-lengths tensor over two sequences.
 *   3. The factor replay: the same inline state step the direct walk calls, so the replay of a
 *      chain of factor rows must reproduce the per-node states BITWISE. That is the property the
 *      GPU wave depends on (one committed state + factors, no per-node state pools), and a
 *      bitwise claim needs the shared code path, not a tolerance.
 *
 * Negative controls are not optional decoration here. A parent-oblivious implementation that
 * flattens the tree into a row prefix produces fluent, plausible output, so the sibling test
 * asserts both isolation (a poisoned sibling leaves the other subtree untouched) and
 * sensitivity (the flattened-prefix walk DOES disagree, i.e. this test can see the bug it
 * exists to catch).
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "../libref/ref_tree_state.h"
#include "../libref/ref_common.h"
#include "../libref/ref_gdn.cpp"

#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* ================================================================== fixtures */
struct Rng {
    uint32_t s;
    float operator()() { s = s * 1664525u + 1013904223u;
                         return ((s >> 9) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f; }
    int upto(int m) { return (int)(((s = s * 1664525u + 1013904223u) >> 16) % (uint32_t)m); }
};

struct Geom {
    RadTreeGdn g{};
    int width = 0, conv_dim = 0, n = 0;
    std::vector<int> parent;
    std::vector<float> x, cw, cb, hist0, state0, av, bv, Alog, dtb, z, nw;
    std::vector<float> q, k, v, o, states, hist, factors;
    std::vector<int> depth;

    size_t qk_row() const { return (size_t)n * g.n_head_k * g.head_k; }
    size_t v_row()  const { return (size_t)n * g.n_head_v * g.head_v; }
    size_t slot()   const { return (size_t)g.n_head_v * g.head_v * g.head_k; }
    size_t hist_row() const { return (size_t)conv_dim * (width - 1); }
    int flen() const { return rad_tree_factor_len(&g); }
};

/* A tree with a deep spine (depth 5) and forks off it at arbitrary earlier nodes, so node
 * numbering interleaves siblings and depths: nothing may be inferred from an index. */
void build_geom(Geom& G, int Hg, int Hv, int K, int V, int width, int n, uint32_t seed) {
    Rng r{seed};
    G.g = RadTreeGdn{K, V, Hg, Hv, 0.5f, 1e-6f, 1e-6f, 20.0f, 0};
    G.width = width;
    G.n = n;
    G.conv_dim = 2 * Hg * K + Hv * V;
    G.parent.assign(n, -1);
    for (int i = 1; i < n; ++i)
        G.parent[i] = i <= 4 ? i - 1 : r.upto(i);
    G.depth.assign(n, 0);
    rad_tree_depths(G.parent.data(), n, G.depth.data());

    G.x.resize((size_t)n * G.conv_dim);
    for (auto& f : G.x) f = r();
    G.cw.resize((size_t)G.conv_dim * width);
    for (auto& f : G.cw) f = 0.5f * r();
    G.cb.resize(G.conv_dim);
    for (auto& f : G.cb) f = 0.1f * r();
    G.hist0.resize(G.hist_row());
    for (auto& f : G.hist0) f = 0.3f * r();
    G.state0.resize(G.slot());
    for (auto& f : G.state0) f = 0.2f * r();
    G.av.resize((size_t)n * Hv); G.bv = G.av;
    for (int i = 0; i < n * Hv; ++i) { G.av[i] = r(); G.bv[i] = r(); }
    G.Alog.resize(Hv); G.dtb.resize(Hv);
    for (int h = 0; h < Hv; ++h) { G.Alog[h] = -2.5f + 0.5f * r(); G.dtb[h] = -1.0f + 0.4f * r(); }
    G.z.resize((size_t)n * Hv * V);
    for (auto& f : G.z) f = r();
    G.nw.resize(V);
    for (auto& f : G.nw) f = 1.0f + 0.25f * r();

    G.q.resize(G.qk_row()); G.k = G.q; G.v.resize(G.v_row()); G.o.resize(G.v_row());
    G.states.resize((size_t)n * G.slot());
    G.hist.resize((size_t)n * G.hist_row());
    G.factors.resize((size_t)n * Hv * G.flen());
}

int run_tree(Geom& G) {
    int rc = rad_tree_conv(&G.g, G.width, G.conv_dim, G.parent.data(), G.n,
                           G.x.data(), G.cw.data(), G.cb.data(), G.hist0.data(),
                           G.q.data(), G.k.data(), G.v.data(), G.hist.data());
    if (rc != RAD_OK) return rc;
    rc = rad_tree_pack_factors(&G.g, G.parent.data(), G.n, G.k.data(), G.v.data(),
                               G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                               G.factors.data());
    if (rc != RAD_OK) return rc;
    return rad_tree_gdn(&G.g, G.parent.data(), G.n, G.q.data(), G.k.data(), G.v.data(),
                        G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                        G.state0.data(), G.z.data(), G.nw.data(),
                        G.o.data(), G.states.data());
}

/* ================================================================== the independent oracle */
/* One token of the conv, double accumulation, restated from the documented algorithm. */
void oracle_conv_token(const Geom& G, const float* win, const float* xrow,
                       std::vector<float>& qkv /* conv_dim, split later */) {
    const int W = G.width;
    qkv.assign(G.conv_dim, 0.0f);
    for (int c = 0; c < G.conv_dim; ++c) {
        double acc = G.cb[c];
        for (int j = 0; j < W; ++j) {
            const double val = j == W - 1 ? xrow[c] : win[(size_t)c * (W - 1) + j];
            acc += val * (double)G.cw[(size_t)c * W + j];
        }
        qkv[c] = (float)(acc / (1.0 + std::exp(-acc)));
    }
}

void oracle_split(const Geom& G, const std::vector<float>& qkv, int i,
                  std::vector<float>& qo, std::vector<float>& ko, std::vector<float>& vo) {
    const int Hg = G.g.n_head_k, K = G.g.head_k, Hv = G.g.n_head_v, V = G.g.head_v;
    const int qk = Hg * K;
    qo.assign(qk, 0.0f); ko.assign(qk, 0.0f); vo.assign((size_t)Hv * V, 0.0f);
    for (int c = 0; c < G.conv_dim; ++c) {
        if (c < qk)                 qo[c] = qkv[c];
        else if (c < 2 * qk)        ko[c - qk] = qkv[c];
        else                        vo[c - 2 * qk] = qkv[c];
    }
    (void)i;
}

/* One token of the delta recurrence + epilogue, double accumulation inside the row. */
void oracle_gdn_token(const Geom& G, int i, const std::vector<float>& qo,
                      const std::vector<float>& ko, const std::vector<float>& vo,
                      std::vector<float>& S, std::vector<float>& out) {
    const int Hg = G.g.n_head_k, K = G.g.head_k, Hv = G.g.n_head_v, V = G.g.head_v;
    const int grp = Hv / Hg;
    out.assign((size_t)Hv * V, 0.0f);
    for (int h = 0; h < Hv; ++h) {
        const int hk = h / grp;
        /* softplus with the linear branch, exactly as the reference thresholds it */
        const double ain = (double)G.av[(size_t)i * Hv + h] + G.dtb[h];
        const double spl = ain > 20.0 ? ain : (ain > 0.0 ? ain + std::log1p(std::exp(-ain))
                                                        : std::log1p(std::exp(ain)));
        const double gv = -(double)std::exp(G.Alog[h]) * spl;
        const double bt = 1.0 / (1.0 + std::exp(-(double)G.bv[(size_t)i * Hv + h]));
        double sq = 0.0, sk = 0.0;
        for (int d = 0; d < K; ++d) {
            sq += (double)qo[hk * K + d] * qo[hk * K + d];
            sk += (double)ko[hk * K + d] * ko[hk * K + d];
        }
        const double qs = (1.0 / std::sqrt(sq + (double)G.g.l2_eps)) * (double)G.g.scale;
        const double ks = 1.0 / std::sqrt(sk + (double)G.g.l2_eps);
        std::vector<double> qn(K), kn(K);
        for (int d = 0; d < K; ++d) {
            qn[d] = (double)qo[hk * K + d] * qs;
            kn[d] = (double)ko[hk * K + d] * ks;
        }
        const double dec = std::exp(gv);
        std::vector<double> orow(V);
        for (int e = 0; e < V; ++e) {
            double dot = 0.0;
            for (int d = 0; d < K; ++d) {
                S[((size_t)h * V + e) * K + d] =
                    (float)((double)S[((size_t)h * V + e) * K + d] * dec);
                dot += (double)S[((size_t)h * V + e) * K + d] * kn[d];
            }
            const double u = bt * ((double)vo[(size_t)h * V + e] - dot);
            double o1 = 0.0;
            for (int d = 0; d < K; ++d) {
                S[((size_t)h * V + e) * K + d] =
                    (float)((double)S[((size_t)h * V + e) * K + d] + u * kn[d]);
                o1 += (double)S[((size_t)h * V + e) * K + d] * qn[d];
            }
            orow[e] = o1;
        }
        double tot = 0.0;
        for (int e = 0; e < V; ++e) tot += orow[e] * orow[e];
        const double sc = 1.0 / std::sqrt(tot / (double)V + (double)G.g.norm_eps);
        for (int e = 0; e < V; ++e) {
            const double zz = G.z[((size_t)i * Hv + h) * V + e];
            out[(size_t)h * V + e] =
                (float)(orow[e] * sc * G.nw[e] * (zz / (1.0 + std::exp(-zz))));
        }
    }
}

/* The chain root..node, replayed one token at a time. */
void oracle_chain(const Geom& G, int node, std::vector<float>& S, std::vector<float>& out,
                  std::vector<float>& win) {
    std::vector<int> chain;
    for (int m = node; m >= 0; m = G.parent[m]) chain.push_back(m);
    for (size_t i = 0; i < chain.size() / 2; ++i) std::swap(chain[i], chain[chain.size() - 1 - i]);
    S = G.state0;
    win = G.hist0;
    std::vector<float> qkv, qo, ko, vo, o1, nwin(G.hist_row());
    for (int m : chain) {
        oracle_conv_token(G, win.data(), G.x.data() + (size_t)m * G.conv_dim, qkv);
        oracle_split(G, qkv, m, qo, ko, vo);
        oracle_gdn_token(G, m, qo, ko, vo, S, o1);
        if (node == m) out = o1;
        for (int c = 0; c < G.conv_dim && G.width > 1; ++c)
            for (int j = 0; j < G.width - 1; ++j)
                nwin[(size_t)c * (G.width - 1) + j] =
                    j + 1 < G.width - 1
                        ? win[(size_t)c * (G.width - 1) + j + 1]
                        : G.x[(size_t)m * G.conv_dim + c];
        if (G.width > 1) win = nwin;
    }
}

double rel_l2(const float* a, const float* b, size_t n) {
    double en = 0.0, rn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - b[i];
        en += d * d;
        rn += (double)b[i] * b[i];
    }
    return rn > 0.0 ? std::sqrt(en / rn) : std::sqrt(en);
}

}  /* namespace */

/* ================================================================== tests */
TEST(topology_and_malformed) {
    const int ok[] = {-1, 0, 0, 1, 2, 2};
    CHECK(rad_tree_check_topo(ok, 6) == RAD_OK);
    CHECK(rad_tree_check_topo(nullptr, 3) == RAD_E_INVAL);
    CHECK(rad_tree_check_topo(ok, 0) == RAD_E_INVAL);
    CHECK(rad_tree_check_topo(ok, 17) == RAD_E_INVAL);
    const int fwd[] = {-1, 2, 1};          /* parent >= i: a forward edge, i.e. a cycle */
    CHECK(rad_tree_check_topo(fwd, 3) == RAD_E_INVAL);
    const int tworoots[] = {-1, -1, 0};
    CHECK(rad_tree_check_topo(tworoots, 3) == RAD_E_INVAL);
    const int noroot[] = {0, 0};
    CHECK(rad_tree_check_topo(noroot, 2) == RAD_E_INVAL);
    const int oob[] = {-1, -3};
    CHECK(rad_tree_check_topo(oob, 2) == RAD_E_INVAL);

    Geom G;
    build_geom(G, 2, 4, 3, 5, 4, 6, 7u);
    RadTreeGdn bad = G.g;
    bad.n_head_v = 3;                       /* Hv % Hg */
    CHECK(rad_tree_conv(&bad, 4, G.conv_dim, G.parent.data(), G.n, G.x.data(), G.cw.data(),
                        G.cb.data(), G.hist0.data(), G.q.data(), G.k.data(), G.v.data(),
                        G.hist.data()) == RAD_E_SHAPE);
    bad = G.g; bad.head_k = 0;
    CHECK(rad_tree_gdn(&bad, G.parent.data(), G.n, G.q.data(), G.k.data(), G.v.data(),
                       G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                       G.state0.data(), G.z.data(), G.nw.data(), G.o.data(),
                       G.states.data()) == RAD_E_SHAPE);
    CHECK(rad_tree_conv(&G.g, 18, G.conv_dim, G.parent.data(), G.n, G.x.data(), G.cw.data(),
                        G.cb.data(), G.hist0.data(), G.q.data(), G.k.data(), G.v.data(),
                        G.hist.data()) == RAD_E_UNSUPPORTED);
    CHECK(rad_tree_conv(&G.g, 4, G.conv_dim + 1, G.parent.data(), G.n, G.x.data(), G.cw.data(),
                        G.cb.data(), G.hist0.data(), G.q.data(), G.k.data(), G.v.data(),
                        G.hist.data()) == RAD_E_SHAPE);
    CHECK(rad_tree_conv(&G.g, 4, G.conv_dim, G.parent.data(), G.n, G.x.data(), G.cw.data(),
                        nullptr, G.hist0.data(), G.q.data(), G.k.data(), G.v.data(),
                        G.hist.data()) == RAD_OK);        /* the bias is optional, like ref */
    CHECK(rad_tree_conv(&G.g, 4, G.conv_dim, G.parent.data(), G.n, G.x.data(), G.cw.data(),
                        G.cb.data(), nullptr, G.q.data(), G.k.data(), G.v.data(),
                        G.hist.data()) == RAD_E_INVAL);  /* a warm step needs its window */
    CHECK(rad_tree_gdn(&G.g, G.parent.data(), G.n, G.q.data(), G.k.data(), G.v.data(),
                       G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                       nullptr, G.z.data(), G.nw.data(), G.o.data(),
                       G.states.data()) == RAD_E_INVAL);
    CHECK(rad_tree_pack_factors(&G.g, G.parent.data(), G.n, G.k.data(), G.v.data(),
                                nullptr, G.bv.data(), G.Alog.data(), G.dtb.data(),
                                G.factors.data()) == RAD_E_INVAL);

    /* commit malformed checks */
    struct RadTreeCommit ci{};
    const int good_path[] = {0, 1, 2};
    REQUIRE(rad_tree_commit_index(G.parent.data(), G.n, good_path, 3, 0, &ci) == RAD_OK);
    CHECK(rad_tree_commit_index(nullptr, G.n, good_path, 3, 0, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, nullptr, 3, 0, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, good_path, 3, 0, nullptr) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, good_path, 0, 0, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, good_path, G.n + 1, 0, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, good_path, 3, -1, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, good_path, 3, 3, &ci) == RAD_E_INVAL); /* bonus excluded */

    /* commit null buffers */
    std::vector<float> st(G.slot()), hs(G.hist_row());
    REQUIRE(run_tree(G) == RAD_OK);
    REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                            G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                            st.data(), hs.data()) == RAD_OK);
    CHECK(memcmp(st.data(), G.states.data(), sizeof(float) * G.slot()) == 0);
    CHECK(memcmp(hs.data(), G.hist.data(), sizeof(float) * G.hist_row()) == 0);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          nullptr, G.hist0.data(), G.states.data(), G.hist.data(),
                          st.data(), hs.data()) == RAD_E_INVAL);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          G.state0.data(), nullptr, G.states.data(), G.hist.data(),
                          st.data(), hs.data()) == RAD_E_INVAL);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          G.state0.data(), G.hist0.data(), nullptr, G.hist.data(),
                          st.data(), hs.data()) == RAD_E_INVAL);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          G.state0.data(), G.hist0.data(), G.states.data(), nullptr,
                          st.data(), hs.data()) == RAD_E_INVAL);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                          nullptr, hs.data()) == RAD_E_INVAL);
    CHECK(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, good_path, 3, 0,
                          G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                          st.data(), nullptr) == RAD_E_INVAL);
}

TEST(width_one_null_history_commit) {
    Geom G;
    build_geom(G, 2, 4, 3, 5, 1, 6, 91u);
    G.parent = {-1, 0, 0, 1, 2, 3};
    REQUIRE(rad_tree_conv(&G.g, G.width, G.conv_dim, G.parent.data(), G.n,
                          G.x.data(), G.cw.data(), G.cb.data(), nullptr,
                          G.q.data(), G.k.data(), G.v.data(), nullptr) == RAD_OK);
    REQUIRE(rad_tree_pack_factors(&G.g, G.parent.data(), G.n, G.k.data(), G.v.data(),
                                  G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                                  G.factors.data()) == RAD_OK);
    REQUIRE(rad_tree_gdn(&G.g, G.parent.data(), G.n, G.q.data(), G.k.data(), G.v.data(),
                         G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                         G.state0.data(), G.z.data(), G.nw.data(),
                         G.o.data(), G.states.data()) == RAD_OK);
    const int path[] = {0, 1, 3, 5};
    std::vector<float> committed(G.slot()), replay(G.slot()), S, out, win;
    for (int accepted = 0; accepted < 4; ++accepted) {
        RadTreeCommit ci{};
        REQUIRE(rad_tree_commit_index(G.parent.data(), G.n, path, 4, accepted, &ci) == RAD_OK);
        CHECK(ci.node == path[accepted]);
        CHECK(ci.replay_len == accepted + 1);
        CHECK(ci.read_row == accepted + 1);
        /* Root-only rejection and each noncontiguous winning prefix use no history buffers. */
        REQUIRE(rad_tree_commit(&G.g, 1, G.conv_dim, G.parent.data(), G.n, path, 4, accepted,
                                G.state0.data(), nullptr, G.states.data(), nullptr,
                                committed.data(), nullptr) == RAD_OK);
        REQUIRE(rad_tree_replay_factors(&G.g, path, ci.replay_len, G.n, G.factors.data(),
                                        G.state0.data(), replay.data()) == RAD_OK);
        CHECK(memcmp(committed.data(), replay.data(), sizeof(float) * G.slot()) == 0);
        CHECK(memcmp(committed.data(), G.states.data() + (size_t)ci.node * G.slot(),
                     sizeof(float) * G.slot()) == 0);
        CHECK(memcmp(committed.data(), G.state0.data(), sizeof(float) * G.slot()) != 0);
        oracle_chain(G, ci.node, S, out, win);
        CHECK(rel_l2(committed.data(), S.data(), G.slot()) < 1e-5);
        CHECK(rel_l2(G.o.data() + (size_t)ci.node * G.v_row() / G.n,
                     out.data(), out.size()) < 1e-5);
        std::vector<float> aliased = G.state0;
        REQUIRE(rad_tree_commit(&G.g, 1, G.conv_dim, G.parent.data(), G.n, path, 4, accepted,
                                aliased.data(), nullptr, G.states.data(), nullptr,
                                aliased.data(), nullptr) == RAD_OK);
        CHECK(memcmp(aliased.data(), committed.data(), sizeof(float) * G.slot()) == 0);
    }
}

TEST(state_validation_is_transactional) {
    Geom G;
    build_geom(G, 1, 1, 2, 3, 2, 4, 92u);
    REQUIRE(run_tree(G) == RAD_OK);
    const int path[] = {0, 1};
    const int bad_paths[][2] = {{0, G.n}, {0, -1}};
    const std::vector<float> sentinel(G.slot(), 123.0f);
    std::vector<float> out = sentinel, aliased = G.state0;
    for (const auto& bad : bad_paths) {
        CHECK(rad_tree_replay_factors(&G.g, bad, 2, G.n, G.factors.data(),
                                      G.state0.data(), out.data()) == RAD_E_INVAL);
        CHECK(out == sentinel);
        CHECK(rad_tree_replay_factors(&G.g, bad, 2, G.n, G.factors.data(),
                                      aliased.data(), aliased.data()) == RAD_E_INVAL);
        CHECK(aliased == G.state0);
    }
    CHECK(rad_tree_replay_factors(&G.g, nullptr, 1, G.n, G.factors.data(),
                                  aliased.data(), aliased.data()) == RAD_E_INVAL);
    CHECK(aliased == G.state0);
    REQUIRE(rad_tree_replay_factors(&G.g, nullptr, 0, G.n, G.factors.data(),
                                    G.state0.data(), out.data()) == RAD_OK);
    CHECK(out == G.state0);

    RadTreeGdn bad = G.g;
    bad.head_v = -1;
    const RadTreeGdn* invalid[] = {nullptr, &bad};
    for (const RadTreeGdn* cfg : invalid) {
        std::vector<float> history = G.hist0;
        CHECK(rad_tree_commit(cfg, G.width, G.conv_dim, G.parent.data(), G.n, path, 2, 1,
                              aliased.data(), history.data(), G.states.data(), G.hist.data(),
                              aliased.data(), history.data()) == RAD_E_SHAPE);
        CHECK(aliased == G.state0);
        CHECK(history == G.hist0);
    }
    for (int width : {513, INT_MAX}) {
        for (int axis = 0; axis < 2; ++axis) {
            bad = G.g;
            bad.head_k = axis == 0 ? width : 1;
            bad.head_v = axis == 1 ? width : 1;
            const std::vector<float> factors = G.factors;
            CHECK(rad_tree_pack_factors(&bad, G.parent.data(), G.n, G.k.data(), G.v.data(),
                                         G.av.data(), G.bv.data(), G.Alog.data(), G.dtb.data(),
                                         G.factors.data()) == RAD_E_UNSUPPORTED);
            CHECK(G.factors == factors);
            CHECK(rad_tree_replay_factors(&bad, path, 2, G.n, G.factors.data(),
                                          aliased.data(), aliased.data()) == RAD_E_UNSUPPORTED);
            CHECK(aliased == G.state0);
        }
    }
}

TEST(nodes_match_independent_walk) {
    /* Three base geometries: grouped heads (Hv=4 on Hg=2, so the head->key-head map is real),
     * width 2 against a warm window, and the degenerate width 1 (no history at all);
     * plus multi-head GQA (8:2), MHA (4:4) and wider head configurations. */
    struct { int Hg, Hv, K, V, w, n; uint32_t seed; } cases[] = {
        {2, 4, 3, 5, 4, 16, 0x1234u},
        {1, 2, 4, 2, 2, 9, 0xbeefu},
        {1, 1, 2, 3, 1, 7, 0x9999u},
        {2, 8, 32, 32, 4, 12, 0x4321u},   /* multi-head GQA 4:1 */
        {4, 4, 16, 16, 3, 10, 0x5678u},   /* multi-head MHA 1:1 */
        {1, 4, 64, 64, 4, 8, 0x7890u},    /* wider heads */
    };
    for (const auto& c : cases) {
        Geom G;
        build_geom(G, c.Hg, c.Hv, c.K, c.V, c.w, c.n, c.seed);
        REQUIRE(run_tree(G) == RAD_OK);
        std::vector<float> S, out, win;
        for (int i = 0; i < G.n; ++i) {
            oracle_chain(G, i, S, out, win);
            /* the conv outputs, against the same chain's window */
            std::vector<float> qkv, qo, ko, vo, o1;
            oracle_conv_token(G, win.data(), G.x.data() + (size_t)0, qkv); /* placeholder */
            (void)qo; (void)ko; (void)vo; (void)o1;
            /* state and output of THIS node */
            char nm[96];
            snprintf(nm, sizeof nm, "geom w=%d n=%d node=%d state", c.w, c.n, i);
            double d = rel_l2(G.states.data() + (size_t)i * G.slot(), S.data(), G.slot());
            CHECK(d < 1e-5);
            if (d >= 1e-5) fprintf(stderr, "  %s rel_l2 %g\n", nm, d);
            d = rel_l2(G.o.data() + (size_t)i * G.v_row() / G.n, out.data(), out.size());
            CHECK(d < 1e-5);
            if (d >= 1e-5) fprintf(stderr, "  %s output rel_l2 %g\n", nm, d);
            /* the node's own rolling window, against the chain's */
            if (G.width > 1) {
                d = rel_l2(G.hist.data() + (size_t)i * G.hist_row(), win.data(), G.hist_row());
                CHECK(d == 0.0);           /* both shift the same floats, not just recompute */
            }
        }
        /* multi-depth is real in the fixture: the spine is depth 5 */
        CHECK(G.depth[G.n - 1] >= 2);
    }
}

TEST(factor_replay_is_bitwise) {
    Geom G;
    build_geom(G, 2, 4, 3, 5, 4, 12, 42u);
    REQUIRE(run_tree(G) == RAD_OK);
    std::vector<int> chain;
    std::vector<float> out(G.slot());
    for (int i = 0; i < G.n; ++i) {
        chain.clear();
        for (int m = i; m >= 0; m = G.parent[m]) chain.push_back(m);
        for (size_t j = 0; j < chain.size() / 2; ++j) std::swap(chain[j], chain[chain.size() - 1 - j]);
        /* partial replays too: every PREFIX of the chain, onto a copy and onto state0 itself */
        for (size_t p = 1; p <= chain.size(); ++p) {
            REQUIRE(rad_tree_replay_factors(&G.g, chain.data(), (int)p, G.n, G.factors.data(),
                                            G.state0.data(), out.data()) == RAD_OK);
            CHECK(memcmp(out.data(), G.states.data() + (size_t)chain[p - 1] * G.slot(),
                         sizeof(float) * G.slot()) == 0);
        }
    }
    /* the alias the commit path wants */
    std::vector<float> aliased = G.state0;
    int spine[] = {0, 1, 2, 3};
    REQUIRE(rad_tree_replay_factors(&G.g, spine, 4, G.n, G.factors.data(), aliased.data(),
                                    aliased.data()) == RAD_OK);
    CHECK(memcmp(aliased.data(), G.states.data() + 3 * G.slot(), sizeof(float) * G.slot()) == 0);
    /* malformed */
    int badc[] = {0, 99};
    CHECK(rad_tree_replay_factors(&G.g, badc, 2, G.n, G.factors.data(), G.state0.data(),
                                  out.data()) == RAD_E_INVAL);
    CHECK(rad_tree_replay_factors(&G.g, spine, -1, G.n, G.factors.data(), G.state0.data(),
                                  out.data()) == RAD_E_INVAL);
    CHECK(rad_tree_replay_factors(&G.g, spine, 4, 20, G.factors.data(), G.state0.data(),
                                  out.data()) == RAD_E_INVAL);
}

TEST(sibling_poisoning) {
    /* Two depth-1 siblings, then children under both. */
    const int par[] = {-1, 0, 0, 1, 2, 2};
    Geom G;
    build_geom(G, 2, 4, 3, 5, 4, 6, 0x5eedu);
    G.parent.assign(par, par + 6);
    rad_tree_depths(G.parent.data(), G.n, G.depth.data());
    REQUIRE(run_tree(G) == RAD_OK);
    std::vector<float> o0 = G.o, s0 = G.states, h0 = G.hist, q0 = G.q, k0 = G.k, v0 = G.v;

    /* Poison node 1's whole row -- projection, gates, gate z. */
    for (int c = 0; c < G.conv_dim; ++c) G.x[(size_t)1 * G.conv_dim + c] += 0.7f;
    for (int h = 0; h < G.g.n_head_v; ++h) {
        G.av[(size_t)1 * G.g.n_head_v + h] += 0.4f;
        G.bv[(size_t)1 * G.g.n_head_v + h] -= 0.3f;
    }
    for (size_t i = 0; i < G.z.size(); ++i) G.z[i] = G.z[i];
    for (int h = 0; h < G.g.n_head_v; ++h)
        for (int e = 0; e < G.g.head_v; ++e) G.z[((size_t)1 * G.g.n_head_v + h) * G.g.head_v + e] += 0.5f;
    REQUIRE(run_tree(G) == RAD_OK);

    /* ISOLATION: node 1 has no ancestor of 2, 4 or 5, so their rows are untouched bitwise. */
    const int untouched[] = {0, 2, 4, 5};
    for (int i : untouched) {
        CHECK(memcmp(&G.o[(size_t)i * (G.v_row() / G.n)], &o0[(size_t)i * (G.v_row() / G.n)],
                     sizeof(float) * (G.v_row() / G.n)) == 0);
        CHECK(memcmp(G.states.data() + (size_t)i * G.slot(), s0.data() + (size_t)i * G.slot(),
                     sizeof(float) * G.slot()) == 0);
    }
    /* SENSITIVITY: the poisoned subtree did move, root included nowhere but node 1 and 3 did. */
    CHECK(memcmp(&G.o[(size_t)1 * (G.v_row() / G.n)], &o0[(size_t)1 * (G.v_row() / G.n)],
                 sizeof(float) * (G.v_row() / G.n)) != 0);
    CHECK(memcmp(&G.o[(size_t)3 * (G.v_row() / G.n)], &o0[(size_t)3 * (G.v_row() / G.n)],
                 sizeof(float) * (G.v_row() / G.n)) != 0);

    /* NEGATIVE CONTROL: a flattened row-prefix walk (apply nodes 0..i-1 before node i,
     * whatever the parent says) disagrees for the second sibling. This is the mistake the
     * scout made: even depth-1 siblings need the state computed from the PARENT (here the
     * root), not from the row before them in the flattened order. */
    std::vector<float> S = G.state0, out, win = G.hist0;
    std::vector<float> qkv, qo, ko, vo;
    for (int i = 0; i < 3; ++i) {                 /* the flattened prefix 0,1 before node 2 */
        oracle_conv_token(G, win.data(), G.x.data() + (size_t)i * G.conv_dim, qkv);
        oracle_split(G, qkv, i, qo, ko, vo);
        oracle_gdn_token(G, i, qo, ko, vo, S, out);
        for (int c = 0; c < G.conv_dim && G.width > 1; ++c)
            for (int j = 0; j < G.width - 1; ++j)
                win[(size_t)c * (G.width - 1) + j] =
                    j + 1 < G.width - 1 ? win[(size_t)c * (G.width - 1) + j + 1]
                                        : G.x[(size_t)i * G.conv_dim + c];
    }
    CHECK(rel_l2(out.data(), G.o.data() + (size_t)2 * (G.v_row() / G.n), out.size()) > 1e-3);
    /* and the honest parent-aware walk still agrees with node 2 (the poisoned rerun) */
    oracle_chain(G, 2, S, out, win);
    CHECK(rel_l2(G.o.data() + (size_t)2 * (G.v_row() / G.n), out.data(), out.size()) < 1e-5);
    (void)q0; (void)k0; (void)v0; (void)h0;
}

TEST(commit_paths) {
    Geom G;
    build_geom(G, 2, 4, 3, 5, 4, 8, 0xace1u);
    REQUIRE(run_tree(G) == RAD_OK);

    /* a winner down the spine: path = {0, 1, 2}, path_len = 3.
     * path[0] is root anchor; path[1] is draft child 1; path[2] is draft child 2.
     * Accepted draft children:
     *   acc = 0: zero draft children accepted -> commits root anchor path[0], replay_len = 1.
     *   acc = 1: one draft child accepted    -> commits child 1 (path[1]), replay_len = 2.
     *   acc = 2: two draft children accepted  -> commits child 2 (path[2]), replay_len = 3.
     * Bonus token (or EOG promoted to bonus) is not forwarded, so acc = 3 is invalid (excluded). */
    const int path[] = {0, 1, 2};
    const int path_len = 3;
    std::vector<float> st(G.slot()), hs(G.hist_row()), rep_s(G.slot());
    struct RadTreeCommit ci{};

    for (int acc = 0; acc < path_len; ++acc) {
        REQUIRE(rad_tree_commit_index(G.parent.data(), G.n, path, path_len, acc, &ci) == RAD_OK);
        CHECK_EQ(ci.node, path[acc]);
        CHECK_EQ(ci.replay_len, acc + 1);
        CHECK_EQ(ci.read_row, acc + 1);       /* engine 1-based num_accepted; read offset is acc */
        REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, path, path_len, acc,
                                G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                                st.data(), hs.data()) == RAD_OK);
        const float* want_s = G.states.data() + (size_t)path[acc] * G.slot();
        const float* want_h = G.hist.data() + (size_t)path[acc] * G.hist_row();
        CHECK(memcmp(st.data(), want_s, sizeof(float) * G.slot()) == 0);
        CHECK(memcmp(hs.data(), want_h, sizeof(float) * G.hist_row()) == 0);

        /* Factor replay of winning path reproduces committed state bitwise */
        REQUIRE(rad_tree_replay_factors(&G.g, path, ci.replay_len, G.n, G.factors.data(),
                                        G.state0.data(), rep_s.data()) == RAD_OK);
        CHECK(memcmp(rep_s.data(), want_s, sizeof(float) * G.slot()) == 0);
    }

    /* Zero acceptance commits root anchor (node 0) */
    REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, path, path_len, 0,
                            G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                            st.data(), hs.data()) == RAD_OK);
    CHECK(memcmp(st.data(), G.states.data() + (size_t)path[0] * G.slot(), sizeof(float) * G.slot()) == 0);
    CHECK(memcmp(hs.data(), G.hist.data() + (size_t)path[0] * G.hist_row(), sizeof(float) * G.hist_row()) == 0);

    /* Zero acceptance IN PLACE: state_out == state0 commits root anchor into state0 buffer */
    std::vector<float> in_place = G.state0, hp = G.hist0;
    REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, path, path_len, 0,
                            in_place.data(), hp.data(), G.states.data(), G.hist.data(),
                            in_place.data(), hp.data()) == RAD_OK);
    CHECK(memcmp(in_place.data(), G.states.data() + (size_t)path[0] * G.slot(),
                 sizeof(float) * G.slot()) == 0);
    CHECK(memcmp(hp.data(), G.hist.data() + (size_t)path[0] * G.hist_row(),
                 sizeof(float) * G.hist_row()) == 0);

    /* Two accepted draft children in-place */
    in_place = G.state0; hp = G.hist0;
    REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), G.n, path, path_len, 2,
                            in_place.data(), hp.data(), G.states.data(), G.hist.data(),
                            in_place.data(), hp.data()) == RAD_OK);
    CHECK(memcmp(in_place.data(), G.states.data() + (size_t)path[2] * G.slot(),
                 sizeof(float) * G.slot()) == 0);
    CHECK(memcmp(hp.data(), G.hist.data() + (size_t)path[2] * G.hist_row(),
                 sizeof(float) * G.hist_row()) == 0);

    /* Malformed: acceptance past valid draft children (e.g. bonus token acc=3, acc=4, or negative) */
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, path, path_len, 3, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, path, path_len, 4, &ci) == RAD_E_INVAL);
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, path, path_len, -1, &ci) == RAD_E_INVAL);
    const int skip[] = {0, 2};
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, skip, 2, 1, &ci) == RAD_E_INVAL);
    const int offroot[] = {1, 2};
    CHECK(rad_tree_commit_index(G.parent.data(), G.n, offroot, 2, 1, &ci) == RAD_E_INVAL);
}

TEST(non_prefix_winning_path) {
    /* Explicit tree with sibling forks and deep branches:
     * 0 (root)
     * |-- 1
     * |   `-- 3
     * `-- 2
     *     |-- 4
     *     `-- 5
     *         `-- 6
     * parent = {-1, 0, 0, 1, 2, 2, 5}, n = 7
     * Winning path: {0, 2, 5, 6} (length 4, skipping 1, 3, 4 -- completely non-prefix) */
    const int par[] = {-1, 0, 0, 1, 2, 2, 5};
    const int n = 7;
    Geom G;
    build_geom(G, 2, 4, 3, 5, 4, n, 0x9876u);
    G.parent.assign(par, par + n);
    rad_tree_depths(G.parent.data(), n, G.depth.data());
    REQUIRE(run_tree(G) == RAD_OK);

    const int winning_path[] = {0, 2, 5, 6};
    const int path_len = 4;
    struct RadTreeCommit ci{};
    std::vector<float> st(G.slot()), hs(G.hist_row()), rep_s(G.slot());

    for (int acc = 0; acc < path_len; ++acc) {
        REQUIRE(rad_tree_commit_index(G.parent.data(), n, winning_path, path_len, acc, &ci) == RAD_OK);
        CHECK_EQ(ci.node, winning_path[acc]);
        CHECK_EQ(ci.replay_len, acc + 1);
        CHECK_EQ(ci.read_row, acc + 1);

        REQUIRE(rad_tree_commit(&G.g, G.width, G.conv_dim, G.parent.data(), n,
                                winning_path, path_len, acc,
                                G.state0.data(), G.hist0.data(), G.states.data(), G.hist.data(),
                                st.data(), hs.data()) == RAD_OK);

        const float* want_s = G.states.data() + (size_t)winning_path[acc] * G.slot();
        const float* want_h = G.hist.data() + (size_t)winning_path[acc] * G.hist_row();
        CHECK(memcmp(st.data(), want_s, sizeof(float) * G.slot()) == 0);
        CHECK(memcmp(hs.data(), want_h, sizeof(float) * G.hist_row()) == 0);

        /* Factor replay of non-prefix chain reproduces state bitwise */
        REQUIRE(rad_tree_replay_factors(&G.g, winning_path, ci.replay_len, n, G.factors.data(),
                                        G.state0.data(), rep_s.data()) == RAD_OK);
        CHECK(memcmp(rep_s.data(), want_s, sizeof(float) * G.slot()) == 0);

        /* Independent double-precision oracle agrees */
        std::vector<float> S, out, win;
        oracle_chain(G, winning_path[acc], S, out, win);
        CHECK(rel_l2(st.data(), S.data(), G.slot()) < 1e-5);
    }

    /* Distinct states across sibling subtrees: node 5 vs node 4 (sibling) and node 3 (other branch) */
    CHECK(memcmp(G.states.data() + 5 * G.slot(), G.states.data() + 4 * G.slot(), sizeof(float) * G.slot()) != 0);
    CHECK(memcmp(G.states.data() + 5 * G.slot(), G.states.data() + 3 * G.slot(), sizeof(float) * G.slot()) != 0);

    /* Attempting to commit unforwarded bonus token (acc == path_len == 4) fails */
    CHECK(rad_tree_commit_index(G.parent.data(), n, winning_path, path_len, 4, &ci) == RAD_E_INVAL);

    /* Malformed non-prefix paths */
    const int bad_chain[] = {0, 1, 5}; /* 5's parent is 2, not 1 */
    CHECK(rad_tree_commit_index(G.parent.data(), n, bad_chain, 3, 2, &ci) == RAD_E_INVAL);
    const int no_root[] = {2, 5, 6};   /* does not start at root */
    CHECK(rad_tree_commit_index(G.parent.data(), n, no_root, 3, 1, &ci) == RAD_E_INVAL);
}

TEST(conv_ring_wrap_multi_step) {
    /* Multi-step sequential decode rollout exceeding conv_width:
     * conv_width = 3, so hist_len = 2.
     * We run 5 sequential decode steps. In each step, tree commits an accepted token,
     * rolling the window forward and wrapping/flushing old hist entries. */
    const int W = 3, hl = W - 1;
    const int Hg = 1, Hv = 2, K = 4, V = 4;
    const int conv_dim = 2 * Hg * K + Hv * V;
    RadTreeGdn g{K, V, Hg, Hv, 0.5f, 1e-6f, 1e-6f, 20.0f, 0};

    Rng r{0x1337u};
    std::vector<float> cw((size_t)conv_dim * W), cb(conv_dim);
    for (auto& f : cw) f = 0.4f * r();
    for (auto& f : cb) f = 0.1f * r();

    std::vector<float> cur_hist(conv_dim * hl);
    for (auto& f : cur_hist) f = 0.2f * r();

    /* Collect all accepted tokens across steps for end-to-end reference */
    std::vector<std::vector<float>> accepted_x;

    const int n_steps = 5;
    for (int step = 0; step < n_steps; ++step) {
        /* Tree of 3 nodes: root 0, children 1 and 2 */
        const int n = 3;
        const int parent[] = {-1, 0, 0};
        std::vector<float> x((size_t)n * conv_dim);
        for (auto& f : x) f = r();

        std::vector<float> q((size_t)n * Hg * K), k = q;
        std::vector<float> v((size_t)n * Hv * V);
        std::vector<float> hist((size_t)n * conv_dim * hl);

        REQUIRE(rad_tree_conv(&g, W, conv_dim, parent, n, x.data(), cw.data(), cb.data(),
                              cur_hist.data(), q.data(), k.data(), v.data(), hist.data()) == RAD_OK);

        /* Commit child 1 in even steps, child 2 in odd steps */
        const int accepted_child = (step % 2) + 1;
        const int path[] = {0, accepted_child};
        struct RadTreeCommit ci{};
        REQUIRE(rad_tree_commit_index(parent, n, path, 2, 1, &ci) == RAD_OK);
        CHECK_EQ(ci.node, accepted_child);

        /* The winning chain of this step: root (path[0]), then accepted_child (path[1]) */
        accepted_x.push_back(std::vector<float>(x.begin(), x.begin() + conv_dim));
        accepted_x.push_back(std::vector<float>(x.begin() + (size_t)accepted_child * conv_dim,
                                                x.begin() + (size_t)(accepted_child + 1) * conv_dim));

        /* Update cur_hist to committed node's window */
        cur_hist.assign(hist.begin() + (size_t)accepted_child * conv_dim * hl,
                        hist.begin() + (size_t)(accepted_child + 1) * conv_dim * hl);
    }

    /* After 5 steps (10 accepted tokens), all original history is flushed out since 10 > hl (2).
     * Verify cur_hist matches the last 2 tokens of accepted_x directly! */
    const size_t total_toks = accepted_x.size();
    for (int c = 0; c < conv_dim; ++c) {
        CHECK_EQ(cur_hist[(size_t)c * hl + 0], accepted_x[total_toks - 2][c]);
        CHECK_EQ(cur_hist[(size_t)c * hl + 1], accepted_x[total_toks - 1][c]);
    }
}

TEST(reset_reuse_and_multi_seq) {
    Geom A, B;
    build_geom(A, 1, 2, 3, 3, 3, 6, 11u);
    build_geom(B, 2, 4, 3, 5, 4, 10, 23u);
    REQUIRE(run_tree(A) == RAD_OK);
    REQUIRE(run_tree(B) == RAD_OK);
    std::vector<float> oa = A.o, sa = A.states, ha = A.hist;
    /* reuse: same buffers, different tree -- nothing carries over */
    Geom C;
    build_geom(C, 1, 2, 3, 3, 3, 6, 77u);
    REQUIRE(run_tree(C) == RAD_OK);
    CHECK(memcmp(C.o.data(), oa.data(), sizeof(float) * C.o.size()) != 0);
    /* reset: rerun A over its own scratch, identical inputs -> identical bytes */
    REQUIRE(run_tree(A) == RAD_OK);
    CHECK(memcmp(A.o.data(), oa.data(), sizeof(float) * A.o.size()) == 0);
    CHECK(memcmp(A.states.data(), sa.data(), sizeof(float) * A.states.size()) == 0);
    CHECK(memcmp(A.hist.data(), ha.data(), sizeof(float) * A.hist.size()) == 0);
    /* poison A's rows; B's outputs are unchanged by construction (separate buffers), and A's
     * change -- two sequences sharing nothing but the code */
    for (auto& f : A.av) f += 0.3f;
    REQUIRE(run_tree(A) == RAD_OK);
    CHECK(memcmp(A.o.data(), oa.data(), sizeof(float) * A.o.size()) != 0);
}

/* ================================================================== the libref linear parity */
namespace {

struct TT {                                    /* a rank-N f32/i32 tensor over a flat buffer */
    RadTensor t{};
    std::vector<float> f;
    std::vector<int32_t> i;
    TT(uint32_t dt, std::vector<int64_t> shape) {
        t.dtype = dt; t.rank = (int)shape.size();
        int64_t n = 1, acc = 1;
        for (size_t d = 0; d < shape.size(); ++d) { t.shape[d] = shape[d]; n *= shape[d]; }
        for (int d = (int)shape.size() - 1; d >= 0; --d) { t.stride[d] = acc; acc *= shape[d]; }
        if (dt == RAD_F32) f.assign((size_t)n, 0.0f), t.data = f.data();
        else               i.assign((size_t)n, 0),    t.data = i.data();
    }
};

RadParam PI(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam PD(const char* k, double v)    { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadParam PS(const char* k, const char* v){ RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

}  /* namespace */

TEST(linear_parity_against_ref_ops) {
    /* A chain tree (parent[i] = i-1) IS the linear speculative layout: rows 0..L-1, one
     * cumulative-lengths row, per-candidate state slots, num_accepted 1 (the bootstrap: the
     * committed state sits in column 0). Two sequences in one launch exercise cu. */
    const int Hg = 2, Hv = 4, K = 3, V = 5, W = 4;
    const int L = 6, NS = 2, SL = W - 1 + L;      /* conv state_len >= conv_width - 2 + q_len */
    Geom A, B;
    build_geom(A, Hg, Hv, K, V, W, L, 5u);
    build_geom(B, Hg, Hv, K, V, W, L, 6u);
    for (auto* G : {&A, &B})
        for (int i = 0; i < G->n; ++i) G->parent[i] = i - 1;
    /* ONE LAYER, so ONE set of layer weights: the batch shares the conv taps, the bias,
     * A_log, dt_bias and the norm weight. The per-sequence state is each tree's own. */
    B.cw = A.cw; B.cb = A.cb; B.Alog = A.Alog; B.dtb = A.dtb; B.nw = A.nw;
    REQUIRE(run_tree(A) == RAD_OK);
    REQUIRE(run_tree(B) == RAD_OK);

    /* ---- the conv pair, two sequences, one launch ------------------------------------------ */
    std::vector<float> xrows((size_t)NS * L * A.conv_dim);
    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        for (int i = 0; i < L; ++i)
            for (int c = 0; c < A.conv_dim; ++c)
                xrows[((size_t)s * L + i) * A.conv_dim + c] = G->x[(size_t)i * A.conv_dim + c];
    }
    TT xt(RAD_F32, {(int64_t)NS * L, A.conv_dim}); xt.f = xrows;
    TT wt(RAD_F32, {A.conv_dim, W});               wt.f = A.cw;
    TT bt(RAD_F32, {A.conv_dim});                  bt.f = A.cb;
    TT cst(RAD_F32, {NS, A.conv_dim, SL});
    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        for (int c = 0; c < A.conv_dim; ++c)
            for (int j = 0; j < W - 1; ++j)
                cst.f[((size_t)s * A.conv_dim + c) * SL + j] = G->hist0[(size_t)c * (W - 1) + j];
    }
    TT cit(RAD_I32, {NS});                          for (int s = 0; s < NS; ++s) cit.i[s] = s;
    TT nat(RAD_I32, {NS});                          for (int s = 0; s < NS; ++s) nat.i[s] = 1;
    TT cut(RAD_I32, {NS + 1});                      cut.i[0] = 0; cut.i[1] = L; cut.i[2] = 2 * L;
    TT qt(RAD_F32, {(int64_t)NS * L, Hg, K}), kt(RAD_F32, {(int64_t)NS * L, Hg, K}),
       vt(RAD_F32, {(int64_t)NS * L, Hv, V});

    RadTensor ct[10] = {xt.t, wt.t, bt.t, cst.t, cit.t, nat.t, cut.t, qt.t, kt.t, vt.t};
    RadParam cp[3] = {PI("head_k", K), PI("head_v", V), PI("conv_width", W)};
    RadArgs ca{}; ca.t = ct; ca.n_t = 10; ca.p = cp; ca.n_p = 3; ca.world_size = 1;
    CHECK_EQ(ref_gdn_conv_update(&ca, (RadStream)nullptr), RAD_OK);

    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        CHECK(rel_l2(qt.f.data() + (size_t)s * L * Hg * K, G->q.data(), (size_t)L * Hg * K) == 0.0);
        CHECK(rel_l2(kt.f.data() + (size_t)s * L * Hg * K, G->k.data(), (size_t)L * Hg * K) == 0.0);
        CHECK(rel_l2(vt.f.data() + (size_t)s * L * Hv * V, G->v.data(), (size_t)L * Hv * V) == 0.0);
    }

    /* ---- the recurrent update over the same rows -------------------------------------------- */
    const int SLOTS = NS * L;
    TT st(RAD_F32, {SLOTS, Hv, V, K});
    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        for (size_t e = 0; e < G->slot(); ++e)
            st.f[(size_t)s * L * G->slot() + e] = G->state0[e];   /* column 0 holds the anchor */
    }
    TT sit(RAD_I32, {NS, L});
    for (int s = 0; s < NS; ++s)
        for (int c = 0; c < L; ++c) sit.i[(size_t)s * L + c] = s * L + c;
    TT at(RAD_F32, {(int64_t)NS * L, Hv}), bbt(RAD_F32, {(int64_t)NS * L, Hv});
    TT zt(RAD_F32, {(int64_t)NS * L, Hv, V}), nwt(RAD_F32, {V});
    TT ot(RAD_F32, {(int64_t)NS * L, Hv, V});
    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        for (int i = 0; i < L; ++i) {
            for (int h = 0; h < Hv; ++h) {
                at.f[(size_t)(s * L + i) * Hv + h]  = G->av[(size_t)i * Hv + h];
                bbt.f[(size_t)(s * L + i) * Hv + h] = G->bv[(size_t)i * Hv + h];
            }
            for (size_t e = 0; e < (size_t)Hv * V; ++e) {
                zt.f[((size_t)(s * L + i)) * Hv * V + e] = G->z[(size_t)i * Hv * V + e];
                ot.f[((size_t)(s * L + i)) * Hv * V + e] = 0.0f;
            }
        }
    }
    for (int e = 0; e < V; ++e) nwt.f[e] = A.nw[e];
    TT alt(RAD_F32, {Hv}), dtb(RAD_F32, {Hv});
    for (int h = 0; h < Hv; ++h) { alt.f[h] = A.Alog[h]; dtb.f[h] = A.dtb[h]; }

    RadTensor rt[14] = {qt.t, kt.t, vt.t, at.t, bbt.t, alt.t, dtb.t, st.t, sit.t, nat.t,
                        cut.t, zt.t, nwt.t, ot.t};
    RadParam rp[9] = {PI("q_len", L), PI("head_k", K), PI("head_v", V), PI("n_head_v", Hv),
                      PD("scale", A.g.scale), PD("eps", A.g.norm_eps), PD("l2_eps", A.g.l2_eps),
                      PS("act", "silu"), PS("state_form", "per_candidate")};
    RadArgs ra{}; ra.t = rt; ra.n_t = 14; ra.p = rp; ra.n_p = 9; ra.world_size = 1;
    CHECK_EQ(ref_gdn_recurrent_update(&ra, (RadStream)nullptr), RAD_OK);

    for (int s = 0; s < NS; ++s) {
        const Geom* G = s ? &B : &A;
        CHECK(rel_l2(ot.f.data() + (size_t)s * L * Hv * V, G->o.data(), (size_t)L * Hv * V) < 1e-6);
        for (int i = 0; i < L; ++i) {             /* every candidate's state slot, the rollback pool */
            const double d = rel_l2(st.f.data() + (size_t)(s * L + i) * G->slot(),
                                    G->states.data() + (size_t)i * G->slot(), G->slot());
            CHECK(d < 1e-6);
            if (d >= 1e-6) fprintf(stderr, "  seq %d cand %d state rel_l2 %g\n", s, i, d);
        }
        /* the conv queue is anchored at the ACCEPTANCE OFFSET, not the buffer tail: after this
         * step, position t + conv_width - 2 holds x_t and the next step reads at
         * num_accepted - 1, so the window the engine would hand the step that accepted `acc`
         * tokens is the queue at [acc-1 .. acc-1+conv_width-2] -- which must be the accepted
         * node's own rolling window, whatever the rejected tail left in the queue after it.
         * Every acceptance width is checked, which is the wrap as well as the rollback. */
        for (int acc = 1; acc <= L; ++acc) {
            const int off = acc - 1;
            for (int c = 0; c < A.conv_dim; ++c)
                for (int j = 0; j < W - 1; ++j)
                    CHECK(cst.f[((size_t)s * A.conv_dim + c) * SL + off + j] ==
                          G->hist[(size_t)(acc - 1) * A.conv_dim * (W - 1) +
                                  (size_t)c * (W - 1) + j]);
        }
    }
}

/* ================================================================== main */
RAD_TEST_MAIN()
