/* kld_test.cpp -- the KL scorer (core/kld.h): its arithmetic and its reference files.
 *
 * No device and no model. A reference is recorded from synthetic logits split over two vocabulary
 * shards, the way two ranks hand them over; a candidate is scored against it; and every position's
 * KL, argmax agreement and NLL is held to a direct computation in double from the two sets of f32
 * logits. The f16 the reference keeps is the only approximation, so the tolerance is what that
 * costs -- and a candidate identical to the reference has to score zero.
 */
#include "rad_test.h"

#include "kld.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t V = 1000;          /* two shards of 500 */
constexpr int64_t W = V / 2;

struct TmpDir {
    std::string path;
    TmpDir() {
        char t[] = "/tmp/kld_test.XXXXXX";
        path = mkdtemp(t) ? t : "";
    }
    ~TmpDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

/* Logits with a confident head and a long tail: one token a few nats clear of a normal spread,
 * which is the shape where an f16 head would cost the most and the exact one has to cover it. */
std::vector<float> make_logits(int64_t rows, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.0f, 2.5f);
    std::uniform_int_distribution<int> pick(0, (int)V - 1);
    std::vector<float> l((size_t)(rows * V));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < V; ++i) l[(size_t)(r * V + i)] = n(g);
        l[(size_t)(r * V + pick(g))] += (r % 3 == 0) ? 2.0f : 9.0f;
    }
    return l;
}

/* The two shards a step hands over, from full rows. */
struct Shards {
    std::vector<float> a, b;
    std::vector<KldShard> s;
    Shards(const std::vector<float>& full, int64_t rows) : a((size_t)(rows * W)), b((size_t)(rows * W)) {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 0; i < W; ++i) {
                a[(size_t)(r * W + i)] = full[(size_t)(r * V + i)];
                b[(size_t)(r * W + i)] = full[(size_t)(r * V + W + i)];
            }
        s = { { a.data(), 0, W }, { b.data(), W, W } };
    }
};

double lse(const float* l) {
    double m = l[0];
    for (int64_t i = 1; i < V; ++i) m = std::max(m, (double)l[i]);
    double s = 0;
    for (int64_t i = 0; i < V; ++i) s += std::exp((double)l[i] - m);
    return m + std::log(s);
}

double kl_direct(const float* p, const float* q) {
    const double zp = lse(p), zq = lse(q);
    double kl = 0;
    for (int64_t i = 0; i < V; ++i) {
        const double lp = (double)p[i] - zp;
        kl += std::exp(lp) * (lp - ((double)q[i] - zq));
    }
    return kl;
}

int64_t argmax(const float* l) {
    int64_t a = 0;
    for (int64_t i = 1; i < V; ++i) if (l[i] > l[a]) a = i;
    return a;
}

/* Two documents: 40 tokens scored from 8, and 25 scored from 0 -- so 31 + 24 rows. */
const std::vector<int32_t> kDocA = [] { std::vector<int32_t> t(40); for (int i = 0; i < 40; ++i) t[i] = (i * 37) % V; return t; }();
const std::vector<int32_t> kDocB = [] { std::vector<int32_t> t(25); for (int i = 0; i < 25; ++i) t[i] = (i * 91 + 5) % V; return t; }();

/* Positions [0, n) of a document, one logits row each, handed over in two steps the way a
 * chunked prefill would: the first chunk ends mid-document. */
void feed(Kld& k, uint64_t id, int64_t n, const std::vector<float>& full) {
    const int64_t cut = n / 2;
    {
        std::vector<float> part(full.begin(), full.begin() + cut * V);
        Shards sh(part, cut);
        CHECK_OK(k.score({ { id, 0, cut, 0 } }, sh.s));
    }
    {
        std::vector<float> part(full.begin() + cut * V, full.begin() + n * V);
        Shards sh(part, n - cut);
        CHECK_OK(k.score({ { id, cut, n - cut, 0 } }, sh.s));
    }
}

std::vector<float> read_rows(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::vector<float> v((size_t)in.tellg() / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
    return v;
}

void record(const std::string& dir, const std::vector<float>& la, const std::vector<float>& lb) {
    Kld rec;
    CHECK_OK(rec.add_doc("a", kDocA, 8));
    CHECK_OK(rec.add_doc("b", kDocB, 0));
    CHECK_OK(rec.record_into(dir, V, "reference"));
    CHECK_EQ(rec.n_rows(), (int64_t)(31 + 24));
    feed(rec, 1, 40, la);
    feed(rec, 2, 25, lb);
    CHECK_OK(rec.finish());
}

}  /* namespace */

/* THE NUMBERS THEMSELVES. Every scored position of a perturbed candidate against a direct double
 * computation from both sets of logits, and the rows outside each document's range left out. */
TEST(a_candidate_scores_what_a_direct_computation_gives) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 1), rb = make_logits(25, 2);
    record(t.path + "/ref", ra, rb);

    /* The candidate: the reference plus noise, enough to flip some argmaxes. */
    std::vector<float> ca = ra, cb = rb;
    std::mt19937 g(7);
    std::normal_distribution<float> n(0.0f, 0.6f);
    for (float& v : ca) v += n(g);
    for (float& v : cb) v += n(g);

    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, ca);
    feed(k, 2, 25, cb);
    CHECK_OK(k.finish());

    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    int64_t r = 0, flips = 0;
    auto check_doc = [&](const std::vector<float>& ref, const std::vector<float>& cand,
                         const std::vector<int32_t>& tok, int64_t from) {
        for (int64_t p = from; p + 1 < (int64_t)tok.size(); ++p, ++r) {
            const float* lp = ref.data() + p * V;
            const float* lq = cand.data() + p * V;
            const double want = kl_direct(lp, lq);
            CHECK_NEAR(rows[(size_t)(r * 4 + 0)], want, 2e-5 + 1e-3 * want);
            CHECK_NEAR(rows[(size_t)(r * 4 + 1)], lse(lq) - lq[tok[(size_t)p + 1]], 1e-4);
            CHECK_NEAR(rows[(size_t)(r * 4 + 2)], lse(lp) - lp[tok[(size_t)p + 1]], 1e-4);
            const bool same = argmax(lp) == argmax(lq);
            CHECK_EQ(rows[(size_t)(r * 4 + 3)], same ? 1.0f : 0.0f);
            flips += !same;
        }
    };
    check_doc(ra, ca, kDocA, 8);
    check_doc(rb, cb, kDocB, 0);
    CHECK_EQ(r, (int64_t)55);
    CHECK(flips > 0);         /* the noise has to have moved something for the check to mean it */
}

/* THE CONTROL. A candidate identical to the reference: every KL is the f16 storage's second-order
 * residue, and every argmax agrees. */
TEST(the_reference_against_itself_scores_zero) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 3), rb = make_logits(25, 4);
    record(t.path + "/ref", ra, rb);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "same"));
    feed(k, 1, 40, ra);
    feed(k, 2, 25, rb);
    CHECK_OK(k.finish());
    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    for (size_t r = 0; r < 55; ++r) {
        CHECK(std::fabs(rows[r * 4 + 0]) < 1e-6f);
        CHECK_EQ(rows[r * 4 + 3], 1.0f);
        CHECK_NEAR(rows[r * 4 + 1], rows[r * 4 + 2], 1e-5);
    }
}

/* A REFERENCE IS NEVER WRITTEN OVER, and one whose recording did not finish is never read. */
TEST(a_reference_is_written_once_and_read_only_when_complete) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 5), rb = make_logits(25, 6);
    record(t.path + "/ref", ra, rb);
    {
        Kld again;
        CHECK_OK(again.add_doc("a", kDocA, 8));
        CHECK_EQ(again.record_into(t.path + "/ref", V, "x"), RAD_E_INVAL);
    }
    /* A recording that stops short: the manifest is never written, finish() says what is missing,
     * and the directory is not a reference. */
    {
        Kld half;
        CHECK_OK(half.add_doc("a", kDocA, 8));
        CHECK_OK(half.add_doc("b", kDocB, 0));
        CHECK_OK(half.record_into(t.path + "/half", V, "x"));
        feed(half, 1, 40, ra);
        CHECK_EQ(half.finish(), RAD_E_STATE);
    }
    Kld k;
    CHECK(k.score_against(t.path + "/half", "", V, "x") < 0);
    /* And a vocabulary that is not the reference's. */
    Kld v;
    CHECK_EQ(v.score_against(t.path + "/ref", "", V + 1, "x"), RAD_E_INVAL);
}

/* A DOCUMENT WITH NOTHING TO SCORE is refused when it is added, not discovered as a missing row. */
/* A TAPS RECORDING (--kld-taps): every scored row's tap parts side by side in taps.bf16 at that
 * row, no logp.f16, a manifest that says both, and a ref run that refuses the directory. Each tap
 * value encodes its document, position, part and column, so a row written to the wrong place --
 * or a part to the wrong columns -- fails by value rather than by luck. */
TEST(a_taps_recording_keeps_each_rows_taps_beside_it) {
    TmpDir t;
    const std::string dir = t.path + "/taps";
    constexpr int64_t PW = 8, P = 3;
    auto tap_value = [](uint64_t id, int64_t pos, int64_t part, int64_t col) {
        return (uint16_t)(id * 4096 + (uint64_t)(pos * 64 + part * 8 + col));
    };
    const auto la = make_logits(40, 11), lb = make_logits(25, 12);

    Kld rec;
    CHECK_OK(rec.add_doc("a", kDocA, 8));
    CHECK_OK(rec.add_doc("b", kDocB, 0));
    CHECK_OK(rec.set_taps("1,23,lm_head_input", P, PW));
    CHECK_OK(rec.record_into(dir, V, "taps"));
    CHECK_EQ(rec.set_taps("1", P, PW), RAD_E_INVAL);          /* after record_into: too late */

    auto feed_taps = [&](uint64_t id, int64_t n, const std::vector<float>& full) {
        const int64_t cut = n / 2;
        for (int half = 0; half < 2; ++half) {
            const int64_t p0 = half ? cut : 0, cnt = half ? n - cut : cut;
            std::vector<float> part(full.begin() + p0 * V, full.begin() + (p0 + cnt) * V);
            Shards sh(part, cnt);
            std::vector<std::vector<uint16_t>> parts((size_t)P, std::vector<uint16_t>((size_t)(cnt * PW)));
            for (int64_t r = 0; r < cnt; ++r)
                for (int64_t k = 0; k < P; ++k)
                    for (int64_t c = 0; c < PW; ++c)
                        parts[(size_t)k][(size_t)(r * PW + c)] = tap_value(id, p0 + r, k, c);
            KldTaps tp;
            tp.part_width = PW;
            for (auto& v : parts) tp.parts.push_back(v.data());
            CHECK_OK(rec.score({ { id, p0, cnt, 0 } }, sh.s, &tp));
        }
    };
    {
        /* a step without its taps is refused, not recorded as zeros */
        std::vector<float> part(la.begin(), la.begin() + V);
        Shards sh(part, 1);
        CHECK_EQ(rec.score({ { 1, 0, 1, 0 } }, sh.s), RAD_E_STATE);
    }
    feed_taps(1, 40, la);
    feed_taps(2, 25, lb);
    CHECK_OK(rec.finish());

    CHECK(!std::filesystem::exists(dir + "/logp.f16"));
    std::ifstream in(dir + "/taps.bf16", std::ios::binary);
    std::vector<uint16_t> taps((size_t)((31 + 24) * P * PW));
    in.read(reinterpret_cast<char*>(taps.data()), (std::streamsize)(taps.size() * 2));
    CHECK((size_t)in.gcount() == taps.size() * 2);
    int64_t wrong = 0;
    auto check_doc = [&](uint64_t id, int64_t row0, int64_t from, int64_t n) {
        for (int64_t p = from; p < n - 1; ++p)
            for (int64_t k = 0; k < P; ++k)
                for (int64_t c = 0; c < PW; ++c)
                    wrong += taps[(size_t)((row0 + p - from) * P * PW + k * PW + c)] != tap_value(id, p, k, c);
    };
    check_doc(1, 0, 8, 40);
    check_doc(2, 31, 0, 25);
    CHECK_EQ(wrong, (int64_t)0);

    std::ifstream mj(dir + "/kld.json");
    const std::string manifest((std::istreambuf_iterator<char>(mj)), std::istreambuf_iterator<char>());
    CHECK(manifest.find("\"logp\": false") != std::string::npos);
    CHECK(manifest.find("\"width\": 24") != std::string::npos);
    CHECK(manifest.find("\"top\": 32") != std::string::npos);       /* kKldTopTaps */
    CHECK_EQ((int64_t)std::filesystem::file_size(dir + "/top.bin"), (int64_t)((31 + 24) * kKldTopTaps * 8));

    Kld ref;
    CHECK(ref.score_against(dir, "", V, "x") < 0);

    /* and taps handed to an ordinary recording are refused, not dropped */
    Kld plain;
    CHECK_OK(plain.add_doc("a", kDocA, 8));
    CHECK_OK(plain.record_into(t.path + "/plain", V, "x"));
    std::vector<float> part(la.begin(), la.begin() + V);
    Shards sh(part, 1);
    std::vector<uint16_t> zeros((size_t)(P * PW));
    KldTaps tp;
    tp.part_width = PW;
    for (int64_t k = 0; k < P; ++k) tp.parts.push_back(zeros.data());
    CHECK_EQ(plain.score({ { 1, 8, 1, 0 } }, sh.s, &tp), RAD_E_STATE);
}

/* A CORPUS LINE CAN CARRY ITS TOKEN IDS, for a document whose tokens are known exactly; it needs no
 * tokenizer, and a line with both a prompt and ids -- or neither -- is refused. */
TEST(a_corpus_line_can_carry_its_token_ids) {
    TmpDir t;
    const std::string corpus = t.path + "/c.jsonl";
    { std::ofstream o(corpus); o << "{\"ids\": [5, 6, 7, 8], \"score_from\": 1, \"source\": \"x\"}\n"; }
    Config cfg;
    cfg.kld_record = t.path + "/r";
    cfg.kld_corpus = corpus;
    Kld k;
    CHECK_OK(k.open(cfg, nullptr, V, "m"));
    CHECK_EQ(k.n_docs(), (size_t)1);
    CHECK(k.tokens(0) == std::vector<int32_t>({ 5, 6, 7, 8 }));
    CHECK_EQ(k.n_rows(), (int64_t)2);

    for (const char* bad : { "{\"ids\": [1, 2], \"prompt\": \"x\"}", "{\"score_from\": 0}",
                             "{\"ids\": [1, -2]}", "{\"ids\": [1, \"2\"]}" }) {
        const std::string path = t.path + "/bad.jsonl";
        { std::ofstream o(path); o << bad << "\n"; }
        Config c2;
        c2.kld_record = t.path + "/bad";
        c2.kld_corpus = path;
        Kld k2;
        CHECK_EQ(k2.open(c2, nullptr, V, "m"), RAD_E_FORMAT);
    }
}

TEST(a_document_must_leave_a_position_to_score) {
    Kld k;
    CHECK_EQ(k.add_doc("x", { 1 }, 0), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, 2), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, -1), RAD_E_INVAL);
    CHECK_OK(k.add_doc("x", { 1, 2, 3 }, 1));
    CHECK_EQ(k.n_rows(), (int64_t)1);
}

RAD_TEST_MAIN()
