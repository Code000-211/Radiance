/* avx_check_hand.cpp -- rad-avx-check's hand-built cases: operands no description draws.
 *
 * libref's opd_shape describes each op one way: one sequence where allowed, dense operands at the
 * declared rank, plain integer operands. The engine passes more, and each hand-built case below is
 * one such shape:
 *
 *   several sequences through a path chosen for uniform queries;
 *   a rank-2 column slice of a fused projection, with the projection's width as row stride;
 *   a speculative step whose `num_accepted` reads a rolling window past offset zero;
 *   a ranged parameter at its band's upper bound over an operand with fewer rows;
 *   head widths that do not divide a kernel's channel panel;
 *   packed weight codes at odd element offsets;
 *   integer stores past the target's range, and exps of -inf and NaN;
 *   bit patterns a float draw cannot make: packed (id, logit) words, sampler rows, e8m0 exponents;
 *   weight formats and scale grids libref does not describe.
 *
 * Both plugins get byte-identical copies, and each OUT/INOUT buffer is compared whole, not just the
 * view, so writes outside the view are caught. Every buffer carries guard bytes, every input must
 * be unchanged, and the parallel rerun applies as for the table (avx_check.cpp). */
#include "avx_check.h"
#include "rad_planes.h"
#include "rad_runtime.h"   /* RAD_VGEOM_* */
#include "rad_sample.h"
#ifdef AVX_CHECK_CORE_DECODER
#include "format/encoding.h"
#endif

#include <deque>

namespace {

struct Hand {
    struct Buf {
        uint32_t dt = RAD_F32;
        int64_t elems = 0;
        std::vector<uint8_t> orig;     /* the bytes both sides start from */
        GBuf a, b;
        bool out = false;
    };
    std::string name, op;
    std::deque<Buf> bufs;
    std::vector<int> buf_of;                  /* per operand: backing buffer index, or -1 */
    std::vector<RadTensor> ta, tb;
    std::vector<RadParam> p;
    /* String parameters' values with their index in `p`. `p` points into them only at launch:
     * a Hand is copied when the case vector grows, which would leave earlier pointers dangling. */
    std::vector<std::pair<size_t, std::string>> strs;
    double tol = 1e-5;
    /* Ids both sides must produce, independent of either: (operand, values). */
    int want_opd = -1;
    std::vector<int64_t> want;
    /* The return both sides must give, RAD_OK unless the case pins a refusal; a refusal must
     * leave every buffer as it was. */
    int want_rc = RAD_OK;
    /* An answer computed by neither library for output operand `oracle_opd`, both sides held to
     * it within `oracle_tol` (rel_l2). Empty when this build cannot compute one. */
    int oracle_opd = -1;
    std::vector<float> oracle;
    double oracle_tol = 0;

    Hand(const char* n, const char* o, double t) : name(n), op(o), tol(t) {}

    /* An operand over a fresh buffer of `span` elements holding `bytes`, viewed with `shape` and
     * `stride`. */
    Hand& bytes(uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t span,
                std::vector<uint8_t> b, bool out) {
        Buf bf;
        bf.dt = dt; bf.elems = span; bf.out = out;
        bf.orig = std::move(b);
        bf.orig.resize(dt_bytes(dt, span), 0);
        bf.a.assign(bf.orig);
        bf.b.assign(bf.orig);
        bufs.push_back(std::move(bf));
        RadTensor t{};
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            t.shape[i] = shape[i];
            t.stride[i] = stride[i];
        }
        ta.push_back(t); tb.push_back(t);
        buf_of.push_back((int)bufs.size() - 1);
        return *this;
    }
    /* The whole buffer is filled, since bytes outside the view stand for other columns. */
    Hand& view(uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t span,
               const std::function<float(int64_t)>& fill, bool out = false) {
        std::vector<float> v((size_t)span);
        for (int64_t i = 0; i < span; ++i) v[(size_t)i] = fill(i);
        std::vector<uint8_t> b(dt_bytes(dt, span));
        h_narrow(v.data(), dt, span, b.data());
        return bytes(dt, std::move(shape), std::move(stride), span, std::move(b), out);
    }
    static std::vector<int64_t> dense_strides(const std::vector<int64_t>& shape, int64_t* elems) {
        std::vector<int64_t> st(shape.size());
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            st[(size_t)i] = acc;
            acc *= shape[(size_t)i];
        }
        *elems = acc;
        return st;
    }
    /* A dense operand of `shape`. */
    Hand& dense(uint32_t dt, std::vector<int64_t> shape, const std::function<float(int64_t)>& fill,
                bool out = false) {
        int64_t n = 0;
        std::vector<int64_t> st = dense_strides(shape, &n);
        return view(dt, std::move(shape), std::move(st), n, fill, out);
    }
    /* A dense operand holding `src`'s bytes verbatim. */
    Hand& raw(uint32_t dt, std::vector<int64_t> shape, const void* src, bool out = false) {
        int64_t n = 0;
        std::vector<int64_t> st = dense_strides(shape, &n);
        const uint8_t* s = (const uint8_t*)src;
        return bytes(dt, std::move(shape), std::move(st), n,
                     std::vector<uint8_t>(s, s + dt_bytes(dt, n)), out);
    }
    Hand& idx(std::vector<int64_t> v, bool out = false) {
        return dense(RAD_I32, { (int64_t)v.size() }, [v](int64_t i) { return (float)v[(size_t)i]; },
                     out);
    }
    /* Another operand over operand `opd`'s buffer and view, which the launch then writes: an
     * input that is also the output (an in-place residual). */
    Hand& alias(int opd) {
        ta.push_back(ta[(size_t)opd]);
        tb.push_back(tb[(size_t)opd]);
        buf_of.push_back(buf_of[(size_t)opd]);
        bufs[(size_t)buf_of[(size_t)opd]].out = true;
        return *this;
    }
    Hand& none() {
        ta.push_back(RadTensor{});
        tb.push_back(RadTensor{});
        buf_of.push_back(-1);
        return *this;
    }
    Hand& pi(const char* k, long long v) {
        p.push_back(RadParam{ k, RAD_P_INT, v, 0, nullptr, 0 });
        return *this;
    }
    Hand& pf(const char* k, double v) {
        p.push_back(RadParam{ k, RAD_P_F64, 0, 0, nullptr, v });
        return *this;
    }
    Hand& ps(const char* k, const char* v) {
        strs.emplace_back(p.size(), v);
        p.push_back(RadParam{ k, RAD_P_STR, 0, 0, nullptr, 0 });
        return *this;
    }
    Hand& refuses(int rc) {
        want_rc = rc;
        return *this;
    }
    /* Operand `opd` (an i32 output) must hold `v` on both sides. */
    Hand& expect(int opd, std::vector<int64_t> v) {
        want_opd = opd;
        want = std::move(v);
        return *this;
    }

    enum Result { PASS, FAIL, ABSENT };

    Result fail(const std::string& why) {
        std::printf("  FAIL %-20s %s\n       %s\n", op.c_str(), name.c_str(), why.c_str());
        return FAIL;
    }

    /* After a launch on one side: guards intact and inputs unchanged. */
    bool intact(bool side_a, const char* who) {
        bool ok = true;
        for (size_t i = 0; i < buf_of.size(); ++i) {
            if (buf_of[i] < 0) continue;
            const Buf& bf = bufs[(size_t)buf_of[i]];
            const GBuf& g = side_a ? bf.a : bf.b;
            if (const int hit = g.guard_hit()) {
                fail(std::string(who) + (hit < 0 ? " wrote before the start of operand "
                                                 : " wrote past the end of operand ") +
                     std::to_string(i));
                ok = false;
            }
            if (!bf.out && std::memcmp(g.data(), bf.orig.data(), bf.orig.size()) != 0) {
                fail(std::string(who) + " changed its input operand " + std::to_string(i));
                ok = false;
            }
        }
        return ok;
    }

    Result run(const Check& ck) {
        const Lib& ref = *ck.ref;
        const Lib& lib = *ck.lib;
        const char* who = lib.info->name;
        for (const auto& [i, v] : strs) p[i].sval = v.c_str();
        const RadKernelInfo* kr = ref.row(op.c_str());
        const RadKernelInfo* ka = lib.select(op.c_str(), p.data(), (int)p.size());
        if (!kr) return fail("no row for the op in libref");
        if (!ka) {
            if (ck.own) return fail(std::string("no row of ") + who + " admits the case");
            if (ck.verbose)
                std::printf("  %-22s %-46s not served by %s\n", op.c_str(), name.c_str(), who);
            return ABSENT;
        }
        for (size_t i = 0; i < ta.size(); ++i) {
            if (buf_of[i] < 0) continue;
            ta[i].data = bufs[(size_t)buf_of[i]].a.data();
            tb[i].data = bufs[(size_t)buf_of[i]].b.data();
        }
        RadArgs aa{}, ab{};
        aa.t = ta.data(); aa.n_t = (int)ta.size();
        aa.p = p.data();  aa.n_p = (int)p.size();
        aa.world_size = 1;
        ab = aa; ab.t = tb.data();
        const int rr = kr->launch(&aa, nullptr);
        const int ra = ka->launch(&ab, nullptr);
        if (want_rc != RAD_OK) {
            bool same = true;
            for (const Buf& bf : bufs)
                same = same && bf.a.guard_hit() == 0 && bf.b.guard_hit() == 0 &&
                       std::memcmp(bf.a.data(), bf.orig.data(), bf.orig.size()) == 0 &&
                       std::memcmp(bf.b.data(), bf.orig.data(), bf.orig.size()) == 0;
            if (rr != want_rc || ra != want_rc || !same)
                return fail("expected both to refuse with " + std::to_string(want_rc) +
                            " and write nothing: libref " + std::to_string(rr) + ", " + who +
                            " " + std::to_string(ra) + (same ? "" : ", and a buffer changed"));
            if (ck.verbose)
                std::printf("  %-22s %-46s refused, nothing written\n", op.c_str(),
                            name.c_str());
            return PASS;
        }
        if (rr < 0)
            return fail("libref refused the case (" + std::to_string(rr) + "); the case is wrong");
        if (ra < 0)
            return fail(std::string(who) + " returned " + std::to_string(ra) +
                        " where libref returned " + std::to_string(rr));
        bool ok = intact(true, "libref");
        ok = intact(false, who) && ok;
        double worst = 0;
        for (const Buf& bf : bufs) {
            if (!bf.out) continue;
            double rel = 0;
            std::string why;
            if (!same_values(bf.dt, bf.a.data(), bf.b.data(), bf.elems, tol, &rel, &why)) {
                ok = false;
                std::printf("  FAIL %-20s %s\n       %s\n", op.c_str(), name.c_str(), why.c_str());
            }
            worst = std::max(worst, rel);
        }
        /* As for the table: some output must hold a value libref wrote, or the case is vacuous. */
        bool wrote = false;
        for (const Buf& bf : bufs) {
            if (!bf.out || wrote) continue;
            std::vector<float> now((size_t)bf.elems), was((size_t)bf.elems);
            h_widen(bf.a.data(), bf.dt, bf.elems, now.data());
            h_widen(bf.orig.data(), bf.dt, bf.elems, was.data());
            for (int64_t i = 0; i < bf.elems && !wrote; ++i)
                wrote = !(now[(size_t)i] == was[(size_t)i]);
        }
        if (!wrote) {
            fail("libref's launch changed no output value, so the comparison is vacuous");
            ok = false;
        }
        if (want_opd >= 0) {
            const Buf& bf = bufs[(size_t)buf_of[(size_t)want_opd]];
            for (int side = 0; side < 2; ++side) {
                const uint8_t* d = side ? bf.b.data() : bf.a.data();
                for (size_t i = 0; i < want.size(); ++i)
                    if (avx::ld_int_dt(bf.dt, d, (int64_t)i) != want[i]) {
                        std::printf("  FAIL %-20s %s\n       %s wrote %lld at %zu, expected %lld\n",
                                    op.c_str(), name.c_str(), side ? who : "libref",
                                    (long long)avx::ld_int_dt(bf.dt, d, (int64_t)i), i,
                                    (long long)want[i]);
                        ok = false;
                        break;
                    }
            }
        }
        double core_rel = -1;
        if (oracle_opd >= 0 && !oracle.empty()) {
            const Buf& bf = bufs[(size_t)buf_of[(size_t)oracle_opd]];
            for (int side = 0; side < 2; ++side) {
                std::vector<float> got((size_t)bf.elems);
                h_widen(side ? bf.b.data() : bf.a.data(), bf.dt, bf.elems, got.data());
                const Err e = compare(got.data(), oracle.data(), bf.elems);
                core_rel = std::max(core_rel, e.rel_l2);
                if (e.nonfinite_gap || !(e.rel_l2 <= oracle_tol)) {
                    std::printf("  FAIL %-20s %s\n       %s is %.3e from the core decoder's answer "
                                "(tolerance %.1e)\n", op.c_str(), name.c_str(),
                                side ? who : "libref", e.rel_l2, oracle_tol);
                    ok = false;
                }
            }
        }
        if (ok && ck.threads > 0 && lib.par_force) {
            std::vector<uint8_t*> outs;
            std::vector<size_t> sizes;
            for (Buf& bf : bufs)
                if (bf.out) {
                    outs.push_back(bf.b.data());
                    sizes.push_back(bf.b.size());
                }
            auto reset = [&] {
                for (Buf& bf : bufs)
                    if (bf.out) std::memcpy(bf.b.data(), bf.orig.data(), bf.orig.size());
            };
            std::string why;
            if (!parallel_rerun(lib, ck.threads, outs, sizes, reset,
                                [&] { return ka->launch(&ab, nullptr); }, &why)) {
                ok = false;
                std::printf("  FAIL %-20s %s\n       %s\n", op.c_str(), name.c_str(), why.c_str());
            }
            ok = intact(false, who) && ok;
        }
        if (ck.verbose && core_rel >= 0)
            std::printf("  %-22s %-46s rel_l2=%.3e tol=%.1e, core decoder %.3e %s\n", op.c_str(),
                        name.c_str(), worst, tol, core_rel, ok ? "ok" : "FAIL");
        else if (ck.verbose)
            std::printf("  %-22s %-46s rel_l2=%.3e tol=%.1e %s\n", op.c_str(), name.c_str(), worst,
                        tol, ok ? "ok" : "FAIL");
        return ok ? PASS : FAIL;
    }
};

/* Deterministic pseudo-random fill in [-sigma, sigma]: element i under seed s. */
std::function<float(int64_t)> hfill(uint64_t s, float sigma = 1.0f) {
    return [s, sigma](int64_t i) {
        uint64_t x = s * 0x9E3779B97F4A7C15ull + (uint64_t)i;
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        return ((float)(x % 2001) / 1000.0f - 1.0f) * sigma;
    };
}

/* ================================================================== operand shapes */
void shape_cases(std::vector<Hand>& hs) {

    /* ---- attention: two sequences through the uniform long-query form, no cu_seqlens ------- */
    {
        const int64_t S = 2, QL = 32, KVH = 1, G = 2, HD = 16, BS = 16, NB = 6;
        Hand h("n_seq=2 q_len=32 no cu_seqlens", "attn_paged", 1e-4);
        h.dense(RAD_F32, { S * QL, KVH * G, HD }, hfill(1))
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(2))
         .dense(RAD_I32, { S, 3 }, [](int64_t i) { return (float)((i * 5) % 6); })
         .idx({ 40, 36 })
         .none().none().none()
         .dense(RAD_F32, { S * QL, KVH * G, HD }, hfill(3), true)
         .pi("q_len", QL).pi("head_dim", HD).pi("gqa", G).pi("block_size", BS).pi("causal", 1)
         .pi("window", 0).ps("q_dtype", "f32").ps("kv_dtype", "f32");
        hs.push_back(std::move(h));
    }
    /* ---- attention and the KV write over column slices of a fused [q|k|v] projection ------- */
    {
        const int64_t T = 5, NH = 2, KVH = 1, HD = 16, W = (NH + 2 * KVH) * HD, BS = 4, NB = 4;
        Hand h("k, v are rank-2 column slices, row stride 64", "kv_store", 1e-6);
        h.view(RAD_F32, { T, KVH * HD }, { W, 1 }, T * W, hfill(4))            /* k at column 0 */
         .view(RAD_F32, { T, KVH * HD }, { W, 1 }, T * W, hfill(5))
         .idx({ 3, 0, 9, 14, 6 })
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(6), true)
         .pi("M", T).pi("head_dim", HD).pi("n_head_kv", KVH).pi("block_size", BS)
         .ps("kv_dtype", "f32");
        hs.push_back(std::move(h));

        Hand g("q and out are rank-2 column slices", "attn_paged", 1e-4);
        g.view(RAD_F32, { T, NH * HD }, { W, 1 }, T * W, hfill(7))
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(8))
         .dense(RAD_I32, { T, 2 }, [](int64_t i) { return (float)(i % NB); })
         .idx({ 3, 5, 7, 2, 8 })
         .none().none().none()
         .view(RAD_F32, { T, NH * HD }, { W, 1 }, T * W, hfill(9), true)
         .pi("q_len", 1).pi("head_dim", HD).pi("gqa", NH / KVH).pi("block_size", BS)
         .pi("causal", 1).pi("window", 0).ps("q_dtype", "f32").ps("kv_dtype", "f32");
        hs.push_back(std::move(g));

        Hand d("q, k, v, out are rank-2 column slices", "attn_dense", 1e-4);
        d.view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(10))
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(11))
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(12))
         .idx({ 0, 2, 5 })
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(13), true)
         .pi("M", T).pi("head_dim", HD).pi("gqa", 1).pi("causal", 1).ps("dtype", "f32");
        hs.push_back(std::move(d));
    }
    /* ---- the GDN decode convolution read past offset zero ---------------------------------- */
    {
        const int64_t S = 2, QL = 3, HG = 2, H = 2, K = 16, V = 16, WID = 4;
        const int64_t CD = 2 * HG * K + H * V, SL = WID - 2 + QL;
        Hand h("num_accepted 2 and 3", "gdn_conv_update", 2e-5);
        h.dense(RAD_F32, { S * QL, CD }, hfill(14))
         .dense(RAD_F32, { CD, WID }, hfill(15, 0.5f))
         .none()
         .dense(RAD_F32, { 3, CD, SL }, hfill(16), true)
         .idx({ 2, 0 })
         .idx({ 2, 3 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(17), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(18), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(19), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(h));

        /* The same step over a channel-contiguous state, [slot][t][c] viewed as [slot][c][t],
         * which takes libavx's row-move path instead of the element loop. */
        Hand m("num_accepted 2 and 3, channel-contiguous state", "gdn_conv_update", 2e-5);
        m.dense(RAD_F32, { S * QL, CD }, hfill(14))
         .dense(RAD_F32, { CD, WID }, hfill(15, 0.5f))
         .none()
         .view(RAD_F32, { 3, CD, SL }, { SL * CD, 1, CD }, 3 * SL * CD, hfill(16), true)
         .idx({ 2, 0 })
         .idx({ 2, 3 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(17), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(18), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(19), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(m));
    }
    /* ---- head widths that do not divide the 1024-channel panel ------------------------------- */
    {
        const int64_t S = 2, QL = 2, HG = 4, H = 8, K = 96, V = 96, WID = 4;
        const int64_t CD = 2 * HG * K + H * V, SL = WID - 2 + QL;
        Hand h("head_k = head_v = 96, conv_dim 1536", "gdn_conv_update", 2e-5);
        h.dense(RAD_F32, { S * QL, CD }, hfill(20))
         .dense(RAD_F32, { CD, WID }, hfill(21, 0.5f))
         .none()
         .dense(RAD_F32, { 2, CD, SL }, hfill(22), true)
         .idx({ 1, 0 })
         .idx({ 1, 2 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(23), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(24), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(25), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(h));

        const int64_t M = 6;
        Hand p("head_k = head_v = 96, conv_dim 1536", "gdn_conv_prep", 2e-5);
        p.dense(RAD_F32, { M, CD + 2 * H }, hfill(26))
         .dense(RAD_F32, { CD, WID }, hfill(27, 0.5f))
         .none()
         .dense(RAD_F32, { H }, hfill(28, 0.5f))
         .dense(RAD_F32, { H }, hfill(29, 0.5f))
         .dense(RAD_F32, { 2, CD, WID - 1 }, hfill(30), true)
         .idx({ 0, 4, M })
         .none().none()
         .idx({ 1, 0 })
         .idx({ 1, 0 })
         .dense(RAD_F32, { M, HG, K }, hfill(31), true)
         .dense(RAD_F32, { M, HG, K }, hfill(32), true)
         .dense(RAD_F32, { M, H, V }, hfill(33), true)
         .dense(RAD_F32, { M, H }, hfill(34), true)
         .dense(RAD_F32, { M, H }, hfill(35), true)
         .pi("M", M).pi("head_k", K).pi("head_v", V).pi("chunk", 4).pi("conv_width", WID)
         .pf("l2_eps", 1e-6).pf("softplus_thr", 20.0);
        hs.push_back(std::move(p));
    }
    /* ---- the PLE ops at their band's upper bound, and a verify over two sequences ------------ */
    {
        const int64_t T = 5, N = 32, HC = 2, BAND = 64;
        Hand g("M = 64 over 5 rows", "ple_gate", 1e-5);
        g.dense(RAD_F32, { T, HC * N }, hfill(36))
         .dense(RAD_F32, { T, HC * N }, hfill(37))
         .dense(RAD_F32, { T, N }, hfill(38))
         .dense(RAD_F32, { HC * N }, hfill(39, 0.2f))
         .dense(RAD_F32, { HC * N }, hfill(40, 0.2f))
         .dense(RAD_F32, { HC * N }, hfill(41, 0.2f))
         .dense(RAD_F32, { T, HC * N }, hfill(42), true)
         .dense(RAD_F32, { T, HC * N }, hfill(43), true)
         .pi("M", BAND).pi("n", N).pi("hc", HC).pf("eps", 1e-6).ps("dtype", "f32")
         .pf("wadd", 1.0);
        hs.push_back(std::move(g));

        /* Two sequences, verifies of 3 and 2, reading at offsets 1 and 2 of a window deep enough
         * for both: hist 6 (width 4, dilation 2) plus a verify's n - 1. */
        const int64_t WIDTH = 4, DIL = 2, HIST = (WIDTH - 1) * DIL, SL = HIST + 2;
        Hand c("M = 64 over 5 rows, verifies of 3 and 2", "ple_conv", 1e-5);
        c.dense(RAD_F32, { T, N }, hfill(44))
         .dense(RAD_F32, { N, WIDTH }, hfill(45, 0.5f))
         .dense(RAD_F32, { 2, N, SL }, hfill(46), true)
         .idx({ 0, 3, 5 })
         .dense(RAD_F32, { T, N }, hfill(47))
         .idx({ 1, 0 })
         .idx({ 7, 9 })
         .idx({ 2, 3 })
         .dense(RAD_F32, { T, N }, hfill(48), true)
         .pi("M", BAND).pi("n", N).pi("width", WIDTH).pi("dilation", DIL).ps("dtype", "f32");
        hs.push_back(std::move(c));

        /* LFM2's form of the same window: width 3, undilated, no residual, no activation. */
        const int64_t CW = 3, CHIST = CW - 1, CSL = CHIST + 2;
        Hand k("M = 64 over 5 rows, verifies of 3 and 2", "causal_conv1d", 1e-5);
        k.dense(RAD_F32, { T, N }, hfill(55))
         .dense(RAD_F32, { N, CW }, hfill(56, 0.5f))
         .dense(RAD_F32, { 2, N, CSL }, hfill(57), true)
         .idx({ 0, 3, 5 })
         .idx({ 1, 0 })
         .idx({ 7, 9 })
         .idx({ 2, 3 })
         .dense(RAD_F32, { T, N }, hfill(58), true)
         .pi("M", BAND).pi("n", N).pi("width", CW).ps("dtype", "f32");
        hs.push_back(std::move(k));

        /* The antialiased resample upscaling a 4 x 4 table (drawn grids never exceed the table),
         * one row downscaling it, and a 1 x 1 grid. */
        static const int32_t kAaCrd[4][6] = { { 0, 6, 3, 1, 0, 5 }, { 0, 8, 4, 2, 0, 1 },
                                              { 7, 7, 7, 2, 1, 7 }, { 9, 9, 9, 3, 1, 9 } };
        Hand ga("a 4 x 4 table onto 7 x 9, 2 x 3 and 1 x 1 grids", "grid_embed_aa", 1e-6);
        ga.dense(RAD_F32, { 6, N }, hfill(59), true)
          .dense(RAD_F32, { 16, N }, hfill(60))
          .dense(RAD_I32, { 4, 6 }, [](int64_t i) { return (float)kAaCrd[i / 6][i % 6]; })
          .pi("M", 6).pi("n", N).pi("side", 4).ps("dtype", "f32");
        hs.push_back(std::move(ga));

        Hand n("M = 64 over 5 rows, verifies of 3 and 2", "ngram_ids", 0.0);
        n.idx({ 11, 12, 13, 21, 22 })
         .dense(RAD_I32, { 2, 2 + 2 }, [](int64_t i) { return (float)(100 + i); }, true)
         .idx({ 0, 3, 5 })
         .dense(RAD_I64, { 3 }, [](int64_t i) { return (float)(1000003 + 7919 * i); })
         .dense(RAD_I64, { 4 }, [](int64_t i) { return (float)(97 + 2 * i); })
         .dense(RAD_I64, { 4 }, [](int64_t i) { return (float)(1000 * i); })
         .idx({ 1, 0 })
         .idx({ 7, 9 })
         .idx({ 2, 3 })
         .dense(RAD_I32, { T, 4 }, [](int64_t) { return -1.0f; }, true)
         .pi("M", BAND).pi("heads", 4).pi("ngram", 3).pi("eos", 5);
        hs.push_back(std::move(n));
    }
    /* GLM's two-copy speculative state, including a nonempty previous replay window. */
    for (const char* mode : { "prefill", "verify" })
        for (int accepted : { 1, 2, 3 })
            for (int cold : { 0, 1 }) {
                constexpr int64_t D = 128;
                Hand h("KDA anchor accepted-prefix replay", "kda_update", 0.01);
                h.dense(RAD_BF16, { 2, 1, D }, hfill(201, 0.25f))
                    .dense(RAD_BF16, { 2, 1, D }, hfill(202, 0.25f))
                    .dense(RAD_BF16, { 2, 1, D }, hfill(203, 0.25f))
                    .dense(RAD_BF16, { 2, 1, D }, hfill(204, 0.25f))
                    .dense(RAD_BF16, { 2, 1 }, hfill(205, 0.25f))
                    .dense(RAD_F32, { 1 }, [](int64_t) { return 0.0f; })
                    .dense(RAD_F32, { D }, [](int64_t) { return -1.0f; })
                    .dense(
                        RAD_F32, { 2, 1, D, D },
                        [](int64_t i) {
                            if (i < D * D) return 0.125f;
                            i -= D * D;
                            if (!i) return 3.0f;
                            if (i > 3 * 385) return 0.0f;
                            const int at = (int)((i - 1) % 385);
                            return at < 128 ? 0.01f : at < 256 ? 0.125f : at < 384 ? 0.75f : 0.5f;
                        },
                        true)
                    .dense(RAD_I32, { 1, 6 }, [](int64_t i) { return i < 3 ? 0.0f : 1.0f; })
                    .idx({ 0, 2 })
                    .idx({ accepted })
                    .idx({ cold ? 0 : 1 })
                    .dense(RAD_BF16, { 2, 1, D }, hfill(206), true)
                    .pi("M", 2)
                    .pi("heads", 1)
                    .pi("head_k", D)
                    .pi("head_v", D)
                    .pf("lower_bound", -5)
                    .pf("scale", 1.0 / std::sqrt(128.0))
                    .pf("l2_eps", 1e-6)
                    .ps("mode", mode)
                    .ps("dtype", "bf16")
                    .ps("state_form", "anchor");
                hs.push_back(std::move(h));
            }
    for (const char* mode : { "absorb", "expand" }) {
        const bool absorb = !std::strcmp(mode, "absorb");
        const int64_t in = absorb ? 17 : 31, out = absorb ? 31 : 17;
        Hand h("MLA flattened linear inputs and outputs", "mla_project", 0.01);
        h.view(RAD_BF16, { 3, 2 * in }, { 2 * in + 7, 1 }, 3 * (2 * in + 7), hfill(215, 0.25f))
            .dense(RAD_BF16, { 68, 31 }, hfill(216, 0.25f))
            .view(RAD_BF16, { 3, 2 * out }, { 2 * out + 9, 1 }, 3 * (2 * out + 9), hfill(217), true)
            .pi("M", 3)
            .pi("heads", 2)
            .pi("head_dim", 17)
            .pi("latent_dim", 31)
            .ps("mode", mode)
            .ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }
    {
        Hand h("MoE preserves F32 probabilities with BF16 experts", "moe_gather", 1e-5);
        h.dense(RAD_BF16, { 2, 1 }, [](int64_t i) { return i ? -1024.0f : 1024.0f; })
            .dense(RAD_F32, { 1, 2 }, [](int64_t i) { return i ? 1.0f : 1.0f + 1.0f / 8192; })
            .idx({ 0, 1 })
            .none()
            .none()
            .dense(
                RAD_BF16, { 1, 1 }, [](int64_t) { return 0.0f; }, true)
            .pi("M", 1)
            .pi("n", 1)
            .pi("top_k", 2)
            .ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }
    /* Packed index pages share code and scale storage, including pitched cache pages. */
    {
        Hand h("DSA pool writes pitched managed pages", "dsa_pool", 0.001);
        h.dense(RAD_BF16, { 28, 256 }, hfill(211, 0.25f))
            .dense(RAD_F32, { 4, 128 }, hfill(212, 0.125f))
            .dense(
                RAD_F8E4M3, { 7, 128 }, [](int64_t) { return 0.0f; }, true)
            .dense(
                RAD_F32, { 7, 1 }, [](int64_t) { return 0.0f; }, true)
            .idx({ 4, -1, 2, 99, 1, 0, 3 })
            .view(
                RAD_U8, { 5, 1, 4, 34 }, { 160, 136, 34, 1 }, 5 * 160,
                [](int64_t) { return 165.0f; }, true)
            .pi("M", 7)
            .pi("n", 128)
            .pi("ratio", 4)
            .ps("dtype", "bf16");
        hs.push_back(std::move(h));
        Hand s("DSA scorer reads embedded scales", "dsa_score", 1e-5);
        s.dense(RAD_F8E4M3, { 64, 128 }, hfill(213, 0.25f))
            .dense(RAD_F32, { 64, 1 }, [](int64_t) { return 0.5f; })
            .dense(RAD_F32, { 2, 32 }, hfill(214))
            .view(RAD_U8, { 5, 1, 4, 34 }, { 160, 136, 34, 1 }, 5 * 160,
                  [](int64_t i) {
                      const int d = (int)(i % 160);
                      if (d < 128) return (float)(48 + i % 9);
                      // Little-endian F32 scale 0.5; all padding stays poisoned.
                      return d == 128 || d == 129 ? 0.0f
                             : d == 130           ? 0.0f
                             : d == 131           ? 63.0f
                                                  : 165.0f;
                  })
            .none()
            .dense(RAD_I32, { 1, 4 },
                   [](int64_t i) {
                       return (float)(i == 1 ? -1 : i == 3 ? 99 : i);
                   })
            .idx({ 0, 2 })
            .idx({ 4 })
            .dense(
                RAD_F32, { 2, 4 }, [](int64_t) { return 0.0f; }, true)
            .pi("M", 2)
            .pi("n", 128)
            .pi("heads", 32)
            .pi("blocks", 4)
            .pi("seqs", 1)
            .ps("dtype", "fp8_e4m3");
        hs.push_back(std::move(s));
    }
    /* ---- packed weight codes at odd element offsets ------------------------------------------ */
    {
        const int64_t M = 2, N = 4, K = 33;
        Hand h("W4 weight, K = 33: odd rows start mid-byte", "gemm_nt_q", 1e-5);
        h.dense(RAD_F32, { M, K }, hfill(49))
         .none()
         .dense(RAD_I4, { N, K }, hfill(50, 2.0f))
         .dense(RAD_F32, { N, 1 }, [](int64_t i) { return 0.5f + 0.25f * (float)i; })
         .dense(RAD_F32, { M, N }, hfill(51), true)
         .pi("M", M).pi("N", N).pi("K", K).ps("dtype", "w4a16");
        hs.push_back(std::move(h));

        Hand w("W2 weight, K = 33", "gemm_nt_q", 1e-5);
        w.dense(RAD_F32, { M, K }, hfill(52))
         .none()
         .dense(RAD_I2, { N, K }, hfill(53, 2.0f))
         .dense(RAD_F32, { N, 1 }, [](int64_t i) { return 1.0f + (float)i; })
         .dense(RAD_F32, { M, N }, hfill(54), true)
         .pi("M", M).pi("N", N).pi("K", K).ps("dtype", "w2a16");
        hs.push_back(std::move(w));
    }
    /* ---- integer stores past the range, and exponents of -inf and NaN ------------------------ */
    {
        const float big[] = { 1e10f, -1e10f, INFINITY, -INFINITY, 2.5f, -2.5f, 3.5f, 127.5f,
                              -128.5f, 300.0f, -300.0f, 0.49f };
        const int64_t n = (int64_t)(sizeof big / sizeof big[0]);
        for (const char* to : { "i8", "u8", "i16" }) {
            /* 51 elements: both the vector body and the scalar tail run at every level. */
            const int64_t len = 4 * n + 3;
            Hand h((std::string("f32 -> ") + to + " past the range").c_str(), "cast", 0.0);
            const uint32_t dt = rad_dtype_parse(to);
            h.dense(RAD_F32, { 1, len }, [&big, n](int64_t i) { return big[i % n]; })
             .dense(dt, { 1, len }, [](int64_t) { return 0.0f; }, true)
             .pi("M", 1).pi("n", len).ps("from", "f32").ps("to", to);
            hs.push_back(std::move(h));
        }
        /* 48 wide, whole vectors at every level, so the vector exp's handling of -inf is what
         * gets tested (a scalar tail's NaN would mask it). */
        Hand s("rows of -inf: one fully masked, one partly", "softmax", 1e-6);
        const int64_t w = 48;
        s.dense(RAD_F32, { 3, w }, [w](int64_t i) {
                const int64_t r = i / w, c = i % w;
                if (r == 0) return -INFINITY;
                if (r == 1 && c % 3 == 0) return -INFINITY;
                return (float)(c % 7) * 0.5f - 1.0f;
            })
         .dense(RAD_F32, { 3, w }, [](int64_t) { return 0.0f; }, true)
         .pi("M", 3).pi("n", w).ps("dtype", "f32");
        hs.push_back(std::move(s));
    }

    /* ---- the drafter's walk with the pairwise term in charge -------------------------------- *
     * Drawn as weights (sigma 0.02), `pred` and `succ` make the pairwise term too small to move a
     * pick. Here the codebooks are unit-scale and `unary` half that, so the pairwise term decides.
     * Some candidates and the second sequence's anchor are outside the vocabulary (the "contributes
     * nothing" branch). Run with and without `hp`. */
    for (int with_hp : { 1, 0 }) {
        const int64_t M = 3, L = 4, K = 8, R = 40, NV = 50;
        Hand h(with_hp ? "unit codebooks, hp present, ids outside the vocab"
                       : "unit codebooks, hp absent (a plane of ones)", "dflash_select", 0.0);
        h.dense(RAD_I32, { M * L, K }, [](int64_t i) { return (float)((i * 7 + 3) % 53 - 1); })
         .dense(RAD_F32, { M * L, K }, hfill(31, 0.5f));
        if (with_hp) h.dense(RAD_BF16, { M * L, R }, hfill(32));
        else         h.none();
        h.idx({ 5, 60, 17 })
         .dense(RAD_BF16, { NV, R }, hfill(33))
         .dense(RAD_BF16, { NV, R }, hfill(34))
         .dense(RAD_I32, { M, L }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("steps", L).pi("top_k", K).pi("rank", R).pi("n_vocab", NV)
         .pi("anchor_stride", 1).ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }

    /* ---- the fp8 fusions without their optional operands ------------------------------------ *
     * The cold descriptions always include `residual` and `out_bf16`. `mode` 1 is libr4d's
     * spelling of sigmoid. */
    {
        const int64_t M = 4, n = 200;
        Hand r("no residual, no out_bf16, ragged group", "rmsnorm_quant_fp8", 4e-3);
        r.dense(RAD_BF16, { M, n }, hfill(41))
         .none()
         .dense(RAD_BF16, { n }, hfill(42, 0.1f))
         .dense(RAD_F8E4M3, { M, n }, [](int64_t) { return 0.0f; }, true)
         .dense(RAD_F32, { M, 4 }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("n", n).pf("eps", 1e-6).pi("group", 64).ps("dtype", "bf16")
         .pf("wadd", 1.0);
        hs.push_back(std::move(r));

        Hand g("mode 1, no out_bf16", "gated_quant_fp8", 6e-3);
        g.dense(RAD_BF16, { M, 2 * n }, hfill(43))
         .dense(RAD_F8E4M3, { M, n }, [](int64_t) { return 0.0f; }, true)
         .dense(RAD_F32, { M, 4 }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("n", n).pi("group", 64).ps("dtype", "bf16").pi("mode", 1);
        hs.push_back(std::move(g));

        /* Int8 codes, as the shared expert's gate writes them for its int8 down projection
         * (quant_act_i8g's codes), with the bf16 product beside them. */
        Hand gi("silu into int8 codes, with out_bf16", "gated_quant_fp8", 1e-2);
        gi.dense(RAD_BF16, { M, 2 * 256 }, hfill(44, 3.0f))
          .dense(RAD_I8, { M, 256 }, [](int64_t) { return 0.0f; }, true)
          .dense(RAD_F32, { M, 2 }, [](int64_t) { return 0.0f; }, true)
          .dense(RAD_BF16, { M, 256 }, [](int64_t) { return 0.0f; }, true)
          .pi("M", M).pi("n", 256).pi("group", 128).ps("dtype", "bf16").ps("act", "silu");
        hs.push_back(std::move(gi));
    }

    /* ---- scale_rows without its optional residual (the table always passes `add`) ---------- */
    {
        Hand h("no add, act sigmoid", "scale_rows", 1e-6);
        h.dense(RAD_F32, { 4, 70 }, hfill(72))
         .dense(RAD_F32, { 4 }, hfill(73, 2.0f))
         .none()
         .dense(RAD_F32, { 4, 70 }, [](int64_t) { return 0.0f; }, true)
         .pi("M", 4).pi("n", 70).ps("act", "sigmoid").ps("dtype", "f32");
        hs.push_back(std::move(h));
    }

    /* ---- scatter_rows: a negative index writes nothing, and unnamed rows are kept ----------- */
    {
        const int64_t n = 70;
        Hand s("idx with -1 over a 12-row pool", "scatter_rows", 0.0);
        s.dense(RAD_F32, { 4, n }, hfill(51))
         .idx({ 5, -1, 0, 9 })
         .dense(RAD_F32, { 12, n }, hfill(52), true)
         .pi("M", 4).pi("n", n).ps("dtype", "f32");
        hs.push_back(std::move(s));

        /* W4 rows of 33 codes: odd source rows start mid-byte. The output's row pitch is even,
         * so no byte holds two rows' codes. */
        Hand g("W4 rows of 33, odd source rows start mid-byte", "gather_rows", 0.0);
        g.dense(RAD_I4, { 5, 33 }, hfill(53, 7.0f))
         .idx({ 3, -1, 0, 1 })
         .view(RAD_I4, { 4, 33 }, { 34, 1 }, 4 * 34, hfill(54, 7.0f), true)
         .pi("M", 4).pi("n", 33).ps("dtype", "i4");
        hs.push_back(std::move(g));
    }

    /* ---- ParoQuant's rotation over real pairings: each stage of each group is a permutation of
     * its 128 columns read as pairs; two parts of three stages. The dense form uses a drawn matrix
     * per (part, group). ------------------------------------------------------------------------ */
    {
        const int64_t M = 5, K = 256, S = 3, P = 2, G = 128;
        std::vector<int32_t> pr((size_t)(P * S * K));
        for (int64_t ps = 0; ps < P * S; ++ps)
            for (int64_t g = 0; g < K / G; ++g) {
                std::vector<int32_t> perm((size_t)G);
                for (int64_t i = 0; i < G; ++i) perm[(size_t)i] = (int32_t)i;
                uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(ps * 31 + g + 1);
                for (int64_t i = G - 1; i > 0; --i) {
                    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
                    std::swap(perm[(size_t)i], perm[(size_t)(x % (uint64_t)(i + 1))]);
                }
                for (int64_t i = 0; i < G; ++i)
                    pr[(size_t)(ps * K + g * G + i)] = (int32_t)(g * G + perm[(size_t)i]);
            }
        const auto h63 = hfill(63);
        const auto scl = [h63](int64_t i) { return 1.25f + 0.75f * h63(i); };   /* (0.5, 2) */
        const auto zero = [](int64_t) { return 0.0f; };
        for (uint32_t ydt : { (uint32_t)RAD_F16, (uint32_t)RAD_F32 }) {
            Hand h(ydt == RAD_F16 ? "two parts of three stages, y f16" : "two parts of three stages, y f32",
                   "givens_rotate", 1e-5);
            h.dense(RAD_BF16, { M, K }, hfill(61))
             .dense(RAD_I32, { P * S, K }, [pr](int64_t i) { return (float)pr[(size_t)i]; })
             .dense(RAD_F32, { P * S, K / 2 }, hfill(62, 3.0f))
             .dense(RAD_F32, { P, K }, scl)
             .dense(ydt, { M, P * K }, zero, true)
             .pi("M", M).pi("n", K).pi("group", G).ps("dtype", "bf16").pi("stages", S)
             .pi("parts", P);
            hs.push_back(std::move(h));
        }
        /* libref sums in f64 and libavx in f32, so a code on a rounding edge can differ by one
         * grid step; the tolerance allows a few such codes in 2560. */
        for (const char* outq : { "fp8", "i8" }) {
            const bool i8 = outq[0] == 'i';
            Hand h(i8 ? "two parts, i8" : "two parts, fp8", "givens_dense_quant", 5e-3);
            h.dense(RAD_BF16, { M, K }, hfill(64))
             .dense(RAD_F16, { P * K, G }, hfill(65, 0.1f))
             .dense(RAD_F32, { P, K }, scl)
             .dense(i8 ? RAD_I8 : RAD_F8E4M3, { M, P * K }, zero, true)
             .dense(RAD_F32, { M, P * K / G }, zero, true)
             .pi("M", M).pi("n", K).pi("group", G).ps("dtype", "bf16").pi("parts", P)
             .ps("out", outq);
            hs.push_back(std::move(h));
        }
    }
}

/* ================================================================== bit patterns and formats */
void format_cases(std::vector<Hand>& hs) {
    const auto zero = [](int64_t) { return 0.0f; };

    /* ---- logit_rerank: ties, and candidates outside this rank's rows ----------------------- *
     * Head rows 3 and 7 hold the same values, so global ids 13 and 17 score exactly the same and
     * the lower id must come first. 9 lies below the shard [10, 30), 30 past it, and -1 is a pad;
     * all three must become pads that sort last. */
    {
        const int64_t M = 3, R = 6, NV = 20, K = 40, VOFF = 10;
        static const int32_t kIds[3][6] = { { 17, 13, 9, 30, -1, 25 },
                                            { 29, 10, 13, 17, 22, 11 },
                                            { -1, -1, 30, 5, 17, 13 } };
        const auto hw = hfill(70);
        Hand h("tied ids, pads below, past and at -1", "logit_rerank", 1e-5);
        h.dense(RAD_F32, { M, K }, hfill(71))
         .dense(RAD_F32, { NV, K }, [hw, K](int64_t i) {
                const int64_t r = i / K;
                return hw((r == 7 ? 3 : r) * K + i % K);
            })
         .dense(RAD_I32, { M, R }, [](int64_t i) { return (float)kIds[i / 6][i % 6]; })
         .dense(RAD_I32, { M, R }, [](int64_t) { return 7.0f; }, true)
         .dense(RAD_F32, { M, R }, zero, true)
         .dense(RAD_F32, { M, R, 2 }, zero, true)
         .pi("M", M).pi("R", R).pi("n_vocab", NV).pi("n_embd", K).pi("vocab_off", VOFF)
         .ps("dtype", "f32");
        hs.push_back(std::move(h));
    }

    /* ---- sample_merge_topk: two ranks' runs of (u32 id, f32 logit) words -------------------- *
     * Row 0 ties across the ranks at 2.0 and 1.5 (the lower id first), carries ids past 2^24
     * (exact as u32, not as f32), and has five live candidates for six slots. Row 1 asks top_k 2
     * of eight, row 2 is greedy (the chain does not run; every slot a pad). 0xffffffff is a hole. */
    {
        const int64_t M = 3, W = 2, PER = 4, NC = 6;
        const uint32_t B = 1u << 24, H = 0xffffffffu;
        const float NI = -INFINITY;
        struct Pair { uint32_t id; float v; };
        const Pair g[3][8] = {
            { { B + 5, 2.0f }, { 7, 1.5f }, { B + 1, 1.0f }, { H, NI },
              { B + 3, 2.0f }, { B + 4, 1.5f }, { H, NI }, { H, NI } },
            { { 100, 3.0f }, { 101, 1.0f }, { 102, 0.5f }, { 103, 0.25f },
              { B + 7, 3.0f }, { 200, 2.0f }, { 201, -1.0f }, { H, NI } },
            { { 9, 4.0f }, { 8, 3.0f }, { H, NI }, { H, NI },
              { B + 9, 5.0f }, { H, NI }, { H, NI }, { H, NI } },
        };
        std::vector<uint32_t> words((size_t)(M * W * PER * 2));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t j = 0; j < W * PER; ++j) {
                words[(size_t)((m * W * PER + j) * 2)] = g[m][j].id;
                std::memcpy(&words[(size_t)((m * W * PER + j) * 2 + 1)], &g[m][j].v, 4);
            }
        RadSampleParams sp[3] = {};
        for (RadSampleParams& s : sp) s.temp = 1.0f;
        sp[1].top_k = 2;
        sp[2].flags = RAD_SP_GREEDY;
        const int64_t b = (int64_t)B;
        Hand h("ties across ranks, ids past 2^24, holes, top_k 2, greedy", "sample_merge_topk", 0.0);
        h.raw(RAD_U32, { M, W * PER * 2 }, words.data())
         .raw(RAD_F32, { M, (int64_t)(sizeof(RadSampleParams) / 4) }, sp)
         .dense(RAD_I32, { M, NC }, [](int64_t) { return 7.0f; }, true)
         .dense(RAD_F32, { M, NC }, zero, true)
         .pi("M", M).pi("n_cand", NC)
         .expect(2, { b + 3, b + 5, 7, b + 4, b + 1, -1,
                      100, b + 7, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1 });
        hs.push_back(std::move(h));
    }

    /* ---- an fp8 cache read with per-(sequence, head) descales ------------------------------- *
     * Decode and a uniform verify of 5 over two sequences, both descales present and different per
     * KV head, which libref's description always leaves absent. */
    for (int64_t ql : { 1, 5 }) {
        const int64_t S = 2, KVH = 2, G = 2, HD = 64, BS = 16, NB = 4;
        Hand h(ql == 1 ? "fp8 cache, k_descale and v_descale, decode"
                       : "fp8 cache, k_descale and v_descale, verify of 5", "attn_paged", 2e-2);
        h.dense(RAD_BF16, { S * ql, KVH * G, HD }, hfill(80))
         .dense(RAD_F8E4M3, { NB, KVH, BS, 2 * HD }, hfill(81, 2.0f))
         .dense(RAD_I32, { S, 2 }, [](int64_t i) { return (float)((i * 3 + 2) % 4); })
         .idx({ 20, 9 })
         .dense(RAD_F32, { S, KVH }, [](int64_t i) { return (float)(0.25 * (1 + (i * 5) % 7)); })
         .dense(RAD_F32, { S, KVH }, [](int64_t i) { return (float)(0.5 * (1 + (i * 3) % 5)); })
         .none()
         .dense(RAD_BF16, { S * ql, KVH * G, HD }, zero, true)
         .pi("q_len", ql).pi("head_dim", HD).pi("gqa", G).pi("block_size", BS).pi("causal", 1)
         .pi("window", 0).ps("q_dtype", "bf16").ps("kv_dtype", "fp8_e4m3").pf("scale", 0.125);
        hs.push_back(std::move(h));
    }

    /* ---- OCP-MXFP4 weights: e2m1 codes, one e8m0 exponent per 32 of K ----------------------- *
     * Exponents 122..130 (scales 2^-5..2^3) and one 0xFF, which is NaN: weight row 5's fourth
     * block, so output column 5 is NaN in every row on both sides. */
    {
        const int64_t M = 3, N = 48, K = 256;
        Hand h("mxfp4a8, e8m0 scales with one 0xFF (NaN)", "gemm_nt_q", 8e-3);
        h.dense(RAD_F8E4M3, { M, K }, hfill(90, 4.0f))
         .dense(RAD_F32, { M, K / 128 }, [](int64_t i) { return 0.01f * (float)(1 + i); })
         .dense(RAD_FP4E2M1, { N, K }, hfill(91, 2.5f))
         .dense(RAD_U8, { N, K / 32 }, [](int64_t i) {
                return i == 5 * 8 + 3 ? 255.0f : (float)(122 + i % 9);
            })
         .dense(RAD_BF16, { M, N }, zero, true)
         .none().none()
         .pi("M", M).pi("N", N).pi("K", K).pi("group", 32).ps("dtype", "mxfp4a8");
        hs.push_back(std::move(h));
    }

    /* ---- The same with exponents no fold of a scale into the codes can take: 253 and 254 in the
     * first 48 rows (a product up to 6 * 2^127 overflows where the grouped sum need not) and 1 in
     * the next 48 (half of 2^-126 is a bf16 subnormal), the last 48 rows ordinary. A small
     * activation scale keeps every output finite. -------------------------------------------- */
    {
        const int64_t M = 3, N = 144, K = 256;
        Hand h("mxfp4a8, e8m0 scales of 1, 253 and 254", "gemm_nt_q", 8e-3);
        h.dense(RAD_F8E4M3, { M, K }, hfill(96, 4.0f))
         .dense(RAD_F32, { M, K / 128 }, [](int64_t i) { return 1e-4f * (float)(1 + i % 3); })
         .dense(RAD_FP4E2M1, { N, K }, hfill(97, 2.5f))
         .dense(RAD_U8, { N, K / 32 }, [](int64_t i) {
                return i == 7 * 8 + 2 ? 253.0f : i == 30 * 8 ? 254.0f : i == 60 * 8 + 5 ? 1.0f
                                                                  : (float)(122 + i % 9);
            })
         .dense(RAD_BF16, { M, N }, zero, true)
         .none().none()
         .pi("M", M).pi("N", N).pi("K", K).pi("group", 32).ps("dtype", "mxfp4a8");
        hs.push_back(std::move(h));
    }

    /* ---- ParoQuant's multi-part GEMM: weight rows [0, pb1), [pb1, pb2), [pb2, N) each meet their
     * own copy of the activation, the copies side by side in `a` and `a_scale`. Three parts, and
     * two (no pb2). -------------------------------------------------------------------------- */
    for (int three : { 1, 0 }) {
        const int64_t M = 3, N = 96, K = 256, P = three ? 3 : 2;
        Hand h(three ? "mxfp4a8 in three parts, pb1 32, pb2 64"
                     : "mxfp4a8 in two parts, pb1 48", "gemm_nt_q", 8e-3);
        h.dense(RAD_F8E4M3, { M, P * K }, hfill(92 + (uint64_t)three, 4.0f))
         .dense(RAD_F32, { M, P * K / 128 }, [](int64_t i) { return 0.01f + 0.002f * (float)i; })
         .dense(RAD_FP4E2M1, { N, K }, hfill(94 + (uint64_t)three, 2.5f))
         .dense(RAD_U8, { N, K / 32 }, [](int64_t i) { return (float)(122 + i % 9); })
         .dense(RAD_BF16, { M, N }, zero, true)
         .none().none()
         .pi("M", M).pi("N", N).pi("K", K).pi("group", 32).ps("dtype", "mxfp4a8");
        if (three) h.pi("pb1", 32).pi("pb2", 64);
        else       h.pi("pb1", 48);
        hs.push_back(std::move(h));
    }

    /* ---- an activation grid finer than the weight's: libref applies such a scale per element.
     * fp8 against a per-channel weight scale (one group of all K) with an activation scale per 64.
     * ---------------------------------------------------------------------------------------- */
    {
        const int64_t M = 3, N = 48, K = 256;
        Hand h("activation scale per 64 against a per-channel weight", "gemm_nt_q", 8e-3);
        h.dense(RAD_F8E4M3, { M, K }, hfill(110, 4.0f))
         .dense(RAD_F32, { M, K / 64 }, [](int64_t i) { return 0.01f + 0.004f * (float)i; })
         .dense(RAD_F8E4M3, { N, K }, hfill(111, 4.0f))
         .dense(RAD_BF16, { N, 1 }, [](int64_t i) { return 0.02f + 0.001f * (float)i; })
         .dense(RAD_BF16, { M, N }, zero, true)
         .none().none()
         .pi("M", M).pi("N", N).pi("K", K).pi("group", K).ps("dtype", "fp8a8");
        hs.push_back(std::move(h));
    }

    /* ---- the grouped GEMM over 4-bit experts, and over a finer activation grid --------------- *
     * Token t's slot j goes to expert (t + 2j) % 4; rows are grouped by expert. Experts are int4
     * with an f16 scale per 128 of K, or e2m1 with an E4M3 scale per 32, against the E4M3
     * activation and its scale per 128; or fp8 per channel against an activation scale per 64. */
    {
        const int64_t M = 5, TOPK = 2, NE = 4, T = M * TOPK, N = 32, K = 256;
        std::vector<int64_t> sorted, off;
        for (int64_t e = 0; e < NE; ++e) {
            off.push_back((int64_t)sorted.size());
            for (int64_t t = 0; t < M; ++t)
                for (int64_t j = 0; j < TOPK; ++j)
                    if ((t + 2 * j) % NE == e) sorted.push_back(t * TOPK + j);
        }
        off.push_back(T);
        for (int form : { 0, 1, 2 }) {
            const bool fine = form == 2;
            const uint32_t wdt = form == 0 ? RAD_I4 : form == 1 ? RAD_FP4E2M1 : RAD_F8E4M3;
            const uint32_t sdt = form == 0 ? RAD_F16 : form == 1 ? RAD_F8E4M3 : RAD_BF16;
            const int64_t sk = form == 0 ? K / 128 : form == 1 ? K / 32 : 1;
            const int64_t ag = fine ? 64 : 128;
            Hand h(form == 0 ? "int4 experts, f16 scales per 128"
                   : form == 1 ? "e2m1 experts, E4M3 scales per 32"
                               : "activation scale per 64 against per-channel experts",
                   "moe_gemm_q", 8e-3);
            h.dense(RAD_F8E4M3, { M, K }, hfill(100 + (uint64_t)form, 4.0f))
             .dense(RAD_F32, { M, K / ag }, [](int64_t i) { return 0.01f + 0.003f * (float)i; })
             .dense(wdt, { NE / 2, N, K }, hfill(103 + (uint64_t)form, 2.5f))
             .dense(sdt, { NE / 2, N, sk }, [](int64_t i) { return 0.05f + 0.01f * (float)(i % 7); })
             .idx(sorted)
             .idx(off)
             .dense(wdt, { NE / 2, N, K }, hfill(106 + (uint64_t)form, 2.5f))
             .dense(sdt, { NE / 2, N, sk }, [](int64_t i) { return 0.04f + 0.01f * (float)(i % 5); })
             .dense(RAD_BF16, { T, N }, zero, true)
             .pi("M", M).pi("N", N).pi("K", K).pi("n_expert", NE).pi("top_k", TOPK)
             .pi("group", ag).ps("a_order", "token")
             .ps("dtype", form == 0 ? "w4a8" : "fp8a8");
            hs.push_back(std::move(h));
        }
    }
}


/* ================================================================== the anchor state form */

/* The anchor layout at a [V, K] head (docs/OPS.md `state_form`; libref/ref_gdn.cpp AnchorLayout):
 * the normalised k once per 32-row group of V, then v, then (g, beta) once per group, a token's
 * factors `fstr` floats apart, as many tokens as fit before the stamp up to libr4d's 16 (`tmax`),
 * the stamp at tmax * fstr. */
struct AnLayout {
    int64_t K, V, groups, fstr, gb, tmax, mark;
    constexpr AnLayout(int64_t k, int64_t v)
        : K(k), V(v), groups((v + 31) / 32), fstr(groups * k + v + 2 * groups),
          gb(groups * k + v), tmax(std::min<int64_t>(16, (v * k - 1) / fstr)),
          mark(tmax * fstr) {}
};

/* Four sequences over 14 state slots: a paged window (slots 0..5), a non-paged pair (6 and 7), a
 * sequence with no rows this step (it still replays and publishes), and a padded one (a negative
 * column: untouched). The factor blocks at column 4 carry stamps 2 and 1 (heads 0 and 1), 0
 * (zeroed: no shift) and 3 and 2, so the anchor is read at columns 2, 3, 2, 1 and 2; their first
 * num_accepted tokens hold factors with a decaying g and a beta in (0, 1). `ql` and `nacc` are the
 * first two sequences' rows and acceptances, at most the head's capacity. */
Hand anchor_case(const AnLayout& L, const char* name, int64_t ql0, int64_t ql1,
                 std::vector<int64_t> nacc) {
    const int64_t H = 2, N = 4, SLOTS = 14, T = ql0 + ql1 + 1;
    static const int32_t kIdx[4][6] = { { 0, 1, 2, 3, 4, 5 }, { 6, 6, 6, 7, 7, 7 },
                                        { 8, 9, 10, 11, 12, 13 }, { -1, 0, 0, 0, 0, 0 } };
    auto stamp = [](int64_t slot, int64_t h) -> float {
        if (slot == 4) return h ? 1.0f : 2.0f;
        if (slot == 7) return 0.0f;
        if (slot == 12) return h ? 2.0f : 3.0f;
        return -1.0f;
    };
    const auto rnd = hfill(120 + (uint64_t)L.K, 0.1f);
    auto state_fill = [stamp, rnd, H, L](int64_t i) {
        const int64_t blk = L.V * L.K, slot = i / (H * blk), h = (i / blk) % H, w = i % blk;
        const float st = stamp(slot, h);
        if (st >= 0.0f) {
            if (w == L.mark) return st;
            const int64_t t = w / L.fstr, r = w % L.fstr;
            if (t < L.tmax && r >= L.gb) return (r - L.gb) % 2 ? 0.3f + 0.05f * (float)t
                                                              : -0.2f - 0.03f * (float)t;
        }
        return rnd(i);
    };
    const int64_t K = L.K, V = L.V;
    Hand h(name, "gdn_recurrent_update", 2e-5);
    h.dense(RAD_F32, { T, 1, K }, hfill(121))
     .dense(RAD_F32, { T, 1, K }, hfill(122))
     .dense(RAD_F32, { T, H, V }, hfill(123))
     .dense(RAD_F32, { T, H }, hfill(124))
     .dense(RAD_F32, { T, H }, hfill(125))
     .dense(RAD_F32, { H }, [](int64_t i) { return -1.5f + 0.5f * (float)i; })
     .dense(RAD_F32, { H }, [](int64_t i) { return -0.75f + 0.25f * (float)i; })
     .dense(RAD_F32, { SLOTS, H, V, K }, state_fill, true)
     .dense(RAD_I32, { N, 6 }, [](int64_t i) { return (float)kIdx[i / 6][i % 6]; })
     .idx(std::move(nacc))
     .idx({ 0, ql0, ql0 + ql1, ql0 + ql1, T })
     .dense(RAD_F32, { T, H, V }, hfill(126))
     .dense(RAD_F32, { V }, [](int64_t i) { return 1.0f + 0.01f * (float)(i % 7); })
     .dense(RAD_F32, { T, H, V }, [](int64_t) { return 0.0f; }, true)
     .pi("q_len", std::max(ql0, ql1)).pi("head_k", K).pi("head_v", V).pi("n_head_v", H)
     .pf("scale", 0.125).pf("eps", 1e-6).pf("l2_eps", 1e-6).pf("softplus_thr", 20.0)
     .ps("act", "silu").ps("state_form", "anchor");
    return h;
}

void anchor_cases(std::vector<Hand>& hs) {
    /* The layout's capacity, by hand: 64 x 64 is libr4d's (16 tokens, stamp at 16 * 196); a
     * 16 x 16 block (256 floats, 34 a token) holds 7 tokens and the stamp at 7 * 34. */
    constexpr AnLayout big(64, 64), small(16, 16);
    static_assert(big.tmax == 16 && big.mark == 3136, "libr4d's layout at 64 x 64");
    static_assert(small.tmax == 7 && small.mark == 238, "a 16 x 16 head holds 7 tokens");
    hs.push_back(anchor_case(big, "64 x 64: paged, non-paged, no rows, padded", 2, 3,
                             { 2, 3, 2, 1 }));
    /* At the 16 x 16 head's capacity: a seven-token step after seven accepted tokens. */
    hs.push_back(anchor_case(small, "16 x 16 at its 7-token capacity", 7, 3, { 7, 3, 2, 1 }));
    /* One past it, in rows or in the accepted prefix: refused before any write, by both. */
    hs.push_back(anchor_case(small, "16 x 16, an 8-token step is refused", 8, 3, { 1, 3, 2, 1 })
                     .refuses(RAD_E_SHAPE));
    hs.push_back(anchor_case(small, "16 x 16, 8 accepted tokens are refused", 7, 3,
                             { 8, 3, 2, 1 })
                     .refuses(RAD_E_SHAPE));
}

/* The anchor form's contract, on the library alone: with an f32 state the replay repeats the
 * per-candidate arithmetic, so over steps accepting `sched` tokens of `ql`-row steps, with the
 * paged window sliding by 0, +1, 0 and -1 pages, both forms give the same `o` bit for bit; a run
 * that ignores num_accepted (always 1) gives a different one. This is what makes greedy
 * speculation emit the same text as greedy decode. Returns false having printed the failure; true
 * where the library does not serve the op. */
bool anchor_replays_per_candidate(const Check& ck, int64_t K, int64_t V, int64_t QL,
                                  const int (&nacc_sched)[4]) {
    const Lib& lib = *ck.lib;
    const int64_t H = 2, N = 2, T = N * QL, PAGES = 7;
    const int win_sched[4] = { 0, 1, 1, 0 };
    const int64_t blk = H * V * K;
    char what[64];
    std::snprintf(what, sizeof what, "anchor replay = per_candidate, %lld x %lld",
                  (long long)V, (long long)K);
    std::vector<float> st_pc((size_t)(N * QL * blk), 0.0f), st_an((size_t)(N * PAGES * blk), 0.0f);
    std::vector<float> st_one = st_an;
    const auto init = hfill(130, 0.1f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t i = 0; i < blk; ++i) {
            const float x = init(n * blk + i);
            st_pc[(size_t)(n * QL * blk + i)] = x;
            st_an[(size_t)((n * PAGES + win_sched[0] + 2) * blk + i)] = x;
            st_one[(size_t)((n * PAGES + win_sched[0] + 2) * blk + i)] = x;
        }
    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t x : shape) t.shape[i++] = x;
        rad_tensor_pack(&t);
        return t;
    };
    std::vector<float> alog = { -2.0f, -1.75f }, dtb = { -1.0f, -0.5f }, nw((size_t)V);
    for (int64_t e = 0; e < V; ++e) nw[(size_t)e] = 1.0f + 0.01f * (float)(e % 9);
    std::vector<int32_t> cu = { 0, (int32_t)QL, (int32_t)(2 * QL) }, idx_pc((size_t)(N * QL));
    for (int64_t i = 0; i < N * QL; ++i) idx_pc[(size_t)i] = (int32_t)i;

    int64_t differ_pc = 0, differ_one = 0;
    for (int step = 0; step < 4; ++step) {
        auto draw = [step](uint64_t s, int64_t n) {
            std::vector<float> v((size_t)n);
            const auto f = hfill(140 + 10 * (uint64_t)step + s);
            for (int64_t i = 0; i < n; ++i) v[(size_t)i] = f(i);
            return v;
        };
        std::vector<float> q = draw(1, T * K), k = draw(2, T * K), v = draw(3, T * H * V);
        std::vector<float> a = draw(4, T * H), b = draw(5, T * H), z = draw(6, T * H * V);
        std::vector<int32_t> nacc = { nacc_sched[step], nacc_sched[step] }, ones = { 1, 1 };
        std::vector<int32_t> idx_an((size_t)(N * 6));
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c = 0; c < 6; ++c)
                idx_an[(size_t)(n * 6 + c)] = (int32_t)(n * PAGES + win_sched[step] + c);
        auto go = [&](std::vector<float>& st, int64_t slots, std::vector<int32_t>& idx,
                      int64_t idx_w, std::vector<int32_t>& na, const char* form,
                      std::vector<float>& o) {
            RadTensor t[16] = {
                tensor(q.data(), RAD_F32, { T, 1, K }), tensor(k.data(), RAD_F32, { T, 1, K }),
                tensor(v.data(), RAD_F32, { T, H, V }), tensor(a.data(), RAD_F32, { T, H }),
                tensor(b.data(), RAD_F32, { T, H }), tensor(alog.data(), RAD_F32, { H }),
                tensor(dtb.data(), RAD_F32, { H }),
                tensor(st.data(), RAD_F32, { slots, H, V, K }),
                tensor(idx.data(), RAD_I32, { N, idx_w }), tensor(na.data(), RAD_I32, { N }),
                tensor(cu.data(), RAD_I32, { N + 1 }), tensor(z.data(), RAD_F32, { T, H, V }),
                tensor(nw.data(), RAD_F32, { V }), tensor(o.data(), RAD_F32, { T, H, V }),
                RadTensor{}, RadTensor{} };
            RadParam p[] = { { "q_len", RAD_P_INT, QL, 0, nullptr, 0 },
                             { "head_k", RAD_P_INT, K, 0, nullptr, 0 },
                             { "head_v", RAD_P_INT, V, 0, nullptr, 0 },
                             { "n_head_v", RAD_P_INT, H, 0, nullptr, 0 },
                             { "scale", RAD_P_F64, 0, 0, nullptr, 0.125 },
                             { "eps", RAD_P_F64, 0, 0, nullptr, 1e-6 },
                             { "act", RAD_P_STR, 0, 0, "silu", 0 },
                             { "state_form", RAD_P_STR, 0, 0, form, 0 } };
            const int np = (int)(sizeof p / sizeof p[0]);
            const RadKernelInfo* r = lib.select("gdn_recurrent_update", p, np);
            if (!r) return 1;
            RadArgs args{};
            args.t = t; args.n_t = 16; args.p = p; args.n_p = np; args.world_size = 1;
            return r->launch(&args, nullptr) == RAD_OK ? 0 : -1;
        };
        std::vector<float> o_pc((size_t)(T * H * V)), o_an = o_pc, o_one = o_pc;
        const int r1 = go(st_pc, N * QL, idx_pc, QL, nacc, "per_candidate", o_pc);
        const int r2 = go(st_an, N * PAGES, idx_an, 6, nacc, "anchor", o_an);
        const int r3 = go(st_one, N * PAGES, idx_an, 6, ones, "anchor", o_one);
        if (r1 > 0 || r2 > 0 || r3 > 0) {
            if (ck.own) {
                std::printf("  FAIL gdn_recurrent_update %s\n       no row of %s admits a form\n",
                            what, lib.info->name);
                return false;
            }
            return true;
        }
        if (r1 < 0 || r2 < 0 || r3 < 0) {
            std::printf("  FAIL gdn_recurrent_update %s\n       step %d refused (per_candidate "
                        "%d, anchor %d, anchor with num_accepted 1 %d)\n", what, step, r1, r2,
                        r3);
            return false;
        }
        differ_pc += std::memcmp(o_pc.data(), o_an.data(), o_pc.size() * sizeof(float)) != 0;
        differ_one += std::memcmp(o_pc.data(), o_one.data(), o_pc.size() * sizeof(float)) != 0;
    }
    if (differ_pc || !differ_one) {
        std::printf("  FAIL gdn_recurrent_update %s\n       %lld of 4 steps' `o` differ from "
                    "per_candidate; the num_accepted-1 run differs in %lld (must be some)\n",
                    what, (long long)differ_pc, (long long)differ_one);
        return false;
    }
    if (ck.verbose)
        std::printf("  %-22s %-46s bit for bit\n", "gdn_recurrent_update", what);
    return true;
}

/* ================================================================== packed stores */
void packed_cases(std::vector<Hand>& hs) {
    /* Nearest code, ties to the even code, saturation and NaN, into e2m1, int4 and int2: the
     * stores a gather or a cast of packed rows rests on. 27 values, so the last byte is half
     * used. */
    static const float kV[] = { 0.74f, 0.76f, -2.5f, 7.0f, -0.25f, 5.0f, 0.25f, 0.75f, 1.25f,
                                1.75f, 2.5f, 3.5f, -5.0f, 6.5f, NAN, -0.0f, 3.0f, -8.5f, -8.0f,
                                7.5f, -0.5f, 0.5f, 1e9f, -1e9f, 1.5f, -1.5f, 3.49f };
    const int64_t n = (int64_t)(sizeof kV / sizeof kV[0]);
    for (const char* to : { "fp4_e2m1", "i4", "i2" }) {
        const uint32_t dt = rad_dtype_parse(to);
        Hand h((std::string("f32 -> ") + to + ": ties, saturation, NaN").c_str(), "cast", 0.0);
        h.dense(RAD_F32, { 1, n }, [](int64_t i) { return kV[i]; })
         .dense(dt, { 1, n }, [](int64_t) { return 0.0f; }, true)
         .pi("M", 1).pi("n", n).ps("from", "f32").ps("to", to);
        hs.push_back(std::move(h));
    }
}

/* ================================================================== the codebook experts
 *
 * moe_gemm_q and moe_gemm_q_mod4 over Qwen3.8-Flash-Next's codebook experts (docs/MOE-W4.md),
 * whose views of several planes no description draws. Each case draws canonical planes (codes
 * over the whole code range, the shipped w4nl or w5nl table, E4M3 or bf16 scales, the fixed 2^-13
 * level), stores them through each library's own layout hooks, and runs each library on its own
 * stored bytes. Besides guard bytes, unchanged inputs and the threaded rerun, it checks:
 *
 *   both libraries store the same bytes, and each unrelayout gives the planes back exactly;
 *   libavx against libref;
 *   libref over the stored form against libref over the logical form the oracle and rad-kbench
 *   pass (table entries as E4M3, scale times level as bf16), bit for bit;
 *   both against the core's decoder (core/format/encoding.h), when the checker is built in the
 *   source tree: the decoded weight, its 128-wide Hadamard undone, against the activation rotated
 *   back, summed in double.
 *
 * And each library's layout hooks refuse the same selections with the same code. */
namespace cb {

constexpr float kW4nl[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                              -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };
constexpr float kW5nl[32] = { 0.8125f, 2.5f, 4.0f, 5.5f, 7.5f, 9.0f, 11.0f, 12.0f,
                              14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 30.0f,
                              -0.8125f, -2.5f, -4.0f, -5.5f, -7.5f, -9.0f, -11.0f, -12.0f,
                              -14.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -30.0f };

struct Form { const char* dtype; int bits; int64_t group; uint32_t sdt; bool level; };
constexpr Form kForms[] = {
    { "w4nla8h",   4, 128, RAD_BF16,   false },
    { "w4nl64a8h", 4, 64,  RAD_F8E4M3, true },
    { "w4nl32a8h", 4, 32,  RAD_F8E4M3, true },
    { "w5nl64a8h", 5, 64,  RAD_F8E4M3, true },
};

/* The encoding libquant writes for the form: codes, scale, its fixed level, then the table. */
RadEncoding encoding(const Form& f) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "fwht128");
    rad_enc_add_plane(&e, "codes", f.bits == 5 ? RAD_U5 : RAD_U4, 1, 1);
    rad_enc_add_plane(&e, "scale", f.sdt, 1, f.group);
    if (f.level) rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, (int64_t)1 << f.bits);
    return e;
}

uint64_t mix(uint64_t s, uint64_t i) {
    uint64_t x = s * 0x9E3779B97F4A7C15ull + i;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return x;
}

/* One expert's planes, in encoding order, dense in their own elements. */
struct Planes {
    std::vector<uint8_t> p[RAD_ENC_MAX_PLANES];
    int64_t rows[RAD_ENC_MAX_PLANES] = {}, cols[RAD_ENC_MAX_PLANES] = {};
    int64_t n = 0, k = 0;
};

Planes draw(const Form& f, const RadEncoding& e, int64_t n, int64_t k, uint64_t seed) {
    Planes w;
    w.n = n;
    w.k = k;
    for (int i = 0; i < e.n_planes; ++i) {
        rad_enc_plane_dims(&e.plane[i], n, k, &w.rows[i], &w.cols[i]);
        w.p[i].assign((size_t)(w.rows[i] * rad_enc_row_bytes(e.plane[i].dtype, w.cols[i])), 0);
    }
    const int ic = rad_enc_find(&e, "codes"), is = rad_enc_find(&e, "scale");
    const int il = rad_enc_find(&e, "scale.1"), it = rad_enc_find(&e, "table");
    const int64_t rbc = rad_enc_row_bytes(e.plane[ic].dtype, k);
    for (int64_t r = 0; r < n; ++r)
        for (int64_t c = 0; c < k; ++c)
            rad_store_code(w.p[ic].data() + r * rbc, e.plane[ic].dtype, c,
                           (int)(mix(seed, (uint64_t)(r * k + c)) % (1u << f.bits)));
    /* E4M3 scales over four binades, as the shipped experts' under 2^-13; bf16 at their size. */
    const int64_t sc = w.cols[is];
    for (int64_t r = 0; r < n; ++r)
        for (int64_t c = 0; c < sc; ++c) {
            const float u = (float)(mix(seed + 1, (uint64_t)(r * sc + c)) % 1000) / 1000.0f;
            const float v = f.sdt == RAD_F8E4M3 ? std::ldexp(1.0f + u, (int)(r + c) % 4 - 2)
                                                : 0.0005f + 0.002f * u;
            rad_store_f32(w.p[is].data() + r * rad_enc_row_bytes(f.sdt, sc), f.sdt, c, v);
        }
    if (il >= 0) rad_store_f32(w.p[il].data(), RAD_F32, 0, 0x1p-13f);
    std::memcpy(w.p[it].data(), f.bits == 5 ? kW5nl : kW4nl, sizeof(float) << f.bits);
    return w;
}

/* The selected planes as the loader hands them to a hook: rank 2, dense, with their bytes. */
std::vector<RadTensor> selection(const RadEncoding& e, const std::vector<int>& sel, Planes& w) {
    std::vector<RadTensor> t;
    for (int i : sel) {
        RadTensor x{};
        x.data = w.p[i].data();
        x.dtype = e.plane[i].dtype;
        x.rank = 2;
        x.shape[0] = w.rows[i];
        x.shape[1] = w.cols[i];
        x.stride[0] = w.cols[i];
        x.stride[1] = 1;
        t.push_back(x);
    }
    return t;
}

/* The bytes kernel `k` stores for one view of an expert, and their dtype: its layout's, or the
 * one selected plane when it stores that as it is. The layout must be the shared joined form, and
 * its inverse must give the planes back. */
bool store(const RadKernelInfo* k, const std::vector<RadParam>& p, int operand,
           const RadEncoding& e, const std::vector<int>& sel, Planes& w, std::vector<uint8_t>* out,
           uint32_t* dt, std::string* why) {
    std::vector<RadTensor> t = selection(e, sel, w);
    const int n = (int)sel.size();
    RadLayout L{};
    int rc = k->layout ? k->layout(p.data(), (int)p.size(), operand, &e, sel.data(), t.data(), n,
                                   &L)
                       : RAD_E_UNSUPPORTED;
    if (rc == RAD_E_UNSUPPORTED) {
        if (n != 1) {
            *why = std::string(k->name) + " stores several planes as they are";
            return false;
        }
        *out = w.p[sel[0]];
        *dt = e.plane[sel[0]].dtype;
        return true;
    }
    if (rc < 0 || !L.tag || std::strcmp(L.tag, RAD_PLANES_TAG) != 0 || !k->relayout ||
        !k->unrelayout) {
        *why = std::string(k->name) + "'s layout answered " + std::to_string(rc) + ", tag " +
               (L.tag ? L.tag : "(none)");
        return false;
    }
    out->assign((size_t)L.bytes, 0);
    rc = k->relayout(p.data(), (int)p.size(), operand, &e, sel.data(), t.data(), n, out->data(),
                     L.bytes);
    if (rc < 0) {
        *why = std::string(k->name) + "'s relayout returned " + std::to_string(rc);
        return false;
    }
    *dt = L.dtype;
    std::vector<std::vector<uint8_t>> back(sel.size());
    std::vector<RadTensor> bt = t;
    for (size_t i = 0; i < sel.size(); ++i) {
        back[i].assign(w.p[sel[i]].size(), 0xEE);
        bt[i].data = back[i].data();
    }
    rc = k->unrelayout(p.data(), (int)p.size(), operand, &e, sel.data(), out->data(), L.bytes,
                       bt.data(), n);
    for (size_t i = 0; rc >= 0 && i < sel.size(); ++i)
        if (back[i] != w.p[sel[i]]) rc = RAD_E_FORMAT;
    if (rc < 0) {
        *why = std::string(k->name) + "'s unrelayout did not give the planes back (" +
               std::to_string(rc) + ")";
        return false;
    }
    return true;
}

struct Case {
    const char* name;
    int     form;
    int     ncls;          /* 2: moe_gemm_q, 4: moe_gemm_q_mod4 */
    bool    gate_up;       /* token order and two parts; else sorted order, one part */
    int64_t M, top_k, ne;
    int64_t w[4];          /* each class's intermediate width */
    int64_t n_embd;
    bool    drop;          /* one slot routed outside this call's experts */
};

/* One library's stored experts, as table operands: a buffer an expert and view, and the pointer
 * arrays. */
struct Stored {
    std::deque<GBuf> bufs;
    std::vector<void*> ptrs[8];
    RadTensor t[8];
};

int run_case(const Check& ck, const Case& cs) {
    const Lib& ref = *ck.ref;
    const Lib& lib = *ck.lib;
    const Form& f = kForms[cs.form];
    const char* op = cs.ncls == 4 ? "moe_gemm_q_mod4" : "moe_gemm_q";
    const int nc = cs.ncls;
    char name[160];
    std::snprintf(name, sizeof name, "%s %s", f.dtype, cs.name);
    auto fail = [&](const std::string& why) {
        std::printf("  FAIL %-20s %s\n       %s\n", op, name, why.c_str());
        return 1;
    };
    int64_t Nq[4], Kq[4], N = 0, K = 0;
    for (int q = 0; q < nc; ++q) {
        Nq[q] = cs.gate_up ? 2 * cs.w[q] : cs.n_embd;
        Kq[q] = cs.gate_up ? cs.n_embd : cs.w[q];
        N = std::max(N, Nq[q]);
        K = std::max(K, Kq[q]);
    }
    const int64_t parts = cs.gate_up ? 2 : 1;

    /* The routing, as moe_scatter leaves it: slot (t, j) to expert (5t + j) % ne. */
    const int64_t T = cs.M * cs.top_k, ne = cs.ne;
    std::vector<int64_t> sorted, offs;
    auto expert_of = [&](int64_t i) -> int64_t {
        const int64_t t = i / cs.top_k, j = i % cs.top_k;
        if (cs.drop && t == 1 && j == cs.top_k - 1) return -1;
        return (5 * t + j) % ne;
    };
    for (int64_t e = 0; e < ne; ++e) {
        offs.push_back((int64_t)sorted.size());
        for (int64_t i = 0; i < T; ++i)
            if (expert_of(i) == e) sorted.push_back(i);
    }
    offs.push_back((int64_t)sorted.size());
    while ((int64_t)sorted.size() < T) sorted.push_back(-1);
    const int64_t AR = cs.gate_up ? cs.M : T;

    /* Parameters. */
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) { p.push_back(RadParam{ k, RAD_P_INT, v, 0, nullptr, 0 }); };
    auto ps = [&](const char* k, const char* v) { p.push_back(RadParam{ k, RAD_P_STR, 0, 0, v, 0 }); };
    pi("M", cs.M); pi("N", Nq[0]); pi("K", Kq[0]); pi("n_expert", ne); pi("top_k", cs.top_k);
    pi("group", 128); ps("a_order", cs.gate_up ? "token" : "sorted"); ps("dtype", f.dtype);
    static const char* const kN[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK[4] = { "K", "K_1", "K_2", "K_3" };
    for (int q = 1; q < nc; ++q) {
        pi(nc == 4 ? kN[q] : "N_odd", Nq[q]);
        pi(nc == 4 ? kK[q] : "K_odd", Kq[q]);
    }
    pi("parts", parts);

    const RadKernelInfo* kr = ref.row(op);
    const RadKernelInfo* ka = lib.select(op, p.data(), (int)p.size());
    if (!kr) return fail("no row for the op in libref");
    if (!ka) return ck.own ? fail(std::string("no row of ") + lib.info->name + " admits it") : 0;

    /* The experts' planes and each library's stored bytes. */
    const RadEncoding enc = encoding(f);
    const std::vector<int> sel_c = { rad_enc_find(&enc, "codes"), rad_enc_find(&enc, "table") };
    const std::vector<int> sel_s = f.level ? std::vector<int>{ rad_enc_find(&enc, "scale"),
                                                                rad_enc_find(&enc, "scale.1") }
                                           : std::vector<int>{ rad_enc_find(&enc, "scale") };
    std::vector<Planes> ex((size_t)ne);
    for (int64_t e = 0; e < ne; ++e)
        ex[(size_t)e] = draw(f, enc, Nq[e % nc], Kq[e % nc], 1000 + (uint64_t)(e * 7 + cs.form));
    /* Operand positions of class q's codes and scales. */
    auto opd_w = [&](int q) { return q == 0 ? 2 : 4 + 2 * q; };
    Stored st[2];
    std::vector<uint8_t> all_bytes[2];
    for (int side = 0; side < 2; ++side) {
        const RadKernelInfo* k = side ? ka : kr;
        for (int q = 0; q < nc; ++q) {
            uint32_t dtc = 0, dts = 0;
            for (int64_t e = q; e < ne; e += nc) {
                std::vector<uint8_t> bc, bs;
                std::string why;
                if (!store(k, p, opd_w(q), enc, sel_c, ex[(size_t)e], &bc, &dtc, &why) ||
                    !store(k, p, opd_w(q) + 1, enc, sel_s, ex[(size_t)e], &bs, &dts, &why))
                    return fail(why);
                all_bytes[side].insert(all_bytes[side].end(), bc.begin(), bc.end());
                all_bytes[side].insert(all_bytes[side].end(), bs.begin(), bs.end());
                st[side].bufs.emplace_back();
                st[side].bufs.back().assign(bc);
                st[side].ptrs[2 * q].push_back(st[side].bufs.back().data());
                st[side].bufs.emplace_back();
                st[side].bufs.back().assign(bs);
                st[side].ptrs[2 * q + 1].push_back(st[side].bufs.back().data());
            }
            /* The declared views: codes at the weight's extents, a joined scale view too, a plain
             * scale at its plane's. */
            const int64_t nq = (ne + nc - 1 - q) / nc;
            const int64_t scols = f.level ? Kq[q] : (Kq[q] + f.group - 1) / f.group;
            for (int v = 0; v < 2; ++v) {
                RadTensor& x = st[side].t[2 * q + v];
                x = RadTensor{};
                x.data = st[side].ptrs[2 * q + v].data();
                x.dtype = v ? dts : dtc;
                x.rank = 3;
                x.shape[0] = nq;
                x.shape[1] = Nq[q];
                x.shape[2] = v ? scols : Kq[q];
                x.stride[0] = 0;
                x.stride[1] = x.shape[2];
                x.stride[2] = 1;
            }
        }
    }
    if (all_bytes[0] != all_bytes[1])
        return fail(std::string("libref and ") + lib.info->name + " store different bytes");

    /* The activation (E4M3 codes and an f32 scale per 128), the routing, and the output. */
    const int64_t SG = (K + 127) / 128;
    std::vector<uint8_t> aq((size_t)(AR * K)), asb((size_t)(AR * SG * 4)), yb((size_t)(T * N * 2));
    for (int64_t i = 0; i < AR * K; ++i)
        rad_store_f32(aq.data(), RAD_F8E4M3, i,
                      ((float)(mix(77, (uint64_t)i) % 2001) / 1000.0f - 1.0f) * 6.0f);
    for (int64_t i = 0; i < AR * SG; ++i)
        rad_store_f32(asb.data(), RAD_F32, i, 0.01f + 0.004f * (float)(i % 9));
    for (int64_t i = 0; i < T * N; ++i) rad_store_f32(yb.data(), RAD_BF16, i, 7.0f);
    std::vector<uint8_t> sb((size_t)T * 4), ob((size_t)(ne + 1) * 4);
    for (int64_t i = 0; i < T; ++i) ((int32_t*)sb.data())[i] = (int32_t)sorted[(size_t)i];
    for (int64_t i = 0; i <= ne; ++i) ((int32_t*)ob.data())[i] = (int32_t)offs[(size_t)i];

    auto tensor = [](void* d, uint32_t dt, std::vector<int64_t> shape) {
        RadTensor t{};
        t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
        for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
        rad_tensor_pack(&t);
        return t;
    };
    /* Guarded copies, one set a run. */
    struct Io { GBuf a, as, s, o, y; };
    auto make_io = [&](Io& io) {
        io.a.assign(aq); io.as.assign(asb); io.s.assign(sb); io.o.assign(ob); io.y.assign(yb);
    };
    auto launch = [&](const RadKernelInfo* k, Io& io, const RadTensor* tabs) {
        std::vector<RadTensor> t((size_t)(nc == 4 ? 13 : 9));
        t[0] = tensor(io.a.data(), RAD_F8E4M3, { AR, K });
        t[1] = tensor(io.as.data(), RAD_F32, { AR, SG });
        t[2] = tabs[0]; t[3] = tabs[1];
        t[4] = tensor(io.s.data(), RAD_I32, { T });
        t[5] = tensor(io.o.data(), RAD_I32, { ne + 1 });
        for (int q = 1; q < nc; ++q) { t[(size_t)(4 + 2 * q)] = tabs[2 * q]; t[(size_t)(5 + 2 * q)] = tabs[2 * q + 1]; }
        t.back() = tensor(io.y.data(), RAD_BF16, { T, N });
        RadArgs a{};
        a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
        a.world_size = 1;
        return k->launch(&a, nullptr);
    };
    auto intact = [&](Io& io, Stored& s, const char* who) -> std::string {
        if (io.y.guard_hit()) return std::string(who) + " wrote outside the output";
        if (std::memcmp(io.a.data(), aq.data(), aq.size()) ||
            std::memcmp(io.as.data(), asb.data(), asb.size()) ||
            std::memcmp(io.s.data(), sb.data(), sb.size()) ||
            std::memcmp(io.o.data(), ob.data(), ob.size()) || io.a.guard_hit() ||
            io.as.guard_hit() || io.s.guard_hit() || io.o.guard_hit())
            return std::string(who) + " changed an input";
        for (const GBuf& g : s.bufs)
            if (g.guard_hit()) return std::string(who) + " touched a stored expert's guard";
        return "";
    };

    Io ir, ia;
    make_io(ir);
    make_io(ia);
    int rc = launch(kr, ir, st[0].t);
    if (rc < 0) return fail("libref returned " + std::to_string(rc));
    rc = launch(ka, ia, st[1].t);
    if (rc < 0) return fail(std::string(lib.info->name) + " returned " + std::to_string(rc));
    for (const std::string& w : { intact(ir, st[0], "libref"), intact(ia, st[1], lib.info->name) })
        if (!w.empty()) return fail(w);
    if (!std::memcmp(ir.y.data(), yb.data(), yb.size()))
        return fail("libref's launch changed no output value, so the comparison is vacuous");
    double rel = 0;
    std::string why;
    if (!same_values(RAD_BF16, ir.y.data(), ia.y.data(), T * N, 8e-3, &rel, &why))
        return fail(std::string(lib.info->name) + " against libref: " + why);
    const double rel_lib = rel;

    /* libref over the logical form: table entries as E4M3 codes, scale times level as bf16, the
     * stacked form. */
    {
        std::vector<std::vector<uint8_t>> lc(nc), ls(nc);
        RadTensor tabs[8];
        for (int q = 0; q < nc; ++q) {
            const int64_t nq = (ne + nc - 1 - q) / nc, n = Nq[q], k = Kq[q];
            const int64_t g = f.group, sg = (k + g - 1) / g;
            lc[(size_t)q].assign((size_t)(nq * n * k), 0);
            ls[(size_t)q].assign((size_t)(nq * n * sg * 2), 0);
            for (int64_t e = q, x = 0; e < ne; e += nc, ++x) {
                const Planes& w = ex[(size_t)e];
                const int ic = sel_c[0], is = sel_s[0];
                const float lv = f.level ? rad_load_f32(w.p[sel_s[1]].data(), RAD_F32, 0) : 1.0f;
                const float* tab = (const float*)w.p[sel_c[1]].data();
                const int64_t rbc = rad_enc_row_bytes(enc.plane[ic].dtype, k);
                const int64_t rbs = rad_enc_row_bytes(f.sdt, sg);
                for (int64_t r = 0; r < n; ++r) {
                    for (int64_t c = 0; c < k; ++c)
                        rad_store_f32(lc[(size_t)q].data(), RAD_F8E4M3, (x * n + r) * k + c,
                                      tab[(int)rad_load_f32(w.p[ic].data() + r * rbc,
                                                            enc.plane[ic].dtype, c)]);
                    for (int64_t c = 0; c < sg; ++c)
                        rad_store_f32(ls[(size_t)q].data(), RAD_BF16, (x * n + r) * sg + c,
                                      rad_load_f32(w.p[is].data() + r * rbs, f.sdt, c) * lv);
                }
            }
            tabs[2 * q] = tensor(lc[(size_t)q].data(), RAD_F8E4M3, { nq, n, k });
            tabs[2 * q + 1] = tensor(ls[(size_t)q].data(), RAD_BF16, { nq, n, sg });
        }
        Io il;
        make_io(il);
        rc = launch(kr, il, tabs);
        if (rc < 0) return fail("libref over the logical form returned " + std::to_string(rc));
        if (std::memcmp(il.y.data(), ir.y.data(), yb.size()) != 0)
            return fail("libref over the stored form and over the logical form differ");
    }

#ifdef AVX_CHECK_CORE_DECODER
    /* The core's decoder. The rotation undone on both sides where it tiles K; otherwise both stay
     * in the rotated basis, where the product is the same. */
    double rel_or = 0, rel_oa = 0;
    {
        bool rot = true;
        for (int q = 0; q < nc; ++q) rot = rot && Kq[q] % 128 == 0;
        std::vector<std::vector<float>> W((size_t)ne);
        for (int64_t e = 0; e < ne; ++e) {
            RadEncoding d = enc;
            if (!rot) d.transform[0] = 0;
            const Planes& w = ex[(size_t)e];
            const void* data[RAD_ENC_MAX_PLANES] = {};
            for (int i = 0; i < d.n_planes; ++i) data[i] = w.p[i].data();
            rad::EncodedView v;
            rad::enc_view(d, w.n, w.k, data, &v);
            W[(size_t)e].assign((size_t)(w.n * w.k), 0.0f);
            std::string dw;
            if (rad::enc_decode_rows(v, 0, w.n, W[(size_t)e].data(), &dw) != RAD_OK)
                return fail("the core decoder refused expert " + std::to_string(e) + ": " + dw);
        }
        std::vector<float> al((size_t)(AR * K));
        for (int64_t r = 0; r < AR; ++r) {
            float* x = al.data() + r * K;
            for (int64_t c = 0; c < K; ++c)
                x[c] = rad_load_f32(aq.data(), RAD_F8E4M3, r * K + c) *
                       rad_load_f32(asb.data(), RAD_F32, r * SG + c / 128);
            if (rot)
                for (int64_t g = 0; g < K; g += 128) {
                    rad::fwht_inplace(x + g, 128);
                    for (int64_t c = g; c < g + 128; ++c) x[c] /= 128.0f;
                }
        }
        std::vector<uint8_t> yo((size_t)(T * N * 2), 0);
        const int64_t fy = N / parts;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t flat = sorted[(size_t)i];
            int64_t e = 0;
            while (e + 1 < ne && offs[(size_t)(e + 1)] <= i) ++e;
            const int64_t tok = flat < 0 ? -1 : (cs.gate_up ? flat / cs.top_k : i);
            const int64_t n = Nq[e % nc], k = Kq[e % nc], fc = n / parts;
            for (int64_t col = 0; col < N; ++col) {
                const int64_t pp = col / fy, j = col % fy;
                double s = 0;
                if (tok >= 0 && j < fc)
                    for (int64_t c = 0; c < k; ++c)
                        s += (double)W[(size_t)e][(size_t)((pp * fc + j) * k + c)] *
                             (double)al[(size_t)(tok * K + c)];
                rad_store_f32(yo.data(), RAD_BF16, i * N + col, (float)s);
            }
        }
        if (!same_values(RAD_BF16, yo.data(), ir.y.data(), T * N, 1e-2, &rel_or, &why))
            return fail("libref against the core decoder: " + why);
        if (!same_values(RAD_BF16, yo.data(), ia.y.data(), T * N, 1e-2, &rel_oa, &why))
            return fail(std::string(lib.info->name) + " against the core decoder: " + why);
    }
#endif

    if (ck.threads > 0 && lib.par_force) {
        std::string pw;
        if (!parallel_rerun(lib, ck.threads, { ia.y.data() }, { ia.y.size() },
                            [&] { std::memcpy(ia.y.data(), yb.data(), yb.size()); },
                            [&] { return launch(ka, ia, st[1].t); }, &pw))
            return fail(pw);
    }
    if (ck.verbose) {
#ifdef AVX_CHECK_CORE_DECODER
        std::printf("  %-22s %-46s rel_l2=%.3e, vs decoder %.3e / %.3e ok\n", op, name, rel_lib,
                    rel_or, rel_oa);
#else
        std::printf("  %-22s %-46s rel_l2=%.3e ok (no core decoder in this build)\n", op, name,
                    rel_lib);
#endif
    }
    return 0;
}

/* Each library's hooks refuse the selections libref refuses, with the same code: a form's planes
 * under another codebook dtype, under a dtype that is no codebook, unrotated, or at an operand
 * that is neither codes nor scales; and store a single plane as it is. */
int hook_refusals(const Check& ck) {
    struct R { int form; const char* dtype; int operand; bool scale_view; bool unrotated; };
    static const R kR[] = {
        { 1, "w4nl32a8h", 3, true,  false },   /* scale per 64 named as per 32 */
        { 3, "w4nl64a8h", 2, false, false },   /* five-bit codes named four-bit */
        { 1, "fp8a8",     2, false, false },   /* no codebook */
        { 1, "w4nl64a8h", 2, false, true },    /* unrotated */
        { 1, "w4nl64a8h", 4, false, false },   /* sorted_tok's position */
        { 0, "w4nla8h",   3, true,  false },   /* one plane: stored as it is */
    };
    int fails = 0;
    for (const R& r : kR) {
        const Form& f = kForms[r.form];
        RadEncoding e = encoding(f);
        if (r.unrotated) e.transform[0] = 0;
        std::vector<int> sel;
        if (r.scale_view) {
            sel.push_back(rad_enc_find(&e, "scale"));
            if (f.level) sel.push_back(rad_enc_find(&e, "scale.1"));
        } else {
            sel = { rad_enc_find(&e, "codes"), rad_enc_find(&e, "table") };
        }
        Planes w = draw(f, e, 32, 128, 5);
        std::vector<RadTensor> t = selection(e, sel, w);
        std::vector<RadParam> p = { RadParam{ "dtype", RAD_P_STR, 0, 0, r.dtype, 0 } };
        int got[2] = { 0, 0 };
        for (int side = 0; side < 2; ++side) {
            const Lib& L = side ? *ck.lib : *ck.ref;
            const RadKernelInfo* k = L.row("moe_gemm_q");
            RadLayout lay{};
            got[side] = k && k->layout ? k->layout(p.data(), 1, r.operand, &e, sel.data(),
                                                   t.data(), (int)sel.size(), &lay)
                                       : RAD_E_UNSUPPORTED;
        }
        const int want = sel.size() == 1 ? RAD_E_UNSUPPORTED : RAD_E_DTYPE;
        if (got[0] != want || got[1] != want) {
            std::printf("  FAIL %-20s layout of %s planes as %s at operand %d\n       libref %d, "
                        "%s %d, expected %d\n", "moe_gemm_q", f.dtype, r.dtype, r.operand, got[0],
                        ck.lib->info->name, got[1], want);
            ++fails;
        }
    }
    if (!fails && ck.verbose)
        std::printf("  %-22s %-46s same refusals\n", "moe_gemm_q", "codebook layout hooks");
    return fails;
}

}  // namespace cb

/* Every form at four geometries, which reach each of libavx's decodes at each level: a
 * decode-sized gate_up over two classes (one or two rows an expert, the fused form); a
 * prefill-sized down over four classes (about a hundred rows an expert, the panel form, a partial
 * row block, a dropped slot, an odd output width); a down whose K is not whole AVX-512 decode
 * chunks and ends in a partial scale group (the generic decode there, AVX2's chunks); and a gate_up
 * whose K is whole chunks but not whole 128-code lines (AVX-512's chunk decode). Returns the
 * failures, having printed them, and counts the cases run. */
int codebook_cases(const Check& ck, int* run) {
    static const cb::Case kCases[] = {
        { "gate_up, 2 classes 128 / 96 wide, M 5", 0, 2, true, 5, 2, 8, { 128, 96, 0, 0 }, 256,
          false },
        { "down, 4 classes 256 / 128 / 128 / 128, M 130", 0, 4, false, 130, 3, 4,
          { 256, 128, 128, 128 }, 97, true },
        { "down, 2 classes K 208 / 144, N 61, M 9", 0, 2, false, 9, 2, 6, { 208, 144, 0, 0 }, 61,
          false },
        { "gate_up, 2 classes 64 / 32 wide, K 160, M 3", 0, 2, true, 3, 2, 4, { 64, 32, 0, 0 },
          160, false },
    };
    int fails = 0;
    for (int fm = 0; fm < 4; ++fm)
        for (cb::Case c : kCases) {
            c.form = fm;
            const char* op = c.ncls == 4 ? "moe_gemm_q_mod4" : "moe_gemm_q";
            if (!ck.only_op.empty() && ck.only_op != op) continue;
            ++*run;
            fails += cb::run_case(ck, c);
        }
    if (ck.only_op.empty() || ck.only_op == "moe_gemm_q") {
        ++*run;
        fails += cb::hook_refusals(ck) ? 1 : 0;
    }
    return fails;
}

/* ================================================================== the image generator */
/* Forms libref's descriptions do not draw: several entries and an empty one, a modulation row as a
 * column slice of the shared [n_seq, 4D] projection, the entry flags and pairings the sampler
 * reads, attention over entries of different lengths and prefixes with a missing block and
 * column-sliced projections, images whose rows are not back to back, and the signed gather libref
 * declines to describe. The sampler, the gate and the copies are held to zero error: they are
 * libref's arithmetic bit for bit. */
void gen_cases(std::vector<Hand>& hs) {
    const auto zero = [](int64_t) { return 0.0f; };
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const std::string sfx = std::string(", ") + dn;
        const double norm_tol = dt == RAD_F32 ? 1e-5 : 6e-3;

        /* ---- gather_rows2: both sources, and rows past either ------------------------------ */
        {
            Hand h(("a and b rows, past both ends" + sfx).c_str(), "gather_rows2", 0.0);
            h.dense(dt, { 5, 67 }, hfill(201))
             .dense(dt, { 3, 67 }, hfill(202))
             .idx({ 0, -1, 4, -3, 5, -4, 2 })
             .dense(dt, { 7, 67 }, hfill(203), true)
             .pi("M", 7).pi("n", 67).ps("dtype", dn);
            hs.push_back(std::move(h));
        }

        /* ---- the DiT's norms: three entries (one empty), the modulation at a 4D pitch ------ */
        {
            const int64_t R = 11, D = 160, W = 4 * D;
            Hand h(("three entries, one empty, s at a 4D pitch" + sfx).c_str(), "ln_mod", norm_tol);
            h.dense(dt, { R, D }, hfill(210, 2.0f))
             .view(dt, { 3, D }, { W, 1 }, 3 * W, hfill(211, 0.5f))
             .idx({ 0, 4, 4, R })
             .dense(dt, { R, D }, zero, true)
             .pi("M", R).pi("n", D).pf("eps", 1e-6).ps("dtype", dn);
            hs.push_back(std::move(h));

            Hand g(("three entries, one empty, g at a 4D pitch" + sfx).c_str(), "gate_add", 0.0);
            g.dense(dt, { R, D }, hfill(212))
             .view(dt, { 3, D }, { W, 1 }, 3 * W, hfill(213))
             .idx({ 0, 4, 4, R })
             .dense(dt, { R, D }, hfill(214), true)
             .pi("M", R).pi("n", D).ps("dtype", dn);
            hs.push_back(std::move(g));

            for (const bool gated : { true, false }) {
                Hand q((std::string(gated ? "gated" : "the first block's, no branch") +
                        ", three entries, 4D pitch" + sfx).c_str(), "ln_mod_quant_fp8", 1e-2);
                if (gated) {
                    q.dense(dt, { R, D }, hfill(215))
                     .view(dt, { 3, D }, { W, 1 }, 3 * W, hfill(216));
                } else {
                    q.none().none();
                }
                q.view(dt, { 3, D }, { W, 1 }, 3 * W, hfill(217, 0.5f))
                 .idx({ 0, 4, 4, R })
                 .dense(dt, { R, D }, hfill(218, 2.0f), true)
                 .dense(RAD_F8E4M3, { R, D }, zero, true)
                 .dense(RAD_F32, { R, D / 32 }, zero, true)
                 .dense(RAD_BF16, { R, D }, zero, true)
                 .pi("M", R).pi("n", D).pf("eps", 1e-6).pi("group", 32)
                 .pi("gated", gated ? 1 : 0).ps("dtype", dn);
                hs.push_back(std::move(q));
            }
        }

        /* ---- the sampler's state: seeds, flags, pairs and lanes ----------------------------- */
        {
            /* Three entries over lanes 2, 0 and 3 of four ten-row lanes, two of them first steps,
             * seven channels so a Philox block of four straddles rows. */
            static const uint32_t kSeeds[6] = { 0x9e3779b9u, 1u, 7u, 0u, 0xdeadbeefu, 0x80000001u };
            Hand h(("two first steps of three, n 7" + sfx).c_str(), "randn_lanes", 0.0);
            h.idx({ 1, 0, 5 })
             .raw(RAD_U32, { 6 }, kSeeds)
             .idx({ 2, 0, 3 })
             .idx({ 0, 4, 6, 13 })
             .dense(dt, { 40, 7 }, hfill(220), true)
             .pi("M", 13).pi("n", 7).pi("stride", 10).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const int64_t n : { 64, 37 }) {
            /* Entry 0 guided against entry 1 (a negative conditioning: pair -2, its lane left as it
             * is), with a preview and a repaint; entry 2 unguided with a repaint; entry 3
             * unguided, neither. */
            static const uint32_t kSeeds[8] = { 3u, 0u, 4u, 0u, 5u, 9u, 0xfffffffeu, 2u };
            Hand h(("guided, parked, repainted, previewed, n " + std::to_string(n) + sfx).c_str(),
                   "flow_euler", 0.0);
            h.dense(dt, { 17, n }, hfill(221))
             .dense(RAD_F32, { 4 }, [](int64_t i) { return 0.8f - 0.1f * (float)i; })
             .dense(RAD_F32, { 4 }, [](int64_t i) { return 0.55f - 0.1f * (float)i; })
             .dense(RAD_F32, { 4 }, [](int64_t) { return 3.5f; })
             .idx({ 1, -2, -1, -1 })
             .idx({ 0, 1, 2, 3 })
             .idx({ 6, 0, 2, 0 })
             .raw(RAD_U32, { 8 }, kSeeds)
             .idx({ 0, 5, 10, 13, 17 })
             .dense(dt, { 24, n }, hfill(222))
             .dense(RAD_F32, { 24 }, [](int64_t i) { return 0.25f * (float)(i % 5); })
             .dense(dt, { 24, n }, hfill(223), true)
             .dense(dt, { 24, n }, hfill(224), true)
             .pi("M", 17).pi("n", n).pi("stride", 6).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* The k-diffusion input: entry 0 a first step guided against entry 1 (pair -2, which
             * only its positive entry writes), entry 2 a later step, unguided, in lane 3. */
            static const uint32_t kSeeds[6] = { 11u, 0u, 0u, 0u, 0xfeedbeefu, 3u };
            Hand h(("first, parked, later, n 4" + sfx).c_str(), "kd_input", 0.0);
            h.dense(RAD_F32, { 3 }, [](int64_t i) { return 14.6f - 6.0f * (float)i; })
             .idx({ 1, -2, -1 })
             .idx({ 0, 1, 3 })
             .idx({ 1, 0, 0 })
             .raw(RAD_U32, { 6 }, kSeeds)
             .idx({ 0, 5, 10, 13 })
             .dense(RAD_F32, { 24, 4 }, hfill(227), true)
             .dense(dt, { 13, 4 }, zero, true)
             .pi("M", 13).pi("n", 4).pi("stride", 6).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const char* pred : { "eps", "v" }) {
            /* Every sampler: entry 0 Euler guided against entry 1 with a preview, entry 2 Euler
             * ancestral at step 7, entry 3 DPM++ 2M past its first step, entry 4 DPM++ 2M's first
             * step, entry 5 Euler ancestral to sigma 0, entry 6 DPM++ 2M to sigma 0. */
            static const uint32_t kSeeds[14] = { 3u, 0u, 4u, 0u, 5u, 9u, 6u, 1u, 7u, 0u, 8u, 0u,
                                                 9u, 2u };
            Hand h(("seven entries, every sampler, " + std::string(pred) + sfx).c_str(), "kd_step",
                   0.0);
            static const float kS[7]  = { 6.0f, 6.0f, 3.5f, 1.25f, 9.0f, 0.4f, 0.3f };
            static const float kSn[7] = { 4.5f, 4.5f, 2.0f, 0.7f, 6.5f, 0.0f, 0.0f };
            static const float kSp[7] = { 0.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.9f, 0.6f };
            h.dense(dt, { 21, 5 }, hfill(228))
             .dense(RAD_F32, { 7 }, [](int64_t i) { return kS[i]; })
             .dense(RAD_F32, { 7 }, [](int64_t i) { return kSn[i]; })
             .dense(RAD_F32, { 7 }, [](int64_t i) { return kSp[i]; })
             .dense(RAD_F32, { 7 }, [](int64_t) { return 6.5f; })
             .idx({ 1, -2, -1, -1, -1, -1, -1 })
             .idx({ 0, 1, 2, 3, 4, 5, 6 })
             .idx({ 4, 0, 0, 0, 1, 0, 0 })
             .raw(RAD_U32, { 14 }, kSeeds)
             .idx({ 0, 0, 1, 2, 2, 1, 2 })
             .idx({ 0, 0, 7, 4, 0, 20, 21 })
             .idx({ 0, 3, 6, 9, 12, 15, 18, 21 })
             .dense(RAD_F32, { 28, 5 }, hfill(229, 3.0f), true)
             .dense(RAD_F32, { 28, 5 }, hfill(230), true)
             .dense(RAD_F32, { 28, 5 }, zero, true)
             .pi("M", 21).pi("n", 5).pi("stride", 4).ps("pred", pred).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            Hand h(("a mask column into three lanes" + sfx).c_str(), "lane_store", 0.0);
            h.dense(dt, { 9, 1 }, hfill(225))
             .idx({ 3, 0, 1 })
             .dense(dt, { 40, 1 }, hfill(226), true)
             .pi("M", 9).pi("n", 1).pi("stride", 10).ps("dtype", dn);
            hs.push_back(std::move(h));
        }

        /* ---- the decoder ---------------------------------------------------------------------- */
        {
            /* Pixels past [-1, 1], NaN, and ties at the byte (0 maps to 127.5). */
            static const float kPx[12] = { -2.0f, -1.0f, -0.999f, 0.0f, 0.5f, 1.0f, 1.5f, NAN,
                                           0.25f, -0.25f, 0.998f, -INFINITY };
            Hand h(("clamps, NaN and ties" + sfx).c_str(), "to_rgba8", 0.0);
            h.dense(dt, { 3, 4 }, [](int64_t i) { return kPx[i]; })
             .dense(RAD_U8, { 3, 4 }, zero, true)
             .pi("M", 3).pi("n", 4).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const int64_t cout : { 52, 4 }) {
            /* Output rows five elements apart from the next (one band a row), and a 1 x 1 layer
             * over input rows three apart (patches, not the input read in place). */
            const int64_t H = 6, Wd = 7, cin = 20, yr = Wd * cout + 5, xr = Wd * cin + 3;
            Hand h(("3 x 3, output rows at a wider pitch, cout " + std::to_string(cout) + sfx)
                       .c_str(), "conv2d_nhwc", norm_tol);
            h.dense(dt, { H, Wd * cin }, hfill(230))
             .dense(dt, { cout, cin, 3, 3 }, hfill(231, 0.1f))
             .dense(dt, { cout }, hfill(232, 0.1f))
             .view(dt, { H, Wd * cout }, { yr, 1 }, H * yr, hfill(233), true)
             .pi("M", H * Wd).pi("cin", cin).pi("cout", cout).pi("k", 3).pi("stride", 1)
             .ps("pad", "same").pi("up", 1).ps("dtype", dn);
            hs.push_back(std::move(h));

            Hand g(("1 x 1, input rows at a wider pitch, cout " + std::to_string(cout) + sfx)
                       .c_str(), "conv2d_nhwc", norm_tol);
            g.view(dt, { H, Wd * cin }, { xr, 1 }, H * xr, hfill(234))
             .dense(dt, { cout, cin, 1, 1 }, hfill(235, 0.2f))
             .dense(dt, { cout }, hfill(236, 0.1f))
             .dense(dt, { H, Wd * cout }, zero, true)
             .pi("M", H * Wd).pi("cin", cin).pi("cout", cout).pi("k", 1).pi("stride", 1)
             .ps("pad", "same").pi("up", 1).ps("dtype", dn);
            hs.push_back(std::move(g));
        }
        {
            /* An odd 13-column picture downsampled by the stride-2 `down` form, and upsampled. */
            const int64_t cin = 12, cout = 52;
            Hand h(("stride 2 down to 13 x 13" + sfx).c_str(), "conv2d_nhwc", norm_tol);
            h.dense(dt, { 26, 26 * cin }, hfill(237))
             .dense(dt, { cout, cin, 3, 3 }, hfill(238, 0.1f))
             .dense(dt, { cout }, hfill(239, 0.1f))
             .dense(dt, { 13, 13 * cout }, zero, true)
             .pi("M", 169).pi("cin", cin).pi("cout", cout).pi("k", 3).pi("stride", 2)
             .ps("pad", "down").pi("up", 1).ps("dtype", dn);
            hs.push_back(std::move(h));

            Hand g(("upsampled 13 x 13 to 26 x 26" + sfx).c_str(), "conv2d_nhwc", norm_tol);
            g.dense(dt, { 13, 13 * cin }, hfill(240))
             .dense(dt, { cout, cin, 3, 3 }, hfill(241, 0.1f))
             .dense(dt, { cout }, hfill(242, 0.1f))
             .dense(dt, { 26, 26 * cout }, zero, true)
             .pi("M", 676).pi("cin", cin).pi("cout", cout).pi("k", 3).pi("stride", 1)
             .ps("pad", "same").pi("up", 2).ps("dtype", dn);
            hs.push_back(std::move(g));
        }
        for (const int64_t ft : { 1, 2 }) {
            Hand h(("576 -> 288 pattern, ft " + std::to_string(ft) + sfx).c_str(), "d2s_add", 0.0);
            h.dense(dt, { 3, 5 * 48 }, hfill(243))
             .dense(dt, { 6, 10 * 24 }, hfill(244), true)
             .pi("M", 60).pi("cin", 48).pi("cout", 24).pi("ft", ft).ps("dtype", dn);
            hs.push_back(std::move(h));

            Hand g(("96 -> 192 pattern, ft " + std::to_string(ft) + sfx).c_str(), "s2d_avg_add",
                   0.0);
            g.dense(dt, { 6, 10 * 12 }, hfill(245))
             .dense(dt, { 3, 5 * 24 }, hfill(246), true)
             .pi("M", 15).pi("cin", 12).pi("cout", 24).pi("ft", ft).ps("dtype", dn);
            hs.push_back(std::move(g));
        }

        /* ---- the rotaries over three position components ---------------------------------- */
        {
            const int64_t M = 9, hd = 64, nh = 2, nkv = 2, W = (nh + 2 * nkv) * hd;
            auto pos = [M](int64_t i) {
                const int64_t c = i / M, m = i % M;
                return (float)(c == 0 ? 3 : c == 1 ? m / 3 : m % 3);
            };
            Hand h(("the DiT's 8 12 12 adjacent pairs over (frame, row, column)" + sfx).c_str(),
                   "qk_norm_rope", norm_tol);
            h.dense(dt, { M, W }, hfill(247), true)
             .dense(RAD_I32, { 3, M }, pos)
             .dense(dt, { hd }, hfill(248, 0.1f))
             .dense(dt, { hd }, hfill(249, 0.1f))
             .pi("M", M).pi("head_dim", hd).pi("n_head", nh).pi("n_head_kv", nkv)
             .pf("theta", 10000.0).pf("eps", 1e-6).ps("mode", "gptj_blocks")
             .pi("rotary_dim", hd).ps("sections", "8 12 12");
            hs.push_back(std::move(h));

            Hand g(("Qwen3-VL's interleaved M-RoPE over three components" + sfx).c_str(),
                   "qk_norm_rope", norm_tol);
            g.dense(dt, { M, W }, hfill(250), true)
             .dense(RAD_I32, { 3, M }, pos)
             .dense(dt, { hd }, hfill(251, 0.1f))
             .dense(dt, { hd }, hfill(252, 0.1f))
             .pi("M", M).pi("head_dim", hd).pi("n_head", nh).pi("n_head_kv", nkv)
             .pf("theta", 5e6).pf("eps", 1e-6).ps("mode", "imrope").pi("rotary_dim", hd)
             .ps("sections", "12 10 10");
            hs.push_back(std::move(g));
        }

        /* ---- attention over entries of different prefixes ---------------------------------- *
         * Two entries of 37 and 8 rows (one query block and part of a second, and a short one)
         * against prefixes of 50 and 20 keys; entry 0's second block is missing (-1: zero keys
         * that still take their softmax share). q, k, v and out are column slices of a fused
         * projection, rows `W` apart. */
        for (const int64_t hd : { 64, 40 })
            for (const bool i8 : { false, true })
                for (const int causal : { 0, 1 }) {
                    const int64_t T = 45, kvh = 2, gqa = 2, qh = 4, BS = 16, NB = 6;
                    const int64_t W = (qh + 2 * kvh) * hd;
                    char nm[128];
                    std::snprintf(nm, sizeof nm, "2 entries, prefixes 50/20, a missing block, "
                                  "hd %lld%s%s%s", (long long)hd, i8 ? ", i8" : "",
                                  causal ? ", causal" : "", sfx.c_str());
                    Hand h(nm, "attn_prefix_self", dt == RAD_F32 ? 1e-4 : 2e-2);
                    h.view(dt, { T, qh * hd }, { W, 1 }, T * W, hfill(260))
                     .view(dt, { T, kvh * hd }, { W, 1 }, T * W, hfill(261))
                     .view(dt, { T, kvh * hd }, { W, 1 }, T * W, hfill(262))
                     .dense(dt, { NB, kvh, BS, 2 * hd }, hfill(263))
                     .dense(RAD_I32, { 2, 4 }, [](int64_t i) {
                         static const int32_t kBt[8] = { 0, -1, 2, 3, 4, 5, 1, -1 };
                         return (float)kBt[i];
                     })
                     .idx({ 50, 20 })
                     .idx({ 0, 37, T })
                     .none().none();
                    if (i8) {
                        h.dense(RAD_I8, { T, kvh * hd }, [](int64_t i) {
                             return (float)((int)((i * 37) % 255) - 127);
                         })
                         .dense(RAD_F32, { T, kvh }, [](int64_t i) {
                             return 0.004f + 0.0005f * (float)(i % 7);
                         });
                    } else {
                        h.none().none();
                    }
                    h.view(dt, { T, qh * hd }, { W, 1 }, T * W, zero, true)
                     .pi("q_len", T).pi("head_dim", hd).pi("gqa", gqa).pi("block_size", BS)
                     .pi("causal", causal).ps("q_dtype", dn).ps("kv_dtype", dn)
                     .pf("scale", 1.0 / std::sqrt((double)hd)).ps("qk", i8 ? "i8" : "bf16");
                    hs.push_back(std::move(h));
                }
        /* ---- the prefix-free form: no cache, block table or prefix lengths; two entries of
         * 37 and 8 rows, bf16 and int8 scores, columns of a fused projection. */
        for (const bool pv8 : { false, true })
        for (const bool i8 : { false, true })
            for (const int causal : { 0, 1 }) {
                const int64_t T = 45, kvh = 2, gqa = 1, qh = 2, hd = 64;
                const int64_t W = (qh + 2 * kvh) * hd;
                char nm[128];
                std::snprintf(nm, sizeof nm, "no prefix, 2 entries%s%s%s%s", i8 ? ", i8" : "",
                              pv8 ? ", fp8 P.V" : "", causal ? ", causal" : "", sfx.c_str());
                Hand h(nm, "attn_prefix_self", dt == RAD_F32 ? 1e-4 : 2e-2);
                h.view(dt, { T, qh * hd }, { W, 1 }, T * W, hfill(280))
                 .view(dt, { T, kvh * hd }, { W, 1 }, T * W, hfill(281));
                if (pv8) h.dense(RAD_F8E4M3, { T, kvh * hd }, hfill(282, 64.0f));
                else     h.view(dt, { T, kvh * hd }, { W, 1 }, T * W, hfill(282));
                h.none().none().none()
                 .idx({ 0, 37, T })
                 .none().none();
                if (i8) {
                    h.dense(RAD_I8, { T, kvh * hd }, [](int64_t i) {
                         return (float)((int)((i * 41) % 255) - 127);
                     })
                     .dense(RAD_F32, { T, kvh }, [](int64_t i) {
                         return 0.004f + 0.0005f * (float)(i % 5);
                     });
                } else {
                    h.none().none();
                }
                h.view(dt, { T, qh * hd }, { W, 1 }, T * W, zero, true);
                if (pv8) h.dense(RAD_F32, { kvh * hd }, [](int64_t i) {
                    return 0.01f + 0.001f * (float)(i % 11);
                });
                else     h.none();
                h.pi("q_len", T).pi("head_dim", hd).pi("gqa", gqa).pi("block_size", 16)
                 .pi("causal", causal).ps("q_dtype", dn).ps("kv_dtype", dn).pi("max_ctx", 0)
                 .pf("scale", 1.0 / std::sqrt((double)hd)).ps("qk", i8 ? "i8" : "bf16")
                 .ps("pv", pv8 ? "fp8" : "bf16");
                hs.push_back(std::move(h));
            }
        {
            /* One segment of a longer sequence: cu is its two bounding offsets, rows [5, 30), and
             * the rows outside it stay as they were. */
            const int64_t T = 30, kvh = 1, hd = 64, BS = 16;
            Hand h(("a segment at rows 5..30, causal" + sfx).c_str(), "attn_prefix_self",
                   dt == RAD_F32 ? 1e-4 : 2e-2);
            h.dense(dt, { T, kvh * hd }, hfill(264))
             .dense(dt, { T, kvh * hd }, hfill(265))
             .dense(dt, { T, kvh * hd }, hfill(266))
             .dense(dt, { 3, kvh, BS, 2 * hd }, hfill(267))
             .dense(RAD_I32, { 1, 3 }, [](int64_t i) { return (float)(2 - i); })
             .idx({ 40 })
             .idx({ 5, T })
             .none().none().none().none()
             .dense(dt, { T, kvh * hd }, hfill(268), true)
             .pi("q_len", T).pi("head_dim", hd).pi("gqa", 1).pi("block_size", BS)
             .pi("causal", 1).ps("q_dtype", dn).ps("kv_dtype", dn).pf("scale", 0.125);
            hs.push_back(std::move(h));
        }
    }

    /* ---- an fp8 cache read with per-(entry, kv head) descales ------------------------------ */
    {
        const int64_t T = 20, kvh = 2, gqa = 1, hd = 64, BS = 16;
        Hand h("fp8 cache, k and v descales, bf16 rows", "attn_prefix_self", 2e-2);
        h.dense(RAD_BF16, { T, kvh * gqa * hd }, hfill(270))
         .dense(RAD_BF16, { T, kvh * hd }, hfill(271))
         .dense(RAD_BF16, { T, kvh * hd }, hfill(272))
         .dense(RAD_F8E4M3, { 4, kvh, BS, 2 * hd }, hfill(273, 8.0f))
         .dense(RAD_I32, { 2, 2 }, [](int64_t i) { return (float)i; })
         .idx({ 30, 17 })
         .idx({ 0, 12, T })
         .dense(RAD_F32, { 4 }, [](int64_t i) { return 0.1f + 0.05f * (float)i; })
         .dense(RAD_F32, { 4 }, [](int64_t i) { return 0.2f - 0.03f * (float)i; })
         .none().none()
         .dense(RAD_BF16, { T, kvh * gqa * hd }, zero, true)
         .pi("q_len", T).pi("head_dim", hd).pi("gqa", gqa).pi("block_size", BS).pi("causal", 0)
         .ps("q_dtype", "bf16").ps("kv_dtype", "fp8_e4m3").pf("scale", 0.125);
        hs.push_back(std::move(h));
    }

    /* ---- scatter_add_rows: a repeated destination adds in index order ----------------------- */
    {
        Hand h("a destination three times, one skip", "scatter_add_rows", 0.0);
        h.dense(RAD_BF16, { 5, 67 }, hfill(280))
         .idx({ 2, -1, 2, 0, 2 })
         .dense(RAD_BF16, { 4, 67 }, hfill(281), true)
         .pi("M", 5).pi("n", 67).ps("dtype", "bf16");
        hs.push_back(std::move(h));

        Hand r("an index past x's rows is refused", "scatter_add_rows", 0.0);
        r.dense(RAD_F32, { 2, 67 }, hfill(282))
         .idx({ 1, 7 })
         .dense(RAD_F32, { 4, 67 }, hfill(283), true)
         .pi("M", 2).pi("n", 67).ps("dtype", "f32")
         .refuses(RAD_E_INVAL);
        hs.push_back(std::move(r));
    }
}

/* ================================================================== the video generator */
/* Forms libref's descriptions do not draw: f32 positions as a [3, M] view of a wider plane, absent
 * gains and a partial rotary; the DiT's norm over three entries (one empty) with [shift | scale] a
 * column slice of a wider modulation table; a latent grid of different extents on every axis, so
 * the tiles overlap unevenly; a chunk between two others (the temporal blend and the overlap it
 * hands on), a last one and a first one; positions into a view of a wider plane. Everything but
 * the norms and the rotaries is held to zero error: libref's arithmetic bit for bit. */
void video_cases(std::vector<Hand>& hs) {
    const auto zero = [](int64_t) { return 0.0f; };
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const std::string sfx = std::string(", ") + dn;
        const double norm_tol = dt == RAD_F32 ? 1e-5 : 6e-3;

        /* ---- the rotaries: positions f32, component-major, rows 32 apart ------------------- */
        {
            const int64_t M = 9, hd = 128, nh = 2, nkv = 2, W = (nh + 2 * nkv) * hd;
            const auto pos = [](int64_t i) { return 0.37f * (float)(i % 32) - 2.5f; };
            Hand h(("the DiT's axial 16 16 16 over 96 of 128, f32 positions" + sfx).c_str(),
                   "qk_norm_rope", norm_tol);
            h.dense(dt, { M, W }, hfill(400), true)
             .view(RAD_F32, { 3, M }, { 32, 1 }, 96, pos)
             .dense(dt, { hd }, hfill(401, 0.1f))
             .dense(dt, { hd }, hfill(402, 0.1f))
             .pi("M", M).pi("head_dim", hd).pi("n_head", nh).pi("n_head_kv", nkv)
             .pf("theta", 10000.0).pf("eps", 1e-5).ps("mode", "axial").pi("rotary_dim", 96)
             .ps("sections", "16 16 16");
            hs.push_back(std::move(h));

            /* The decoder's: no gains, positions 2 pi (2 (c + 0.5) / n - 1), theta 100. */
            const int64_t hv = 64, Wv = 3 * 4 * hv;
            const auto vpos = [](int64_t i) {
                const int64_t c = i / 32, m = i % 32;
                const int64_t n = c == 0 ? 3 : 4, k = c == 0 ? m % 3 : c == 1 ? (m / 4) % 4 : m % 4;
                return 6.2831855f * (2.0f * (((float)k + 0.5f) / (float)n) - 1.0f);
            };
            Hand g(("the decoder's axial 8 8 8 over 48 of 64, no gains" + sfx).c_str(),
                   "qk_norm_rope", norm_tol);
            g.dense(dt, { M, Wv }, hfill(403), true)
             .view(RAD_F32, { 3, M }, { 32, 1 }, 96, vpos)
             .none().none()
             .pi("M", M).pi("head_dim", hv).pi("n_head", 4).pi("n_head_kv", 4)
             .pf("theta", 100.0).pf("eps", 1e-5).ps("mode", "axial").pi("rotary_dim", 48)
             .ps("sections", "8 8 8");
            hs.push_back(std::move(g));

            Hand r(("rope over fractional f32 positions, scale 2" + sfx).c_str(), "rope",
                   norm_tol);
            r.dense(dt, { M, 4 * 64 }, hfill(404), true)
             .dense(RAD_F32, { M }, [](int64_t i) { return 0.73f * (float)i - 1.1f; })
             .pi("M", M).pi("head_dim", 64).pi("n_head", 2).pi("n_head_kv", 1)
             .pf("theta", 10000.0).pf("scale", 2.0).ps("mode", "neox");
            hs.push_back(std::move(r));
        }

        /* ---- the DiT's norm: three entries (one empty), [shift | scale] at a 6n pitch ------- */
        {
            const int64_t R = 11, D = 160, W6 = 6 * D;
            for (const bool gated : { true, false }) {
                Hand h((std::string(gated ? "gated" : "ungated") + ", three entries, s at a 6n "
                        "pitch" + sfx).c_str(), "rms_mod", norm_tol);
                if (gated) {
                    h.dense(dt, { R, D }, hfill(410))
                     .view(dt, { 3, D }, { W6, 1 }, 3 * W6, hfill(411, 0.5f));
                } else {
                    h.none().none();
                }
                h.view(dt, { 3, 2 * D }, { W6, 1 }, 3 * W6, hfill(412, 0.5f))
                 .dense(dt, { D }, [](int64_t i) { return 0.8f + 0.003f * (float)i; })
                 .idx({ 0, 4, 4, R })
                 .dense(dt, { R, D }, hfill(413, 2.0f), true)
                 .dense(dt, { R, D }, zero, true)
                 .pi("M", R).pi("n", D).pf("eps", 1e-5).pi("gated", gated ? 1 : 0)
                 .ps("dtype", dn);
                hs.push_back(std::move(h));

                Hand q((std::string(gated ? "gated" : "ungated") + ", quantised, three entries" +
                        sfx).c_str(), "rms_mod_quant_fp8", 1e-2);
                if (gated) {
                    q.dense(dt, { R, D }, hfill(414))
                     .view(dt, { 3, D }, { W6, 1 }, 3 * W6, hfill(415, 0.5f));
                } else {
                    q.none().none();
                }
                q.view(dt, { 3, 2 * D }, { W6, 1 }, 3 * W6, hfill(416, 0.5f))
                 .dense(dt, { D }, [](int64_t i) { return 0.8f + 0.003f * (float)i; })
                 .idx({ 0, 4, 4, R })
                 .dense(dt, { R, D }, hfill(417, 2.0f), true)
                 .dense(RAD_F8E4M3, { R, D }, zero, true)
                 .dense(RAD_F32, { R, D / 32 }, zero, true)
                 .dense(RAD_BF16, { R, D }, zero, true)
                 .pi("M", R).pi("n", D).pf("eps", 1e-5).pi("group", 32)
                 .pi("gated", gated ? 1 : 0).ps("dtype", dn);
                hs.push_back(std::move(q));
            }
        }

        /* ---- the sampler: a bf16 velocity into an f32 state ------------------------------- */
        {
            Hand h(("bf16 velocity, f32 state, rows at a wider pitch" + sfx).c_str(),
                   "flow_x0_step", 0.0);
            h.dense(dt, { 7, 96 }, hfill(420, 2.0f))
             .dense(RAD_F32, { 1 }, [](int64_t) { return 0.3046875f; })
             .dense(RAD_F32, { 1 }, [](int64_t) { return 0.6953125f; })
             .dense(RAD_F32, { 1 }, [](int64_t) { return 0.4413f; })
             .view(RAD_F32, { 7, 96 }, { 101, 1 }, 7 * 101, hfill(421), true)
             .pi("M", 7).pi("n", 96).ps("dtype", dn);
            hs.push_back(std::move(h));
        }

        /* ---- the decoder's geometry: 3 frames of 6 x 9 latents, tiles of 4 over 1 ------------ *
         * Rows split 6 -> starts 0, 2 (overlap 2); columns 9 -> 0, 2, 5 (overlaps 2, 1). */
        {
            Hand h(("2 x 3 x 2 patches of 2 x 2 x 2, rows past the voxels" + sfx).c_str(),
                   "patch_unfold", 0.0);
            h.dense(RAD_F32, { 12, 5 * 8 }, hfill(430, 3.0f))
             .idx({ 2, 3, 2 })
             .dense(dt, { 100, 5 }, hfill(431), true)
             .pi("M", 100).pi("C", 5).pi("pt", 2).pi("ph", 2).pi("pw", 2)
             .ps("mul", "0.5 1.25 -2 0.3 1.7").ps("add", "0.1 -0.2 0.3 0 2.5").ps("dtype", dn);
            hs.push_back(std::move(h));

            /* Two register rows and two zero rows a tile. */
            const int64_t n = 37, period = 52, tiles = 6;
            Hand g(("6 x 9 in tiles of 4 over 1, registers and zero rows" + sfx).c_str(),
                   "vtile_gather", 0.0);
            g.dense(dt, { 3 * 6 * 9, n }, hfill(432))
             .dense(dt, { 2, n }, hfill(433))
             .idx({ 3, 6, 9 })
             .dense(dt, { tiles * period, n }, hfill(434), true)
             .pi("M", tiles * period).pi("n", n).pi("tile", 4).pi("overlap", 1).pi("frames", 3)
             .pi("period", period).pi("n_reg", 2).ps("dtype", dn);
            hs.push_back(std::move(g));

            /* The decoded tiles at scale 2, two clip frames a latent one: skip 1, keep 3, blend 1;
             * each tile's 48 token rows, then 4 rows the stitch does not read. */
            struct Chunk { int first, last; const char* what; };
            static const Chunk kChunks[] = { { 0, 0, "a chunk between two others" },
                                             { 0, 1, "a video's last chunk" },
                                             { 1, 0, "a video's first chunk" } };
            const int64_t npx = 12 * 18;
            for (const Chunk& ch : kChunks) {
                const int64_t fout = 3 + (ch.last ? 1 : 0);
                Hand s((std::string(ch.what) + ", 6 x 9 in tiles of 4 over 1" + sfx).c_str(),
                       "vtile_stitch", 0.0);
                s.dense(dt, { tiles * period, 3 * 2 * 2 * 2 }, hfill(435, 2.0f))
                 .idx({ 3, 6, 9, ch.first, ch.last })
                 .dense(RAD_BF16, { npx, 3 }, hfill(436, 2.0f), true)
                 .dense(RAD_U8, { fout * npx, 3 }, zero, true)
                 .pi("M", fout * npx).pi("tile", 4).pi("overlap", 1).pi("scale", 2).pi("pt", 2)
                 .pi("skip", 1).pi("keep", 3).pi("blend", 1).pi("period", period)
                 .ps("mean", "0.485 0.456 0.406").ps("std", "0.229 0.224 0.225");
                hs.push_back(std::move(s));
            }
        }
    }

    /* ---- positions into a [3, 40] view of a [3, 64] plane ---------------------------------- */
    {
        Hand h("two periods of 2 x 3 x 4 and its registers, rows 64 apart", "grid_pos3", 0.0);
        h.view(RAD_F32, { 3, 40 }, { 64, 1 }, 192, [](int64_t) { return -7.0f; }, true)
         .pi("M", 40).pi("T", 2).pi("H", 3).pi("W", 4).pi("period", 27)
         .pf("scale", 6.283185307179586);
        hs.push_back(std::move(h));
    }
}

/* ================================================================== the latent upscaler */
/* Forms libref's cube descriptions do not draw: volumes of a different extent on every axis, an
 * activation at a wider row pitch, the residual in place (res = y), a group norm in place (y = x)
 * without its modulation and with a modulation row of the other dtype, statistics over several
 * of avx's 64-row blocks and a part of one, kernels of other shapes, real resizes (MiniMax-H3's
 * x2 and 4.2 x 4 ratios, and shrinking), output rows past the volume, the soundtrack's and the
 * video's re-noise widths. tconv_dw, resize_bilinear, patch_fold and flow_renoise are held to
 * zero error, libref's arithmetic bit for bit; the convolution's GEMM and the norm's statistics
 * sum in another order and are held to the tolerance table. */
void upscaler_cases(std::vector<Hand>& hs) {
    const auto zero = [](int64_t) { return 0.0f; };
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const uint32_t other = dt == RAD_F32 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
        const std::string sfx = std::string(", ") + dn;
        const double tol = dt == RAD_F32 ? 1e-5 : 6e-3;

        /* ---- conv3d_ndhwc ------------------------------------------------------------------ */
        {
            const int64_t T = 2, H = 3, W = 4, R = T * H * W, ci = 7, co = 11;
            Hand h(("2 x 3 x 4, x at a pitch, the residual in place, f32 bias" + sfx).c_str(),
                   "conv3d_ndhwc", tol);
            h.view(dt, { R, ci }, { ci + 3, 1 }, R * (ci + 3), hfill(450))
             .dense(dt, { co, ci, 3, 3, 3 }, hfill(451, 0.3f))
             .dense(RAD_F32, { co }, hfill(452, 0.5f))
             .idx({ T, H, W })
             .dense(dt, { R, co }, hfill(453), true)
             .alias(4)
             .pi("M", R).pi("cin", ci).pi("cout", co).pi("k", 3).pi("kt", 3).ps("dtype", dn);
            hs.push_back(std::move(h));

            /* A temporal-only kernel over six frames of 2 x 3, two output rows past the volume. */
            const int64_t R2 = 6 * 2 * 3, ci2 = 9, co2 = 50;
            Hand g(("6 x 2 x 3 at kernel 5 x 1 x 1, rows past the volume" + sfx).c_str(),
                   "conv3d_ndhwc", tol);
            g.dense(dt, { R2, ci2 }, hfill(454))
             .dense(dt, { co2, ci2, 5, 1, 1 }, hfill(455, 0.3f))
             .dense(dt, { co2 }, hfill(456, 0.5f))
             .idx({ 6, 2, 3 })
             .none()
             .dense(dt, { R2 + 2, co2 }, hfill(457), true)
             .pi("M", R2 + 2).pi("cin", ci2).pi("cout", co2).pi("k", 1).pi("kt", 5)
             .ps("dtype", dn);
            hs.push_back(std::move(g));

            Hand e(("an even kernel is refused" + sfx).c_str(), "conv3d_ndhwc", tol);
            e.dense(dt, { R, ci }, hfill(458))
             .dense(dt, { co, ci, 3, 2, 2 }, hfill(459))
             .dense(dt, { co }, hfill(459))
             .idx({ T, H, W })
             .none()
             .dense(dt, { R, co }, zero, true)
             .pi("M", R).pi("cin", ci).pi("cout", co).pi("k", 2).pi("kt", 3).ps("dtype", dn)
             .refuses(RAD_E_SHAPE);
            hs.push_back(std::move(e));
        }

        /* ---- group_norm: stats f32 of exactly groups (2 + 3 1024) values --------------------- */
        {
            const int64_t R = 200, n = 48, G = 6;
            const auto offset = [](int64_t i) { return 2.5f + 3.0f * hfill(460)(i); };
            Hand h(("200 rows in 6 groups, y = x at a pitch, unmodulated" + sfx).c_str(),
                   "group_norm", tol);
            h.view(dt, { R, n }, { n + 5, 1 }, R * (n + 5), offset)
             .dense(dt, { n }, hfill(461))
             .dense(dt, { n }, hfill(462))
             .none()
             .dense(RAD_F32, { G * (2 + 3 * 1024) }, zero, true)
             .alias(0)
             .pi("M", R).pi("n", n).pi("groups", G).pf("eps", 1e-5).ps("act", "none")
             .ps("dtype", dn);
            hs.push_back(std::move(h));

            const int64_t R2 = 150, n2 = 96, G2 = 32;
            Hand g(("150 rows in 32 groups of 3, a modulation row of the other dtype, silu" +
                    sfx).c_str(), "group_norm", tol);
            g.dense(dt, { R2, n2 }, hfill(463, 2.0f))
             .dense(dt, { n2 }, hfill(464))
             .dense(dt, { n2 }, hfill(465))
             .dense(other, { 2 * n2 }, hfill(466, 0.5f))
             .dense(RAD_F32, { G2 * (2 + 3 * 1024) }, zero, true)
             .dense(dt, { R2, n2 }, zero, true)
             .pi("M", R2).pi("n", n2).pi("groups", G2).pf("eps", 1e-5).ps("act", "silu")
             .ps("dtype", dn);
            hs.push_back(std::move(g));
        }

        /* ---- tconv_dw: seven frames of 2 x 3, x at a pitch, rows past the volume ------------ */
        {
            const int64_t R = 7 * 2 * 3, n = 37;
            Hand h(("7 x 2 x 3 at k 5, x at a pitch, rows past the volume" + sfx).c_str(),
                   "tconv_dw", 0.0);
            h.view(dt, { R, n }, { n + 3, 1 }, R * (n + 3), hfill(470))
             .dense(dt, { n, 1, 5, 1, 1 }, hfill(471, 0.5f))
             .dense(RAD_F32, { n }, hfill(472, 0.5f))
             .idx({ 7, 2, 3 })
             .dense(dt, { R + 2, n }, hfill(473), true)
             .pi("M", R + 2).pi("n", n).pi("k", 5).ps("dtype", dn);
            hs.push_back(std::move(h));
        }

        /* ---- resize_bilinear: up and down at once, H3's ratios, shrinking ------------------- */
        {
            struct Rs { int64_t T, h, w, H, W, n; const char* what; };
            static const Rs kRs[] = { { 2, 3, 5, 7, 4, 37, "3 x 5 to 7 x 4" },
                                      { 2, 5, 3, 10, 6, 16, "5 x 3 to 10 x 6 (x2)" },
                                      { 1, 5, 3, 21, 12, 16, "5 x 3 to 21 x 12 (4.2 x 4)" },
                                      { 3, 6, 7, 4, 2, 20, "6 x 7 to 4 x 2" } };
            for (const Rs& v : kRs) {
                const int64_t rin = v.T * v.h * v.w, rout = v.T * v.H * v.W;
                Hand h((std::string(v.what) + ", x at a pitch, rows past" + sfx).c_str(),
                       "resize_bilinear", 0.0);
                h.view(dt, { rin, v.n }, { v.n + 2, 1 }, rin * (v.n + 2), hfill(480 + v.h))
                 .idx({ v.T, v.h, v.w, v.H, v.W })
                 .dense(dt, { rout + 3, v.n }, hfill(481), true)
                 .pi("M", rout + 3).pi("n", v.n).ps("dtype", dn);
                hs.push_back(std::move(h));
            }
        }

        /* ---- patch_fold: 2 x 3 x 2 patches of 2 x 2 x 2, x at a pitch, rows past ------------ */
        {
            Hand h(("2 x 3 x 2 patches of 2 x 2 x 2, x at a pitch, rows past" + sfx).c_str(),
                   "patch_fold", 0.0);
            h.view(dt, { 96, 5 }, { 8, 1 }, 96 * 8, hfill(490, 3.0f))
             .idx({ 2, 3, 2 })
             .dense(RAD_F32, { 14, 40 }, hfill(491), true)
             .pi("M", 14).pi("C", 5).pi("pt", 2).pi("ph", 2).pi("pw", 2)
             .ps("mul", "0.5 1.25 -2 0.3 1.7").ps("add", "0.1 -0.2 0.3 0 2.5").ps("dtype", dn);
            hs.push_back(std::move(h));
        }
    }

    /* ---- flow_renoise: the video's 96 and the soundtrack's 32, rows at a wider pitch --------- */
    for (const int64_t n : { 96, 32 }) {
        Hand h(("n " + std::to_string(n) + ", rows at a wider pitch").c_str(), "flow_renoise", 0.0);
        h.dense(RAD_F32, { 1 }, [](int64_t) { return 0.37f; })
         .idx({ 12345, 81 })
         .view(RAD_F32, { 7, n }, { n + 5, 1 }, 7 * (n + 5), hfill(495, 2.0f), true)
         .pi("M", 7).pi("n", n).pi("stream", 1).ps("dtype", "f32");
        hs.push_back(std::move(h));
    }
}

/* ================================================================== video sparse attention
 * What libref's descriptions cannot draw: uneven dense segments (13 and 14 rows: a tile each in
 * boxes of 4, two each in boxes of 2) before a 6 x 5 x 7 grid (a 10 x 14 latent at patch 2) with
 * edge boxes on every axis, and `geom`'s other entries holding values no tile reads; q, k, v and
 * the gate at a fused projection's row pitch and the output at a pitch of its own; `sel` and
 * `coarse` wider than the tiles and the selection; attn_vsa over selections of every size from 1
 * to V that are not a top-k of anything; and what both refuse. Outputs start as a sentinel, so an
 * element one side writes and the other leaves shows as a difference, and attn_vsa's `sel` holds
 * one wherever the op must not read. */

/* attn_vsa's `sel`, [nh, cap, kc]: per head and video query tile, k distinct video tiles of
 * [P, N) in ascending order, drawn from `seed`; `fill` everywhere else. */
std::vector<float> vsa_selection(int64_t nh, int64_t cap, int64_t kc, int64_t P, int64_t N,
                                 int64_t k, uint64_t seed, float fill) {
    std::vector<float> s((size_t)(nh * cap * kc), fill);
    std::vector<int64_t> pool;
    uint64_t x = seed;
    for (int64_t h = 0; h < nh; ++h)
        for (int64_t i = P; i < N; ++i) {
            pool.clear();
            for (int64_t j = P; j < N; ++j) pool.push_back(j);
            for (int64_t a = 0; a < k; ++a) {
                const int64_t b = a + (int64_t)(hsplitmix(x) % (uint64_t)(N - P - a));
                std::swap(pool[(size_t)a], pool[(size_t)b]);
            }
            std::sort(pool.begin(), pool.begin() + k);
            for (int64_t a = 0; a < k; ++a)
                s[(size_t)((h * cap + i) * kc + a)] = (float)pool[(size_t)a];
        }
    return s;
}

void vsa_cases(std::vector<Hand>& hs) {
    const int64_t R = 237, NH = 2, HD = 64, W = 4 * NH * HD, WO = NH * HD + 24;
    const auto sentinel = [](int64_t) { return -7.0f; };
    const auto geom = [](int64_t i) {
        return i == RAD_VGEOM_LAT_FRAMES ? 6.0f : i == RAD_VGEOM_LAT_H ? 10.0f
             : i == RAD_VGEOM_LAT_W ? 14.0f : (float)(1000 + i);
    };
    struct Geo { int64_t tt, P, V; const char* what; };
    static const Geo kGeo[] = { { 4, 2, 8, "boxes of 4" }, { 2, 4, 36, "boxes of 2" } };
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const double sel_tol = dt == RAD_F32 ? 1e-5 : 6e-3, att_tol = dt == RAD_F32 ? 1e-4 : 2e-2;
        for (const Geo& g : kGeo)
            for (const double sp : { 0.8, 0.0, 0.95 }) {
                const int64_t N = g.P + g.V;
                const int64_t k = std::max<int64_t>(
                    1, std::min<int64_t>((int64_t)std::ceil((1.0 - sp) * (double)g.V), g.V));
                char sfx[96];
                std::snprintf(sfx, sizeof sfx, ", 13 | 14 then 6 x 5 x 7, %s, sparsity %g, %s",
                              g.what, sp, dn);
                for (const bool coarse : { true, false }) {
                    Hand h((std::string(coarse ? "q, k, v at a pitch, coarse" : "q, k at a pitch, "
                                        "no coarse") + sfx).c_str(), "vsa_select", sel_tol);
                    h.view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(500))
                     .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(501));
                    if (coarse) h.view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(502, 2.0f));
                    else        h.none();
                    h.idx({ 0, 13, 27, R })
                     .dense(RAD_I32, { RAD_VGEOM_N }, geom)
                     .dense(RAD_I32, { NH, N + 2, k + 2 }, sentinel, true);
                    if (coarse) h.dense(dt, { NH, N + 1, HD }, sentinel, true);
                    else        h.none();
                    h.pi("M", R).pi("head_dim", HD).pi("tile_t", g.tt).pi("tile_h", g.tt)
                     .pi("tile_w", g.tt).pi("patch", 2).pf("sparsity", sp);
                    hs.push_back(std::move(h));

                    const std::vector<float> sv =
                        vsa_selection(NH, N + 1, k + 1, g.P, N, k, 503 + (uint64_t)k, -9.0f);
                    Hand a((std::string(coarse ? "q, k, v, gate at a pitch, coarse, out at its "
                                        "own" : "q, k, v at a pitch, out at its own") + sfx +
                            ", a drawn selection").c_str(), "attn_vsa", att_tol);
                    a.view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(504))
                     .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(505))
                     .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(506, 2.0f))
                     .idx({ 0, 13, 27, R })
                     .dense(RAD_I32, { RAD_VGEOM_N }, geom)
                     .dense(RAD_I32, { NH, N + 1, k + 1 },
                            [sv](int64_t i) { return sv[(size_t)i]; });
                    if (coarse) {
                        a.dense(dt, { NH, N + 1, HD }, hfill(507, 0.5f))
                         .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(508));
                    } else {
                        a.none().none();
                    }
                    a.view(dt, { R, NH * HD }, { WO, 1 }, R * WO, sentinel, true)
                     .pi("M", R).pi("head_dim", HD).pi("tile_t", g.tt).pi("tile_h", g.tt)
                     .pi("tile_w", g.tt).pi("patch", 2).pf("sparsity", sp);
                    hs.push_back(std::move(a));
                    if (!coarse) continue;
                    /* The int8 score form: k's codes and per-(row, head) scales beside it. */
                    Hand q8((std::string("int8 scores, coarse") + sfx + ", a drawn selection").c_str(),
                            "attn_vsa", att_tol);
                    q8.view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(504))
                      .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(505))
                      .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(506, 2.0f))
                      .idx({ 0, 13, 27, R })
                      .dense(RAD_I32, { RAD_VGEOM_N }, geom)
                      .dense(RAD_I32, { NH, N + 1, k + 1 },
                             [sv](int64_t i) { return sv[(size_t)i]; })
                      .dense(dt, { NH, N + 1, HD }, hfill(507, 0.5f))
                      .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(508))
                      .view(dt, { R, NH * HD }, { WO, 1 }, R * WO, sentinel, true)
                      .dense(RAD_I8, { R, NH * HD }, [](int64_t i) {
                          return (float)((int)((i * 53) % 255) - 127);
                      })
                      .dense(RAD_F32, { R, NH }, [](int64_t i) { return 0.004f + 0.0005f * (float)(i % 9); })
                      .pi("M", R).pi("head_dim", HD).pi("tile_t", g.tt).pi("tile_h", g.tt)
                      .pi("tile_w", g.tt).pi("patch", 2).pf("sparsity", sp).ps("qk", "i8");
                    hs.push_back(std::move(q8));
                    /* The same with the fp8 P.V form: E4M3 values and a scale per column. */
                    Hand p8((std::string("int8 scores, fp8 P.V, coarse") + sfx +
                             ", a drawn selection").c_str(), "attn_vsa", att_tol);
                    p8.view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(504))
                      .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(505))
                      .dense(RAD_F8E4M3, { R, NH * HD }, hfill(506, 64.0f))
                      .idx({ 0, 13, 27, R })
                      .dense(RAD_I32, { RAD_VGEOM_N }, geom)
                      .dense(RAD_I32, { NH, N + 1, k + 1 },
                             [sv](int64_t i) { return sv[(size_t)i]; })
                      .dense(dt, { NH, N + 1, HD }, hfill(507, 0.5f))
                      .view(dt, { R, NH * HD }, { W, 1 }, R * W, hfill(508))
                      .view(dt, { R, NH * HD }, { WO, 1 }, R * WO, sentinel, true)
                      .dense(RAD_I8, { R, NH * HD }, [](int64_t i) {
                          return (float)((int)((i * 53) % 255) - 127);
                      })
                      .dense(RAD_F32, { R, NH }, [](int64_t i) {
                          return 0.004f + 0.0005f * (float)(i % 9);
                      })
                      .dense(RAD_F32, { NH * HD }, [](int64_t i) {
                          return 0.01f + 0.001f * (float)(i % 13);
                      })
                      .pi("M", R).pi("head_dim", HD).pi("tile_t", g.tt).pi("tile_h", g.tt)
                      .pi("tile_w", g.tt).pi("patch", 2).pf("sparsity", sp).ps("qk", "i8")
                      .ps("pv", "fp8");
                    hs.push_back(std::move(p8));
                }
            }
    }

    /* ---- refusals: nothing written --------------------------------------------------------- */
    const int64_t N = 10, k = 2;
    std::vector<float> sv = vsa_selection(NH, N, k, 2, N, k, 509, -9.0f);
    std::swap(sv[(size_t)(5 * k)], sv[(size_t)(5 * k + 1)]);     /* head 0, tile 5: descending */
    struct Bad { const char* what; int rc; bool gate; float frames; };
    static const Bad kBad[] = {
        { "a selection row out of order", RAD_E_INVAL, true, 6.0f },
        { "coarse without its gate", RAD_E_SHAPE, false, 6.0f },
        { "a grid of 5 frames short of the last segment", RAD_E_SHAPE, true, 5.0f },
    };
    for (const Bad& b : kBad) {
        const float fr = b.frames;
        const std::vector<float> sel = b.rc == RAD_E_INVAL
                                     ? sv : vsa_selection(NH, N, k, 2, N, k, 509, -9.0f);
        Hand a((std::string("refused: ") + b.what).c_str(), "attn_vsa", 2e-2);
        a.dense(RAD_BF16, { R, NH * HD }, hfill(510))
         .dense(RAD_BF16, { R, NH * HD }, hfill(511))
         .dense(RAD_BF16, { R, NH * HD }, hfill(512))
         .idx({ 0, 13, 27, R })
         .dense(RAD_I32, { RAD_VGEOM_N },
                [fr, geom](int64_t i) { return i == RAD_VGEOM_LAT_FRAMES ? fr : geom(i); })
         .dense(RAD_I32, { NH, N, k }, [sel](int64_t i) { return sel[(size_t)i]; })
         .dense(RAD_BF16, { NH, N, HD }, hfill(513));
        if (b.gate) a.dense(RAD_BF16, { R, NH * HD }, hfill(514));
        else        a.none();
        a.dense(RAD_BF16, { R, NH * HD }, sentinel, true)
         .pi("M", R).pi("head_dim", HD).pi("tile_t", 4).pi("tile_h", 4).pi("tile_w", 4)
         .pi("patch", 2).pf("sparsity", 0.8).refuses(b.rc);
        hs.push_back(std::move(a));
    }
    {
        Hand s("refused: a grid of 5 frames short of the last segment", "vsa_select", 6e-3);
        s.dense(RAD_BF16, { R, NH * HD }, hfill(515))
         .dense(RAD_BF16, { R, NH * HD }, hfill(516))
         .none()
         .idx({ 0, 13, 27, R })
         .dense(RAD_I32, { RAD_VGEOM_N },
                [geom](int64_t i) { return i == RAD_VGEOM_LAT_FRAMES ? 5.0f : geom(i); })
         .dense(RAD_I32, { NH, N, k }, sentinel, true)
         .none()
         .pi("M", R).pi("head_dim", HD).pi("tile_t", 4).pi("tile_h", 4).pi("tile_w", 4)
         .pi("patch", 2).pf("sparsity", 0.8).refuses(RAD_E_SHAPE);
        hs.push_back(std::move(s));
    }
}

/* ================================================================== the QSA indexer and DFlash2
 * What libref's descriptions cannot draw: cu not starting at 0 and rows no sequence holds, -1
 * pages, the score plane as a column slice of the declared width, the block keys in the KV
 * cache's rank, ring wrap in the tail and the work list, a slot-less sequence, M-RoPE positions,
 * and the delta and base views the drafter issues. Outputs start as a sentinel, so an element one
 * side writes and the other leaves shows as a difference. */
void qsa_cases(std::vector<Hand>& hs) {
    const auto sent = [](int64_t) { return 7.0f; };

    /* ---- qsa_score ----------------------------------------------------------------------- */
    {
        const int64_t n = 128, H = 4, S = 3, M = 12, P = 200, W = 64, live = 50, pitch = 80;
        Hand h("cu from 2, unheld rows, -1 pages, a dead sequence, a padded plane, rank-5 keys",
               "qsa_score", 6e-3);
        h.dense(RAD_BF16, { M, H * n }, hfill(201))
         .dense(RAD_BF16, { P, 2, 4, 1, 16 }, hfill(202))
         .dense(RAD_I32, { S, W }, [=](int64_t i) {
             return (i / W == 2 || i % 7 == 3) ? -1.0f : (float)((i * 37 + 11) % P);
         })
         .idx({ 2, 5, 6, 10 })
         .idx({ 37, 50, 17 })
         .view(RAD_F32, { M, live }, { pitch, 1 }, M * pitch, sent, true)
         .pi("M", M).pi("n", n).pi("heads", H).pi("blocks", W).pi("seqs", S).ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }
    {
        const int64_t n = 40, H = 2, M = 19, P = 70;
        Hand h("19 rows of one sequence (three tiles), n 40, q a column slice", "qsa_score", 1e-5);
        h.view(RAD_F32, { M, H * n }, { (H + 1) * n, 1 }, M * (H + 1) * n, hfill(203))
         .dense(RAD_F32, { P, n }, hfill(204))
         .dense(RAD_I32, { 1, P }, [=](int64_t i) { return (float)((i * 23 + 5) % P); })
         .idx({ 0, M })
         .idx({ P })
         .dense(RAD_F32, { M, P }, sent, true)
         .pi("M", M).pi("n", n).pi("heads", H).pi("blocks", P).pi("seqs", 1).ps("dtype", "f32");
        hs.push_back(std::move(h));
    }
    {
        Hand h("a page id past the key store is refused before any write", "qsa_score", 0.0);
        h.dense(RAD_F32, { 2, 32 }, hfill(205))
         .dense(RAD_F32, { 8, 16 }, hfill(206))
         .dense(RAD_I32, { 1, 4 }, [](int64_t i) { return i == 3 ? 8.0f : (float)i; })
         .idx({ 0, 2 })
         .idx({ 4 })
         .dense(RAD_F32, { 2, 4 }, sent, true)
         .pi("M", 2).pi("n", 16).pi("heads", 2).pi("blocks", 4).pi("seqs", 1).ps("dtype", "f32")
         .refuses(RAD_E_INVAL);
        hs.push_back(std::move(h));
    }

    /* ---- qsa_select ---------------------------------------------------------------------- */
    {
        /* libr4d's eight rows (r4d_selftest.cpp: qsa_select_few_live_case), which pinned its
         * selection to libref's rule: 100, 400, 200 and 512 live keys under topk 512; 2000 live
         * of which most score zero; every block dead; 300 blocks with -1 pages; 8000 live. Each
         * row is mid-block, so its partial page is considered. The expected table is the rule
         * itself, computed here, so both libraries are held to it and not only to each other. */
        const int64_t S = 8, M = S, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
        const int64_t nrow[S] = { 8000, 8000, 8000, 8000, 2000, 8000, 300, 8000 };
        std::vector<float> sc((size_t)(M * nb), -INFINITY);
        std::vector<int32_t> bt((size_t)(S * nb), -1), nc((size_t)S), pos((size_t)M),
                             cuh((size_t)(S + 1));
        uint64_t x = 0x9E3779B97F4A7C15ull;
        auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
        for (int64_t s = 0; s < S; ++s) {
            const int64_t n = nrow[s];
            nc[(size_t)s] = (int32_t)n;
            cuh[(size_t)s] = (int32_t)s;
            pos[(size_t)s] = (int32_t)(n * ratio + (s % 3));
            for (int64_t b = 0; b < n; ++b) {
                float*   v  = &sc[(size_t)(s * nb + b)];
                int32_t* pg = &bt[(size_t)(s * nb + b)];
                const uint64_t u = rnd();
                const float cont = (float)(u % 1000003) / 1000.0f + 0.001f;
                *pg = (int32_t)(u % pages);
                switch (s) {
                    case 0: if (b % 80 != 3) *pg = -1; *v = cont; break;
                    case 1: if (b % 20 != 7) *pg = -1;
                            *v = (b / 20) % 4 == 0 ? cont
                               : (b / 20) % 3 == 0 ? -2.5f : (b / 20) % 3 == 1 ? -0.0f : 0.0f;
                            break;
                    case 2: *v = (b % 40 == 1) ? cont : -INFINITY; break;
                    case 3: if (b % 15 != 2 || b >= 15 * 512) *pg = -1;
                            *v = (b % 45 == 2) ? 0.0f : cont; break;
                    case 4: *v = (b % 7 == 0) ? cont : (b % 2 ? 0.0f : -0.0f); break;
                    case 5: *pg = -1; *v = cont; break;
                    case 6: if (b % 9 == 4) *pg = -1; *v = (float)(u % 13) * 0.5f; break;
                    default: *v = cont; break;
                }
            }
            bt[(size_t)(s * nb + n)] = (s % 2 == 0) ? (int32_t)(rnd() % pages) : -1;
        }
        cuh[(size_t)S] = (int32_t)M;
        std::vector<int64_t> want((size_t)(M * (topk + 1)), -1);
        for (int64_t m = 0; m < M; ++m) {
            const int64_t n = nrow[m];
            std::vector<int64_t> live;
            std::vector<float> val((size_t)n, 0.0f);
            for (int64_t b = 0; b < n; ++b) {
                const float v = sc[(size_t)(m * nb + b)];
                if (bt[(size_t)(m * nb + b)] < 0 || !(v > -1e30f)) continue;
                val[(size_t)b] = v > 0.0f ? v : 0.0f;
                live.push_back(b);
            }
            std::vector<int64_t> take;
            if ((int64_t)live.size() <= topk) {
                take = live;
            } else {
                std::vector<float> ord;
                for (int64_t b : live) ord.push_back(val[(size_t)b]);
                std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(),
                                 [](float l, float r) { return l > r; });
                const float thr = ord[(size_t)(topk - 1)];
                int64_t above = 0;
                for (int64_t b : live) above += val[(size_t)b] > thr;
                /* One ascending pass: a tie can sit below a block above the threshold, and the
                 * selection is emitted in block order (docs/QSA.md). */
                int64_t ties = 0;
                for (int64_t b : live) {
                    if (val[(size_t)b] > thr) take.push_back(b);
                    else if (val[(size_t)b] == thr && ties < topk - above) { take.push_back(b); ++ties; }
                }
            }
            int64_t* row = &want[(size_t)(m * (topk + 1))];
            int64_t out = 0;
            for (int64_t b : take) row[out++] = bt[(size_t)(m * nb + b)];
            const int32_t part = (pos[(size_t)m] + 1) % ratio != 0 ? bt[(size_t)(m * nb + n)] : -1;
            if (part >= 0) row[out++] = part;
        }
        Hand h("libr4d's eight few-live rows, against the rule", "qsa_select", 0.0);
        h.raw(RAD_F32, { M, nb }, sc.data())
         .raw(RAD_I32, { S, nb }, bt.data())
         .raw(RAD_I32, { S }, nc.data())
         .raw(RAD_I32, { M }, pos.data())
         .raw(RAD_I32, { S + 1 }, cuh.data())
         .dense(RAD_I32, { M, topk + 1 }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .pi("M", M).pi("blocks", nb).pi("topk", topk).pi("ratio", ratio).pi("seqs", S)
         .expect(5, std::move(want));
        hs.push_back(std::move(h));
    }
    {
        /* Rows 0 and 5 are in no span and read sequence 0. Scores repeat 1.0 (ties at the
         * threshold), and carry NaN and -inf (dead), +inf and -0 (live). */
        const int64_t S = 2, M = 6, live = 40, pitch = 56, W = 48, topk = 8, ratio = 4;
        Hand h("cu from 1, unheld rows, a column slice, ties, NaN, inf and -0", "qsa_select", 0.0);
        h.view(RAD_F32, { M, live }, { pitch, 1 }, M * pitch, [=](int64_t i) {
             switch ((i % pitch + i / pitch) % 9) {
                 case 0: case 1: case 7: return 1.0f;
                 case 2: return NAN;
                 case 3: return INFINITY;
                 case 4: return -0.0f;
                 case 5: return -INFINITY;
                 case 6: return -1.5f;
                 default: return 0.25f * (float)(i % 5);
             }
         })
         .dense(RAD_I32, { S, W }, [](int64_t i) {
             return i % 11 == 4 ? -1.0f : (float)((i * 13 + 3) % 500);
         })
         .idx({ 40, 25 })
         .idx({ 160, 98, 99, 161, 40, 7 })
         .idx({ 1, 3, 5 })
         .dense(RAD_I32, { M, topk + 1 }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .pi("M", M).pi("blocks", W).pi("topk", topk).pi("ratio", ratio).pi("seqs", S);
        hs.push_back(std::move(h));
    }
    {
        /* Row 0: three keys above a threshold 1997 blocks share. Row 1: the same history to a
         * mid-block position, so the partial page follows. Row 2: every odd block dead by score
         * and the rest scoring zero, so the threshold is the smallest live key, in the bin dead
         * keys share. */
        const int64_t nb = 2000, M = 3, topk = 100, ratio = 4;
        Hand h("2000 blocks: 1997-way ties, a partial page, live zeros beside dead keys",
               "qsa_select", 0.0);
        h.dense(RAD_F32, { M, nb }, [=](int64_t i) {
             const int64_t m = i / nb, b = i % nb;
             if (m == 2) return b % 2 ? -INFINITY : -1.0f;
             return (b == 17 || b == 900 || b == 1999) ? 3.0f + (float)b : 0.5f;
         })
         .dense(RAD_I32, { 1, nb }, [](int64_t i) { return (float)(3 * i + 1); })
         .idx({ nb })
         .idx({ 4 * nb + 3, 4001, 4 * nb + 3 })
         .idx({ 0, M })
         .dense(RAD_I32, { M, topk + 1 }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .pi("M", M).pi("blocks", nb).pi("topk", topk).pi("ratio", ratio).pi("seqs", 1);
        hs.push_back(std::move(h));
    }
    {
        /* The signed order, against the rule: row 0 ranks 2 over the tie of -0 and +0 (NaN dead,
         * -1e31 live), row 1 is all negative with -inf dead and -1e30 live. Under the unsigned
         * order both rows would rank as zeros. */
        static const float v[2][8] = {
            { -3.0f, -1.0f, -0.0f, 2.0f, -1.0f, 0.0f, NAN, -1e31f },
            { -5.0f, -2e30f, -7.0f, -6.0f, -INFINITY, -9.0f, -1e30f, -8.0f } };
        Hand h("signed: negative scores, -0 tying +0, NaN and -inf dead", "qsa_select", 0.0);
        h.dense(RAD_F32, { 2, 8 }, [](int64_t i) { return v[i / 8][i % 8]; })
         .dense(RAD_I32, { 2, 8 }, [](int64_t i) { return (float)(100 + 10 * (i / 8) + i % 8); })
         .idx({ 8, 8 })
         .idx({ 31, 31 })
         .idx({ 0, 1, 2 })
         .dense(RAD_I32, { 2, 4 }, sent, true)
         .dense(RAD_I32, { 2 }, sent, true)
         .dense(RAD_I32, { 2 }, sent, true)
         .pi("M", 2).pi("blocks", 8).pi("topk", 3).pi("ratio", 4).pi("seqs", 2).pi("signed", 1)
         .expect(5, { 102, 103, 105, -1, 110, 112, 113, -1 });   /* block order, not rank */
        hs.push_back(std::move(h));
    }
    {
        /* The column-slice case's scores, shifted to straddle zero, under the signed order. */
        const int64_t S = 2, M = 6, live = 40, pitch = 56, W = 48, topk = 8, ratio = 4;
        Hand h("signed: cu from 1, unheld rows, a column slice, ties across zero", "qsa_select",
               0.0);
        h.view(RAD_F32, { M, live }, { pitch, 1 }, M * pitch, [=](int64_t i) {
             switch ((i % pitch + i / pitch) % 9) {
                 case 0: case 1: case 7: return -1.0f;
                 case 2: return NAN;
                 case 3: return INFINITY;
                 case 4: return -0.0f;
                 case 5: return -INFINITY;
                 case 6: return -1.5f;
                 default: return 0.25f * (float)(i % 5) - 0.5f;
             }
         })
         .dense(RAD_I32, { S, W }, [](int64_t i) {
             return i % 11 == 4 ? -1.0f : (float)((i * 13 + 3) % 500);
         })
         .idx({ 40, 25 })
         .idx({ 160, 98, 99, 161, 40, 7 })
         .idx({ 1, 3, 5 })
         .dense(RAD_I32, { M, topk + 1 }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .dense(RAD_I32, { M }, sent, true)
         .pi("M", M).pi("blocks", W).pi("topk", topk).pi("ratio", ratio).pi("seqs", S)
         .pi("signed", 1);
        hs.push_back(std::move(h));
    }
    {
        Hand h("a `signed` other than 0 or 1 is refused before any write", "qsa_select", 0.0);
        h.dense(RAD_F32, { 1, 4 }, hfill(208))
         .dense(RAD_I32, { 1, 4 }, [](int64_t i) { return (float)i; })
         .idx({ 4 })
         .idx({ 15 })
         .idx({ 0, 1 })
         .dense(RAD_I32, { 1, 3 }, sent, true)
         .dense(RAD_I32, { 1 }, sent, true)
         .dense(RAD_I32, { 1 }, sent, true)
         .pi("M", 1).pi("blocks", 4).pi("topk", 2).pi("ratio", 4).pi("seqs", 1).pi("signed", 2)
         .refuses(RAD_E_INVAL);
        hs.push_back(std::move(h));
    }
    {
        Hand h("a negative block count is refused before any write", "qsa_select", 0.0);
        h.dense(RAD_F32, { 1, 4 }, hfill(207))
         .dense(RAD_I32, { 1, 4 }, [](int64_t i) { return (float)i; })
         .idx({ -1 })
         .idx({ 9 })
         .idx({ 0, 1 })
         .dense(RAD_I32, { 1, 3 }, sent, true)
         .dense(RAD_I32, { 1 }, sent, true)
         .dense(RAD_I32, { 1 }, sent, true)
         .pi("M", 1).pi("blocks", 4).pi("topk", 2).pi("ratio", 4).pi("seqs", 1)
         .refuses(RAD_E_INVAL);
        hs.push_back(std::move(h));
    }

    /* ---- qsa_tail_store ------------------------------------------------------------------ */
    {
        /* Sequence 0 is a 12-token chunk at positions 30..40 and 42: only its last 7 rows write,
         * so the ring wraps, the last row overwrites ring row 0, and the row before the seven
         * (position 34, ring row 6) must not write. Sequence 1 has no slot. Rows 0..2 and 19 are
         * in no span. */
        const int64_t n = 32, ratio = 4, ring = 7, S = 3, T = 20, slots = 4;
        std::vector<int64_t> pos((size_t)T);
        for (int64_t t = 0; t < T; ++t)
            pos[(size_t)t] = t < 3 ? 5 : t < 14 ? 30 + (t - 3) : t == 14 ? 42 : t == 15 ? 7
                           : t < 19 ? 100 + (t - 16) : 5;
        Hand h("a chunk past the ring, cu from 3, a slot-less sequence, sidx at pitch 2, k a "
               "column slice", "qsa_tail_store", 0.0);
        h.view(RAD_BF16, { T, n }, { 5 * n, 1 }, T * 5 * n, hfill(210))
         .idx(pos)
         .idx({ 3, 15, 16, 19 })
         .dense(RAD_BF16, { slots, 1, ring, n }, hfill(211), true)
         .dense(RAD_I32, { S, 2 }, [](int64_t i) {
             static const float v[6] = { 2, 3, -1, 0, 0, 1 };
             return v[i];
         })
         .pi("M", T).pi("n", n).pi("ratio", ratio).pi("ring", ring).pi("seqs", S)
         .ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }

    /* ---- qsa_work ------------------------------------------------------------------------ */
    /* Sequence 0 (rows 2..7, positions 6..11) completes blocks 1 and 2, block 1 straddling the
     * chunk start (positions 4 and 5 from the tail); sequence 1 (row 8, position 3) completes block
     * 0 but has no slot; sequence 2 (rows 9..12, positions 13..16) completes block 3, position 12
     * from ring row 5. Four items for eight slots. */
    for (const bool mrope : { false, true }) {
        const int64_t n = 32, ratio = 4, ring = 7, S = 3, T = 14, slots = 3, mw = 8;
        std::vector<int64_t> pos((size_t)T, 0);
        for (int64_t t = 2; t < 8; ++t) pos[(size_t)t] = 4 + t;
        pos[8] = 3;
        for (int64_t t = 9; t < 13; ++t) pos[(size_t)t] = 4 + t;
        Hand h(mrope ? "straddling blocks, a slot-less sequence, items past the count, M-RoPE"
                     : "straddling blocks, a slot-less sequence, items past the count",
               "qsa_work", 0.0);
        h.view(RAD_BF16, { T, n }, { 5 * n, 1 }, T * 5 * n, hfill(212))
         .idx(pos)
         .idx({ 2, 8, 9, 13 })
         .dense(RAD_BF16, { slots, 1, ring, n }, hfill(213))
         .dense(RAD_I32, { S, 2 }, [](int64_t i) {
             static const float v[6] = { 1, 2, -1, 0, 0, 2 };
             return v[i];
         })
         .dense(RAD_I32, { S, 6 }, [](int64_t i) { return (float)(100 + 3 * i); });
        if (mrope)
            h.dense(RAD_I32, { 3, T }, [=](int64_t i) {
                return (float)(1000 * (i / T) + 2 * (i % T) + 7);
            });
        else
            h.none();
        h.dense(RAD_BF16, { mw * ratio, n }, sent, true)
         .dense(RAD_I32, { mw }, sent, true);
        if (mrope) h.dense(RAD_I32, { mw, 3 }, sent, true);
        else       h.dense(RAD_I32, { mw }, sent, true);
        h.dense(RAD_I32, { S }, sent, true)
         .pi("M", T).pi("n", n).pi("ratio", ratio).pi("ring", ring).pi("seqs", S).pi("work", mw)
         .ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }
    {
        /* Three blocks complete and `work` admits two: the list stops at the bound and nc still
         * counts all three. */
        const int64_t n = 16, ratio = 4, ring = 4, T = 12, mw = 2;
        std::vector<int64_t> pos((size_t)T);
        for (int64_t t = 0; t < T; ++t) pos[(size_t)t] = t;
        Hand h("more blocks complete than `work` admits", "qsa_work", 0.0);
        h.dense(RAD_F32, { T, n }, hfill(214))
         .idx(pos)
         .idx({ 0, T })
         .dense(RAD_F32, { 1, 1, ring, n }, hfill(215))
         .idx({ 0 })
         .dense(RAD_I32, { 1, 4 }, [](int64_t i) { return (float)(9 - i); })
         .none()
         .dense(RAD_F32, { mw * ratio, n }, sent, true)
         .dense(RAD_I32, { mw }, sent, true)
         .dense(RAD_I32, { mw }, sent, true)
         .dense(RAD_I32, { 1 }, sent, true)
         .pi("M", T).pi("n", n).pi("ratio", ratio).pi("ring", ring).pi("seqs", 1).pi("work", mw)
         .ps("dtype", "f32");
        hs.push_back(std::move(h));
    }

    /* ---- dflash_conv --------------------------------------------------------------------- */
    {
        /* Groups of 24 take the broadcast path at 8 lanes and the expanded one at 16. */
        const int64_t T = 9, H = 96, taps = 2, G = 24, NG = 4, bsz = 4;
        Hand h("delta [T, taps, NG] at padded strides, a rank-3 base, x at a row pitch, groups "
               "of 24", "dflash_conv", 6e-3);
        h.view(RAD_BF16, { T, H }, { H + 8, 1 }, T * (H + 8), hfill(220))
         .view(RAD_BF16, { T, taps, NG }, { 16, 6, 1 }, T * 16, hfill(221))
         .dense(RAD_BF16, { 1, taps, H }, hfill(222))
         .dense(RAD_BF16, { T, H }, sent, true)
         .pi("T", T).pi("hidden_size", H).pi("taps", taps).pi("group_size", G).pi("NG", NG)
         .pi("block_size", bsz).ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }
    {
        Hand h("zero rows are refused", "dflash_conv", 0.0);
        h.dense(RAD_F32, { 0, 32 }, sent)
         .dense(RAD_F32, { 0, 8 }, sent)
         .dense(RAD_F32, { 2, 32 }, hfill(223))
         .dense(RAD_F32, { 0, 32 }, sent, true)
         .pi("T", 0).pi("hidden_size", 32).pi("taps", 2).pi("group_size", 8).pi("NG", 4)
         .pi("block_size", 4).ps("dtype", "f32")
         .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(h));
    }
}

/* ================================================================== weights of several planes
 *
 * A weight declared over several planes of one stored tensor reaches a host row joined
 * (abi/rad_planes.h): an fp8 or int8 lm head, E4M3 mixing and fc rows with f32 group scales, and
 * the scale and zero of the zero-point grids. No description draws one, so each case builds the
 * canonical planes, asks both libraries' layout hooks for the stored form (which must agree byte
 * for byte and invert back to the planes), and launches both over those bytes.
 *
 * Each GEMM case is also held to an answer neither library computed: the core's affine decoder
 * (core/format/encoding.h) dequantises the planes and the activation's codes, and a double
 * product of the two is the output. In a standalone build, which has no core, that comparison is
 * left out and said so. */

/* A weight's canonical planes over [rows, cols], filled by value. */
struct Planes {
    RadEncoding enc{};
    int64_t rows = 0, cols = 0;
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> prow, pcol;

    Planes(const RadEncoding& e, int64_t r, int64_t c) : enc(e), rows(r), cols(c) {
        for (int i = 0; i < e.n_planes; ++i) {
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&e.plane[i], r, c, &pr, &pc);
            prow.push_back(pr);
            pcol.push_back(pc);
            data.emplace_back((size_t)(pr * rad_enc_row_bytes(e.plane[i].dtype, pc)), 0);
        }
    }
    uint8_t* row_at(int i, int64_t r) {
        return data[(size_t)i].data() + r * rad_enc_row_bytes(enc.plane[i].dtype, pcol[(size_t)i]);
    }
    /* Element (r, c) of plane i: a float stored in its dtype, or an integer code. */
    void set(int i, int64_t r, int64_t c, float v) {
        const uint32_t dt = enc.plane[i].dtype;
        if (dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16) rad_store_f32(row_at(i, r), dt, c, v);
        else rad_store_code(row_at(i, r), dt, c, (int)v);
    }
    void fill(int i, const std::function<float(int64_t, int64_t)>& f) {
        for (int64_t r = 0; r < prow[(size_t)i]; ++r)
            for (int64_t c = 0; c < pcol[(size_t)i]; ++c) set(i, r, c, f(r, c));
    }
    /* The planes `sel` names as a layout hook takes them: dense rank 2, data when `with_data`. */
    std::vector<RadTensor> tensors(const std::vector<int>& sel, bool with_data) {
        std::vector<RadTensor> t;
        for (int i : sel) {
            RadTensor x{};
            x.dtype = enc.plane[i].dtype;
            x.rank = 2;
            x.shape[0] = prow[(size_t)i];
            x.shape[1] = pcol[(size_t)i];
            x.stride[0] = x.shape[1];
            x.stride[1] = 1;
            x.data = with_data ? data[(size_t)i].data() : nullptr;
            t.push_back(x);
        }
        return t;
    }
};

/* The joined bytes of planes `sel` for operand `opd`, through `lib`'s own hooks: RAD_OK from
 * layout, relayout to the size it named, unrelayout back to the same planes. */
bool join_with(const Lib& lib, const char* op, const std::vector<RadParam>& p, int opd,
               Planes& w, const std::vector<int>& sel, RadLayout* lay, std::vector<uint8_t>* out,
               std::string* why) {
    const RadKernelInfo* k = lib.row(op);
    if (!k || !k->layout || !k->relayout || !k->unrelayout) {
        *why = std::string(lib.info->name) + " has no layout hooks on " + op;
        return false;
    }
    std::vector<RadTensor> geo = w.tensors(sel, false);
    const int st = k->layout(p.data(), (int)p.size(), opd, &w.enc, sel.data(), geo.data(),
                             (int)sel.size(), lay);
    if (st != RAD_OK) {
        *why = std::string(lib.info->name) + "'s layout hook answered " + std::to_string(st);
        return false;
    }
    out->assign((size_t)lay->bytes, 0xA5);
    std::vector<RadTensor> full = w.tensors(sel, true);
    if (k->relayout(p.data(), (int)p.size(), opd, &w.enc, sel.data(), full.data(),
                    (int)sel.size(), out->data(), lay->bytes) != RAD_OK) {
        *why = std::string(lib.info->name) + "'s relayout refused its own layout";
        return false;
    }
    std::vector<std::vector<uint8_t>> back;
    std::vector<RadTensor> bt = w.tensors(sel, false);
    for (size_t j = 0; j < sel.size(); ++j) {
        back.emplace_back(w.data[(size_t)sel[j]].size(), 0);
        bt[j].data = back.back().data();
    }
    if (k->unrelayout(p.data(), (int)p.size(), opd, &w.enc, sel.data(), out->data(), lay->bytes,
                      bt.data(), (int)sel.size()) != RAD_OK) {
        *why = std::string(lib.info->name) + "'s unrelayout refused its own stored form";
        return false;
    }
    for (size_t j = 0; j < sel.size(); ++j)
        if (back[j] != w.data[(size_t)sel[j]]) {
            *why = std::string(lib.info->name) + "'s unrelayout did not give back plane " +
                   std::to_string(sel[j]);
            return false;
        }
    return true;
}

/* Both libraries' joined form of `w`: identical layouts and bytes, or a failure in `why`. */
bool join_both(const Check& ck, const char* op, const std::vector<RadParam>& p, int opd,
               Planes& w, const std::vector<int>& sel, std::vector<uint8_t>* out,
               uint32_t* dtype, std::string* why) {
    RadLayout la{}, lb{};
    std::vector<uint8_t> a, b;
    if (!join_with(*ck.ref, op, p, opd, w, sel, &la, &a, why)) return false;
    if (!ck.lib->row(op) || !ck.lib->row(op)->layout) {
        /* A host library without the hooks reads nothing joined; the case reports it absent. */
        *out = a;
        *dtype = la.dtype;
        return true;
    }
    if (!join_with(*ck.lib, op, p, opd, w, sel, &lb, &b, why)) return false;
    if (std::strcmp(la.tag ? la.tag : "", lb.tag ? lb.tag : "") != 0 || la.bytes != lb.bytes ||
        la.dtype != lb.dtype || a != b) {
        *why = "the two libraries store the joined planes differently";
        return false;
    }
    *out = a;
    *dtype = la.dtype;
    return true;
}

/* The weight the core's affine decoder gives for `w`'s planes, f32 [rows, cols]; empty when this
 * build has no core. */
std::vector<float> core_decode(const Planes& w) {
#ifdef AVX_CHECK_CORE_DECODER
    std::vector<const void*> d;
    for (const auto& v : w.data) d.push_back(v.data());
    rad::EncodedView v;
    rad::enc_view(w.enc, w.rows, w.cols, d.data(), &v);
    std::vector<float> out((size_t)(w.rows * w.cols));
    std::string why;
    if (rad::enc_decode_rows(v, 0, w.rows, out.data(), &why) != RAD_OK) {
        std::printf("  core decoder refused a hand-built weight: %s\n", why.c_str());
        return {};
    }
    return out;
#else
    (void)w;
    return {};
#endif
}

/* y[m][n] = sum_k a[m][off(n) + k] * W[n][k] in double: `a` [M, acols], W [N, K], `off(n)` the
 * column where row n's part starts. */
std::vector<float> oracle_gemm(const std::vector<float>& a, int64_t acols,
                               const std::vector<float>& W, int64_t M, int64_t N, int64_t K,
                               const std::function<int64_t(int64_t)>& off) {
    std::vector<float> y((size_t)(M * N), 0.0f);
    if (W.empty()) return {};
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0;
            const int64_t o = off(n);
            for (int64_t k = 0; k < K; ++k)
                s += (double)a[(size_t)(m * acols + o + k)] * (double)W[(size_t)(n * K + k)];
            y[(size_t)(m * N + n)] = (float)s;
        }
    return y;
}

/* Deterministic integer in [lo, hi] for element i under seed s. */
int64_t hint(uint64_t s, int64_t i, int64_t lo, int64_t hi) {
    uint64_t x = s * 0x9E3779B97F4A7C15ull + (uint64_t)i;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return lo + (int64_t)(x % (uint64_t)(hi - lo + 1));
}

/* A case whose joined weight did not build is a failure of the case, reported as one. */
struct Broken {
    std::vector<std::string> why;
};

void planes_cases(const Check& ck, std::vector<Hand>& hs, Broken& broken) {
    const auto zero = [](int64_t) { return 0.0f; };
    auto rp = [](std::initializer_list<std::pair<const char*, long long>> ints,
                 std::initializer_list<std::pair<const char*, const char*>> strs) {
        std::vector<RadParam> p;
        for (auto& [k, v] : ints) p.push_back(RadParam{ k, RAD_P_INT, v, 0, nullptr, 0 });
        for (auto& [k, v] : strs) p.push_back(RadParam{ k, RAD_P_STR, 0, 0, v, 0 });
        return p;
    };
    /* The stored bytes as the engine passes them: the joined form's dtype, the declared extents. */
    auto add_joined = [](Hand& h, uint32_t dt, std::vector<int64_t> shape,
                         const std::vector<uint8_t>& bytes) -> Hand& {
        const int64_t span = (int64_t)bytes.size() * 8 / rad_dtype_bits(dt);
        return h.bytes(dt, shape, { shape.size() > 1 ? shape[1] : 1, 1 }, span, bytes, false);
    };

    /* ---- the lm head: E4M3 with a bf16 scale per 128 x 128, int8 with one per (row, 128) ---- *
     * N = 200 gives a second, partial block row and K = 320 a partial last column block. M = 9
     * is past every register block's edge and 5 inside it. */
    for (int i8 : { 0, 1 }) {
        const int64_t N = i8 ? 100 : 200, K = i8 ? 256 : 320;
        RadEncoding e = rad_enc_affine(i8 ? RAD_I8 : RAD_F8E4M3, RAD_BF16, i8 ? 1 : 128, 128);
        Planes w(e, N, K);
        w.fill(0, [i8](int64_t r, int64_t c) {
            const int64_t v = hint(200 + i8, r * 4099 + c, i8 ? -127 : 0, i8 ? 127 : 254);
            return i8 ? (float)v : (float)(v == 0x7f ? 0x7e : v);
        });
        w.fill(1, [](int64_t r, int64_t c) { return 0.004f * (float)(1 + (r * 7 + c) % 9); });
        const std::vector<float> W = core_decode(w);
        for (const char* adt : { "bf16", "f32" })
            for (int64_t M : { 1, 5, 9 }) {
                const std::vector<RadParam> p =
                    rp({ { "M", M }, { "n_vocab", N }, { "n_embd", K } }, { { "dtype", adt } });
                std::vector<uint8_t> j;
                uint32_t jdt = 0;
                std::string why;
                if (!join_both(ck, "logits_gemm", p, 1, w, { 0, 1 }, &j, &jdt, &why)) {
                    broken.why.push_back(std::string("logits_gemm: ") + why);
                    continue;
                }
                const uint32_t ad = rad_dtype_parse(adt);
                std::vector<float> a((size_t)(M * K));
                for (int64_t t = 0; t < M * K; ++t) a[(size_t)t] = hfill(210)(t);
                if (ad == RAD_BF16)
                    for (float& v : a) v = rad_bf16_to_f32(rad_f32_to_bf16(v));
                const std::string nm =
                    std::string(i8 ? "i8*bf16[1x128] head" : "fp8_e4m3*bf16[128x128] head") +
                    ", " + adt + ", M=" + std::to_string(M);
                Hand h(nm.c_str(), "logits_gemm", ad == RAD_F32 ? 2e-5 : 8e-3);
                h.dense(ad, { M, K }, [&a](int64_t t) { return a[(size_t)t]; });
                add_joined(h, jdt, { N, K }, j);
                h.dense(RAD_F32, { M, N }, zero, true)
                 .pi("M", M).pi("n_vocab", N).pi("n_embd", K).ps("dtype", adt);
                h.oracle_opd = 2;
                h.oracle = oracle_gemm(a, K, W, M, N, K, [](int64_t) { return 0; });
                h.oracle_tol = 1e-5;
                hs.push_back(std::move(h));
            }

        /* logit_rerank over the same head: candidates across both block rows, a pad, and ids
         * outside this rank's rows. */
        const int64_t M = 3, R = 5, VOFF = 20;
        const std::vector<RadParam> p = rp({ { "M", M }, { "R", R }, { "n_vocab", N },
                                             { "n_embd", K }, { "vocab_off", VOFF } },
                                           { { "dtype", "f32" } });
        std::vector<uint8_t> j;
        uint32_t jdt = 0;
        std::string why;
        if (!join_both(ck, "logit_rerank", p, 1, w, { 0, 1 }, &j, &jdt, &why)) {
            broken.why.push_back(std::string("logit_rerank: ") + why);
            continue;
        }
        static const int32_t kIds[3][5] = { { 21, 150, 19, -1, 60 },
                                            { 199, 20, 140, 77, 300 },
                                            { 33, 34, 35, 36, 37 } };
        Hand h(i8 ? "int8 head joined, ids across the shard" : "fp8 head joined, ids across both "
               "block rows", "logit_rerank", 1e-5);
        h.dense(RAD_F32, { M, K }, hfill(220));
        add_joined(h, jdt, { N, K }, j);
        h.dense(RAD_I32, { M, R }, [](int64_t t) { return (float)kIds[t / 5][t % 5]; })
         .dense(RAD_I32, { M, R }, [](int64_t) { return 7.0f; }, true)
         .dense(RAD_F32, { M, R }, zero, true)
         .dense(RAD_F32, { M, R, 2 }, zero, true)
         .pi("M", M).pi("R", R).pi("n_vocab", N).pi("n_embd", K).pi("vocab_off", VOFF)
         .ps("dtype", "f32");
        hs.push_back(std::move(h));
    }

    /* ---- hc_read and mtp_enter over E4M3 rows with f32 scales in their rows' groups --------- *
     * Flash-Next's geometry scaled down: mix_down [lowrank, hc*n] per 128 columns, mix_up
     * [hc*n, lowrank] per quarter row (lowrank 64: groups of 16), the fc matrices per 128. */
    {
        const int64_t M = 5, n = 128, hc = 4, lr = 64, HN = hc * n;
        auto rows = [&](int64_t R, int64_t C, int64_t G, uint64_t seed) {
            Planes w(rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, G), R, C);
            w.fill(0, [seed](int64_t r, int64_t c) {
                const int64_t v = hint(seed, r * 977 + c, 0, 254);
                return (float)(v == 0x7f ? 0x7e : v);
            });
            w.fill(1, [seed](int64_t r, int64_t c) {
                return 0.002f * (float)(1 + hint(seed + 1, r * 31 + c, 0, 6)) + 1e-7f * (float)r;
            });
            return w;
        };
        Planes dn = rows(lr, HN, 128, 230), up = rows(HN, lr, lr / 4, 232);
        for (const char* dt : { "bf16", "f32" })
            for (int inject : { 1, 0 }) {
                const std::vector<RadParam> p =
                    rp({ { "M", M }, { "n", n }, { "hc", hc }, { "lowrank", lr },
                         { "inject", inject }, { "group", 0 } },
                       { { "dtype", dt }, { "mix", "e4m3" } });
                std::vector<uint8_t> jd, ju;
                uint32_t dd = 0, ud = 0;
                std::string why;
                if (!join_both(ck, "hc_read", p, 2, dn, { 0, 1 }, &jd, &dd, &why) ||
                    !join_both(ck, "hc_read", p, 3, up, { 0, 1 }, &ju, &ud, &why)) {
                    broken.why.push_back(std::string("hc_read: ") + why);
                    continue;
                }
                const uint32_t ad = rad_dtype_parse(dt);
                const std::string nm = std::string("mix e4m3 joined, ") + dt +
                                       (inject ? ", inject" : ", final mixer");
                Hand h(nm.c_str(), "hc_read", ad == RAD_F32 ? 1e-5 : 6e-3);
                h.dense(ad, { M, HN }, hfill(234))
                 .dense(RAD_F32, { HN }, [](int64_t i) { return 0.1f * (float)(i % 7); });
                add_joined(h, dd, { lr, HN }, jd);
                add_joined(h, ud, { HN, lr }, ju);
                if (inject) h.dense(RAD_BF16, { hc, HN }, hfill(235, 0.05f));
                else        h.none();
                h.dense(ad, { M, n }, zero, true);
                if (inject) h.dense(RAD_F32, { M, hc }, zero, true);
                else        h.none();
                h.none().none().none().none()
                 .pi("M", M).pi("n", n).pi("hc", hc).pi("lowrank", lr).pf("eps", 1e-6)
                 .ps("dtype", dt).pf("wadd", 1.0).pi("inject", inject).pi("group", 0)
                 .ps("mix", "e4m3");
                hs.push_back(std::move(h));
            }

        Planes fh = rows(n, n, 128, 236), fe = rows(n, n, 128, 238);
        for (const char* dt : { "bf16", "f32" }) {
            const std::vector<RadParam> p =
                rp({ { "M", M }, { "n", n }, { "hc", hc }, { "ngroup", hc } },
                   { { "dtype", dt }, { "fc", "e4m3" } });
            std::vector<uint8_t> jh, je;
            uint32_t hd = 0, ed = 0;
            std::string why;
            if (!join_both(ck, "mtp_enter", p, 4, fh, { 0, 1 }, &jh, &hd, &why) ||
                !join_both(ck, "mtp_enter", p, 5, fe, { 0, 1 }, &je, &ed, &why)) {
                broken.why.push_back(std::string("mtp_enter: ") + why);
                continue;
            }
            const uint32_t ad = rad_dtype_parse(dt);
            Hand h((std::string("fc e4m3 joined, ") + dt).c_str(), "mtp_enter",
                   ad == RAD_F32 ? 1e-5 : 6e-3);
            h.dense(ad, { M, HN }, hfill(240))
             .dense(ad, { M, n }, hfill(241))
             .dense(RAD_F32, { HN }, [](int64_t i) { return 0.05f * (float)(i % 5); })
             .dense(RAD_F32, { n }, [](int64_t i) { return 0.03f * (float)(i % 3); });
            add_joined(h, hd, { n, n }, jh);
            add_joined(h, ed, { n, n }, je);
            h.dense(ad, { M, HN }, zero, true)
             .pi("M", M).pi("n", n).pi("hc", hc).pf("eps", 1e-6).ps("dtype", dt).pf("wadd", 1.0)
             .pi("ngroup", hc).ps("fc", "e4m3");
            hs.push_back(std::move(h));
        }
    }

    /* ---- the zero-point grids of gemm_nt_q ---------------------------------------------------- *
     * Codes across their whole range, zeros at both ends of it (0 and the largest code) beside
     * values between, an f16 scale and a u8 zero per (row, 128 of K). M = 13 is the small-M
     * paths' 8 + 4 + 1 rows, 17 past them, padding quant_act_i8's code sums to 32 rows; N = 40,
     * 41, 48 and 96 are not multiples of the driver's 6-row block, 40 and 41 not of the small-M
     * paths' 16, and 41 leaves the 16-bit path's row pairs one row over. The ParoQuant grids
     * also run in parts (pb1, pb2), each part meeting its own activation copy; the int8 ones also
     * at M = 40 over K = 2304, two of the bf16 grouped driver's K chunks. */
    struct Grid { const char* dtype; uint32_t codes; int act; int parts; };
    static const Grid kGrids[] = {
        { "w2a8", RAD_U2, 0, 1 },         { "w4a8_asym", RAD_U4, 0, 1 },
        { "w4a16_asym", RAD_U4, 2, 1 },   { "w4a16_asym", RAD_U4, 2, 2 },
        { "w5a16", RAD_U5, 2, 1 },        { "w5a16", RAD_U5, 2, 3 },
        { "w4a8i8_asym", RAD_U4, 1, 1 },  { "w4a8i8_asym", RAD_U4, 1, 2 },
        { "w5a8i8", RAD_U5, 1, 1 },       { "w5a8i8", RAD_U5, 1, 3 },
    };
    for (const Grid& gd : kGrids)
        for (int64_t M : { 1, 13, 17, 40 }) {
            if (M == 40 && gd.act != 1) continue;
            const int64_t N = gd.parts == 3 ? 96 : gd.parts == 2 ? 48 : gd.act == 0 ? 40 : 41;
            const int64_t K = M == 40 ? 2304 : 256, GS = 128;
            const int64_t G = K / GS, P = gd.parts;
            const int64_t top = (1 << rad_dtype_bits(gd.codes)) - 1;
            RadEncoding e = rad_enc_affine(gd.codes, RAD_F16, 1, GS);
            rad_enc_add_plane(&e, "zero", RAD_U8, 1, GS);
            Planes w(e, N, K);
            w.fill(0, [top](int64_t r, int64_t c) {
                return (float)hint(250, r * 1031 + c, 0, top);
            });
            w.fill(1, [](int64_t r, int64_t c) { return 0.01f * (float)(1 + (r * 3 + c) % 11); });
            w.fill(2, [top, G](int64_t r, int64_t c) {
                const int64_t k = (r * G + c) % 4;
                return (float)(k == 0 ? 0 : k == 1 ? top : hint(251, r * G + c, 0, top));
            });
            const int64_t pb1 = P > 1 ? (P == 3 ? 32 : 16) : 0, pb2 = P == 3 ? 64 : 0;
            std::vector<RadParam> p = rp({ { "M", M }, { "N", N }, { "K", K }, { "group", GS } },
                                         { { "dtype", gd.dtype } });
            if (pb1) p.push_back(RadParam{ "pb1", RAD_P_INT, pb1, 0, nullptr, 0 });
            if (pb2) p.push_back(RadParam{ "pb2", RAD_P_INT, pb2, 0, nullptr, 0 });
            std::vector<uint8_t> j;
            uint32_t jdt = 0;
            std::string why;
            if (!join_both(ck, "gemm_nt_q", p, 3, w, { 1, 2 }, &j, &jdt, &why)) {
                broken.why.push_back(std::string("gemm_nt_q ") + gd.dtype + ": " + why);
                continue;
            }
            const int64_t AC = P * K;
            /* The activation as the grid means it, for the oracle, and its operands. */
            std::vector<float> af((size_t)(M * AC));
            std::vector<float> codes((size_t)(M * AC)), sa;
            if (gd.act == 0) {
                for (int64_t t = 0; t < M * AC; ++t)
                    codes[(size_t)t] = (float)hint(252, t, -127, 127);
                sa.resize((size_t)M);
                for (int64_t m = 0; m < M; ++m) sa[(size_t)m] = 0.003f * (float)(1 + m % 5);
                for (int64_t t = 0; t < M * AC; ++t)
                    af[(size_t)t] = codes[(size_t)t] * sa[(size_t)(t / AC)];
            } else if (gd.act == 1) {
                for (int64_t t = 0; t < M * AC; ++t)
                    codes[(size_t)t] = (float)hint(253, t, -127, 127);
                sa.resize((size_t)(M * P * G));
                for (size_t t = 0; t < sa.size(); ++t) sa[t] = 0.002f * (float)(1 + t % 7);
                for (int64_t t = 0; t < M * AC; ++t)
                    af[(size_t)t] =
                        codes[(size_t)t] * sa[(size_t)((t / AC) * P * G + (t % AC) / GS)];
            } else {
                for (int64_t t = 0; t < M * AC; ++t)
                    af[(size_t)t] = rad_f16_to_f32(rad_f32_to_f16(hfill(254)(t)));
            }
            const std::string nm = std::string(gd.dtype) +
                                   (P > 1 ? ", " + std::to_string(P) + " parts" : "") + ", M=" +
                                   std::to_string(M) + (K != 256 ? ", K=" + std::to_string(K) : "") +
                                   ", zeros 0 and " + std::to_string(top);
            Hand h(nm.c_str(), "gemm_nt_q", 8e-3);
            if (gd.act == 0) {
                /* quant_act_i8's codes in the step order, its per-row scale and code sums. */
                std::vector<float> st((size_t)(M * K));
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t c = 0; c < K; ++c) {
                        const int64_t pos = (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1);
                        st[(size_t)(m * K + pos)] = codes[(size_t)(m * K + c)];
                    }
                const int64_t mpad = (M + 15) / 16 * 16;
                std::vector<float> sums((size_t)(G * mpad), 0.0f);
                for (int64_t g = 0; g < G; ++g)
                    for (int64_t m = 0; m < M; ++m) {
                        float s = 0;
                        for (int64_t k = g * GS; k < (g + 1) * GS; ++k)
                            s += codes[(size_t)(m * K + k)];
                        sums[(size_t)(g * mpad + m)] = s;
                    }
                h.dense(RAD_I8, { M, K }, [st](int64_t t) { return st[(size_t)t]; })
                 .dense(RAD_F32, { M, 1 }, [sa](int64_t t) { return sa[(size_t)t]; });
                h.raw(gd.codes, { N, K }, w.data[0].data());
                add_joined(h, jdt, { N, K }, j);
                h.dense(RAD_BF16, { M, N }, zero, true)
                 .dense(RAD_F32, { G, mpad }, [sums](int64_t t) { return sums[(size_t)t]; })
                 .none();
            } else if (gd.act == 1) {
                h.dense(RAD_I8, { M, AC }, [codes](int64_t t) { return codes[(size_t)t]; })
                 .dense(RAD_F32, { M, P * G }, [sa](int64_t t) { return sa[(size_t)t]; });
                h.raw(gd.codes, { N, K }, w.data[0].data());
                add_joined(h, jdt, { N, K }, j);
                h.dense(RAD_BF16, { M, N }, zero, true).none().none();
            } else {
                h.dense(RAD_F16, { M, AC }, [af](int64_t t) { return af[(size_t)t]; }).none();
                h.raw(gd.codes, { N, K }, w.data[0].data());
                add_joined(h, jdt, { N, K }, j);
                h.dense(RAD_BF16, { M, N }, zero, true).none().none();
            }
            h.pi("M", M).pi("N", N).pi("K", K).pi("group", GS).ps("dtype", gd.dtype);
            if (pb1) h.pi("pb1", pb1);
            if (pb2) h.pi("pb2", pb2);
            /* The oracle reads the codes untransformed, as the GEMM does: a ParoQuant weight's
             * rotation is the activation's. */
            const int64_t e1 = pb1 ? pb1 : N, e2 = pb2 ? pb2 : N;
            h.oracle_opd = 4;
            h.oracle = oracle_gemm(af, AC, core_decode(w), M, N, K, [e1, e2, K](int64_t n) {
                return (n < e1 ? 0 : n < e2 ? 1 : 2) * K;
            });
            h.oracle_tol = 6e-3;
            hs.push_back(std::move(h));
        }

    /* ---- refusals ------------------------------------------------------------------------ *
     * A zero-point grid's `b_scale` that is not joined has no zero to read: both refuse. */
    {
        const int64_t M = 2, N = 16, K = 128;
        Hand h("w4a16_asym with a plain scale, no zero", "gemm_nt_q", 1e-5);
        h.dense(RAD_F16, { M, K }, hfill(260))
         .none()
         .dense(RAD_U4, { N, K }, [](int64_t i) { return (float)(i % 16); })
         .dense(RAD_F16, { N, 1 }, [](int64_t) { return 0.5f; })
         .dense(RAD_BF16, { M, N }, zero, true)
         .none().none()
         .pi("M", M).pi("N", N).pi("K", K).pi("group", 128).ps("dtype", "w4a16_asym")
         .refuses(RAD_E_DTYPE);
        hs.push_back(std::move(h));
    }
}

/* What the layout hooks refuse, identically in both libraries: an encoding the op does not
 * compute is RAD_E_DTYPE, and a single plane is stored as it is (RAD_E_UNSUPPORTED). */
bool planes_refusals(const Check& ck, int* run) {
    struct R { const char* what; const char* op; int opd; std::vector<RadParam> p; RadEncoding e;
               std::vector<int> sel; int want; };
    std::vector<R> rs;
    auto P = [](std::initializer_list<std::pair<const char*, long long>> ints, const char* dtype) {
        std::vector<RadParam> p;
        for (auto& [k, v] : ints) p.push_back(RadParam{ k, RAD_P_INT, v, 0, nullptr, 0 });
        if (dtype) p.push_back(RadParam{ "dtype", RAD_P_STR, 0, 0, dtype, 0 });
        return p;
    };
    auto zp = [](uint32_t codes, uint32_t zdt, int64_t g, const char* tf) {
        RadEncoding e = rad_enc_affine(codes, RAD_F16, 1, g);
        rad_enc_add_plane(&e, "zero", zdt, 1, g);
        if (tf) rad_enc_copy_str(e.transform, tf);
        return e;
    };
    const auto head = P({ { "M", 4 }, { "n_vocab", 256 }, { "n_embd", 256 } }, "bf16");
    const auto gq = [&](const char* dt) {
        return P({ { "M", 4 }, { "N", 256 }, { "K", 256 }, { "group", 128 } }, dt);
    };
    rs.push_back({ "a u4 head", "logits_gemm", 1, head, rad_enc_affine(RAD_U4, RAD_BF16, 1, 128),
                   { 0, 1 }, RAD_E_DTYPE });
    rs.push_back({ "an fp8 head under fwht128", "logits_gemm", 1, head,
                   [] { RadEncoding e = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
                        rad_enc_copy_str(e.transform, "fwht128"); return e; }(), { 0, 1 },
                   RAD_E_DTYPE });
    rs.push_back({ "a plain bf16 head", "logits_gemm", 1, head, rad_enc_plain(RAD_BF16), { 0 },
                   RAD_E_UNSUPPORTED });
    rs.push_back({ "the scale plane alone", "gemm_nt_q", 3, gq("w2a8"),
                   zp(RAD_U2, RAD_U8, 128, nullptr), { 1 }, RAD_E_UNSUPPORTED });
    rs.push_back({ "w2a8 over u4 codes", "gemm_nt_q", 3, gq("w2a8"),
                   zp(RAD_U4, RAD_U8, 128, nullptr), { 1, 2 }, RAD_E_DTYPE });
    rs.push_back({ "w5a16 with a zero per 64", "gemm_nt_q", 3, gq("w5a16"),
                   zp(RAD_U5, RAD_U8, 64, nullptr), { 1, 2 }, RAD_E_DTYPE });
    rs.push_back({ "w4a16_asym under fwht128", "gemm_nt_q", 3, gq("w4a16_asym"),
                   zp(RAD_U4, RAD_U8, 128, "fwht128"), { 1, 2 }, RAD_E_DTYPE });
    rs.push_back({ "zero then scale", "gemm_nt_q", 3, gq("w4a16_asym"),
                   zp(RAD_U4, RAD_U8, 128, nullptr), { 2, 1 }, RAD_E_DTYPE });
    rs.push_back({ "scale and zero at fp8a8", "gemm_nt_q", 3, gq("fp8a8"),
                   zp(RAD_U4, RAD_U8, 128, nullptr), { 1, 2 }, RAD_E_DTYPE });
    bool ok = true;
    for (R& r : rs) {
        for (const Lib* lib : { ck.ref, ck.lib }) {
            const RadKernelInfo* k = lib->row(r.op);
            if (!k || !k->layout) {
                if (lib == ck.lib && !ck.own) continue;
                std::printf("  FAIL %-20s %s\n       %s has no layout hook\n", r.op, r.what,
                            lib->info->name);
                ok = false;
                continue;
            }
            std::vector<RadTensor> t(r.sel.size());
            for (size_t i = 0; i < r.sel.size(); ++i) {
                int64_t pr = 0, pc = 0;
                rad_enc_plane_dims(&r.e.plane[r.sel[i]], 256, 256, &pr, &pc);
                t[i] = RadTensor{};
                t[i].dtype = r.e.plane[r.sel[i]].dtype;
                t[i].rank = 2;
                t[i].shape[0] = pr;
                t[i].shape[1] = pc;
            }
            RadLayout L{};
            const int st = k->layout(r.p.data(), (int)r.p.size(), r.opd, &r.e, r.sel.data(),
                                     t.data(), (int)t.size(), &L);
            ++*run;
            if (st != r.want) {
                std::printf("  FAIL %-20s %s\n       %s's layout hook answered %d, expected %d\n",
                            r.op, r.what, lib->info->name, st, r.want);
                ok = false;
            } else if (ck.verbose) {
                std::printf("  %-22s %-46s %s: %d\n", r.op, r.what, lib->info->name, st);
            }
        }
    }
    return ok;
}

/* ================================================================== a chunked audio encoder
 * The forms Qwen3-ASR's tower issues and EmbeddingGemma 2's subsampler keeps issuing, which no
 * description draws: conv2d with no bias or no row mask, segment bounds in frames read at a
 * cu_div, attention in blocks over several clips, rows no segment covers left as they were, and
 * a linear whose residual is its own output. */
void encoder_cases(std::vector<Hand>& hs) {
    /* gemm_nt_bias with `res` the output itself, as every pre-norm encoder layer adds its
     * attention output and its feed-forward to the residual stream in place (Qwen3-ASR's tower,
     * the Qwen-VL and SigLIP2 vision towers). The residual must be read before the product is
     * stored over it; both libraries are held to an answer computed apart, since a library that
     * stores first adds the product to itself and agrees with another that does the same. At
     * Qwen3-ASR's geometries: the out-projection (K 1024) and the second feed-forward linear
     * (K 4096), 76 rows (one 6 s clip), and a gelu epilogue on a smaller shape. */
    struct Res { int64_t M, N, K; const char* act; uint32_t dt; };
    for (const Res& c : { Res{ 76, 1024, 1024, "none", RAD_BF16 }, Res{ 76, 256, 4096, "none", RAD_BF16 },
                          Res{ 9, 96, 64, "gelu", RAD_BF16 }, Res{ 9, 96, 64, "none", RAD_F32 } }) {
        auto round = [&](float v) {
            return c.dt == RAD_BF16 ? rad_bf16_to_f32(rad_f32_to_bf16(v)) : v;
        };
        std::vector<float> a((size_t)(c.M * c.K)), w((size_t)(c.N * c.K)), bias((size_t)c.N),
            res((size_t)(c.M * c.N));
        for (size_t i = 0; i < a.size(); ++i) a[i] = round(hfill(951)((int64_t)i));
        for (size_t i = 0; i < w.size(); ++i) w[i] = round(hfill(952, 0.05f)((int64_t)i));
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = round(hfill(953)((int64_t)i));
        for (size_t i = 0; i < res.size(); ++i) res[i] = round(hfill(954, 4.0f)((int64_t)i));
        std::vector<float> want = oracle_gemm(a, c.K, w, c.M, c.N, c.K, [](int64_t) { return 0; });
        for (int64_t m = 0; m < c.M; ++m)
            for (int64_t n = 0; n < c.N; ++n) {
                float v = round(want[(size_t)(m * c.N + n)] + bias[(size_t)n]);
                if (!std::strcmp(c.act, "gelu"))
                    v = round(0.5f * v * (1.0f + sc_erf(v * 0.70710678118654752f)));
                want[(size_t)(m * c.N + n)] = round(res[(size_t)(m * c.N + n)] + v);
            }
        const std::string nm = "res is y, M=" + std::to_string(c.M) + " N=" + std::to_string(c.N) +
                               " K=" + std::to_string(c.K) + " " + c.act +
                               (c.dt == RAD_BF16 ? " bf16" : " f32");
        Hand h(nm.c_str(), "gemm_nt_bias", c.dt == RAD_F32 ? 1e-5 : 8e-3);
        h.dense(c.dt, { c.M, c.K }, [&a](int64_t i) { return a[(size_t)i]; })
         .dense(c.dt, { c.N, c.K }, [&w](int64_t i) { return w[(size_t)i]; })
         .dense(c.dt, { c.N }, [&bias](int64_t i) { return bias[(size_t)i]; })
         .dense(c.dt, { c.M, c.N }, [&res](int64_t i) { return res[(size_t)i]; }, true)
         .alias(3)
         .pi("M", c.M).pi("N", c.N).pi("K", c.K).ps("dtype", c.dt == RAD_BF16 ? "bf16" : "f32")
         .ps("act", c.act);
        h.oracle_opd = 4;
        h.oracle = want;
        h.oracle_tol = c.dt == RAD_F32 ? 1e-5 : 1e-2;
        hs.push_back(std::move(h));
    }
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const double tol = dt == RAD_F32 ? 1e-5 : 6e-3;
        /* Two clips of 10 and 14 frames, 12 output rows in all, with a row mask and no bias: the
         * call a model without a bias makes, five operands as before the bias existed. */
        {
            const int64_t M = 24, W = 16, cin = 1, cout = 8, Wo = 8;
            Hand h(dt == RAD_F32 ? "no bias, masked rows, two clips" : "no bias, masked rows, bf16",
                   "conv2d", tol);
            h.dense(dt, { M, W * cin }, hfill(901))
             .dense(dt, { cout, cin, 3, 3 }, hfill(902, 0.5f))
             .idx({ 0, 10, 24 })
             .dense(RAD_I32, { M }, [](int64_t i) { return (float)(i % 7 != 3); })
             .dense(dt, { 12, Wo * cout }, hfill(903), true)
             .pi("M", M).pi("width", W).pi("cin", cin).pi("cout", cout).pi("kh", 3).pi("kw", 3)
             .pi("stride", 2).pi("pad", 1).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        /* A bias and gelu with no row mask, over a time-last kernel. */
        {
            const int64_t M = 16, W = 8, cin = 3, cout = 5, Wo = 4;
            Hand h(dt == RAD_F32 ? "bias, gelu, time last, no mask" : "bias, gelu, no mask, bf16",
                   "conv2d", tol);
            h.dense(dt, { M, W * cin }, hfill(911))
             .dense(dt, { cout, cin, 3, 3 }, hfill(912, 0.5f))
             .idx({ 0, 6, 16 })
             .none()
             .dense(dt, { 8, Wo * cout }, hfill(913), true)
             .dense(dt, { cout }, hfill(914))
             .pi("M", M).pi("width", W).pi("cin", cin).pi("cout", cout).pi("kh", 3).pi("kw", 3)
             .pi("stride", 2).pi("pad", 1).ps("dtype", dn).pi("t_last", 1).ps("act", "gelu");
            hs.push_back(std::move(h));
        }
        /* The tower's output projection as Qwen3-ASR issues it: a 1 x f3 kernel over rows whose
         * clip bounds are in frames (cu_div 8), no mask, and four rows past the last clip. */
        {
            const int64_t R = 24, f3 = 2, C = 6;
            Hand h(dt == RAD_F32 ? "1 x 2 kernel, cu_div 8, uncovered rows"
                                 : "1 x 2 kernel, cu_div 8, bf16", "conv2d_unfold", tol);
            h.dense(dt, { R, f3 * C }, hfill(921))
             .idx({ 0, 104, 160 })
             .none()
             .dense(dt, { R, C * f3 }, hfill(922), true)
             .pi("M", R).pi("width", f3).pi("cin", C).pi("kh", 1).pi("kw", f3).pi("stride", 1)
             .pi("pad", 0).pi("cu_div", 8).pi("valid_stride", 8).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        /* Attention in 13-row blocks over two clips of 30 and 10 rows, bounds in frames, q, k and v
         * column slices of one fused projection, as the tower's layers read it. */
        {
            const int64_t R = 40, H = 2, HD = 64, Wd = 3 * H * HD;
            Hand h(dt == RAD_F32 ? "blocks of 13 over two clips, cu_div 8, fused qkv"
                                 : "blocks of 13, cu_div 8, fused qkv, bf16", "attn_dense",
                   dt == RAD_F32 ? 1e-4 : 2e-2);
            h.view(dt, { R, H * HD }, { Wd, 1 }, R * Wd, hfill(931))
             .view(dt, { R, H * HD }, { Wd, 1 }, R * Wd, hfill(932))
             .view(dt, { R, H * HD }, { Wd, 1 }, R * Wd, hfill(933))
             .idx({ 0, 240, 320 })
             .dense(dt, { R, H * HD }, hfill(934), true)
             .pi("M", R).pi("head_dim", HD).pi("gqa", 1).pi("causal", 0).ps("dtype", dn)
             .pf("scale", 0.125).pi("max_seqlen", 30).pi("cu_div", 8).pi("block", 13);
            hs.push_back(std::move(h));
        }
        /* Positions restarting every 13 rows of three clips (bounds in frames), and seven rows
         * past the last clip that keep what they held. */
        {
            const int64_t R = 40, n = 64;
            Hand h(dt == RAD_F32 ? "three clips at cu_div 8, uncovered rows"
                                 : "three clips at cu_div 8, bf16", "seg_sinusoid", tol);
            h.idx({ 0, 104, 208, 264 })
             .dense(dt, { R, n }, hfill(941), true)
             .pi("M", R).pi("n", n).pi("period", 13).pi("cu_div", 8).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
    }
}

/* ================================================================== the audio generator */
/* Forms libref's descriptions do not draw: weights at a trained snake's scale (the drawn 0.02
 * leaves the sine almost linear) and arguments past the polynomial's range, SnakeBeta, several
 * sequences of uneven length starting past row 0 with an empty one, a convolution without its bias
 * and residual, a causal one with an even kernel, a depthwise one with a bias, transposed ones at
 * odd strides and at a kernel of one stride, sampler rows full of ties, a sampler with per-lane
 * guidance and a history (one lane twice), the window entries the run phase issues: unconditional
 * branches (pair -2), every flag combination, a window shorter than its overlap, a frame 0, an
 * ended lane; codes as a block of a wider plane; and refusals that must write nothing. Everything
 * but the convolutions and snake is held to zero error. */
void audio_cases(std::vector<Hand>& hs) {
    const auto zero = [](int64_t) { return 0.0f; };
    for (const uint32_t dt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
        const char* dn = dt == RAD_F32 ? "f32" : "bf16";
        const std::string sfx = std::string(", ") + dn;
        const double norm_tol = dt == RAD_F32 ? 1e-5 : 6e-3;
        {
            /* alpha 0.25 .. 4.2 and x in [-8, 8]: arguments to about 34. Row 3 holds 30000 (past
             * the polynomial's range, so the row takes the libm sine), row 2 a NaN, row 4 an
             * infinity. */
            const int64_t n = 67;
            Hand h(("trained alphas, a large argument, NaN and infinity" + sfx).c_str(), "snake",
                   norm_tol);
            h.dense(dt, { 5, n }, [n](int64_t i) {
                 if (i == 3 * n + 4) return 30000.0f;
                 if (i == 2 * n + 10) return NAN;
                 if (i == 4 * n + 66) return INFINITY;
                 return hfill(260, 8.0f)(i);
             })
             .dense(RAD_F32, { n }, [](int64_t c) { return 0.25f + 0.06f * (float)c; })
             .dense(dt, { 5, n }, zero, true)
             .pi("M", 5).pi("n", n).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Three sequences from row 2, the second empty, no bias and no residual; rows 0, 1,
             * 16 and 17 of y are outside every sequence and keep their bytes. */
            const int64_t cin = 20, cout = 13;
            Hand h(("rows 2..16 in three sequences, no bias or residual, tanh" + sfx).c_str(),
                   "conv1d", norm_tol);
            h.dense(dt, { 18, cin }, hfill(261))
             .dense(dt, { cout, cin, 5 }, hfill(262, 0.3f))
             .idx({ 2, 7, 7, 16 })
             .none()
             .dense(dt, { 18, cout }, hfill(263), true)
             .none()
             .pi("M", 18).pi("cin", cin).pi("cout", cout).pi("k", 5).pi("dil", 2)
             .ps("act", "tanh").ps("dtype", dn);
            hs.push_back(std::move(h));

            /* x rows at a wider pitch than its channels, with the residual and the bias. */
            Hand g(("input rows at a wider pitch, residual and bias" + sfx).c_str(), "conv1d",
                   norm_tol);
            g.view(dt, { 16, cin }, { cin + 3, 1 }, 16 * (cin + 3), hfill(264))
             .dense(dt, { cout, cin, 3 }, hfill(265, 0.3f))
             .idx({ 0, 9, 16 })
             .dense(dt, { 16, cout }, hfill(266))
             .dense(dt, { 16, cout }, zero, true)
             .dense(dt, { cout }, hfill(267, 0.5f))
             .pi("M", 16).pi("cin", cin).pi("cout", cout).pi("k", 3).pi("dil", 1)
             .ps("dtype", dn);
            hs.push_back(std::move(g));
        }
        for (const int64_t s : { 4, 2 }) {
            /* Sequences of 3, 1 and 5 rows from row 1: output rows [s, 10 s); the first s rows of
             * y keep their bytes. */
            const int64_t cin = 12, cout = 7;
            Hand h(("three uneven sequences from row 1, stride " + std::to_string(s) + sfx).c_str(),
                   "conv_t1d", norm_tol);
            h.dense(dt, { 10, cin }, hfill(268))
             .dense(dt, { cin, cout, 2 * s }, hfill(269, 0.3f))
             .dense(dt, { cout }, hfill(270, 0.5f))
             .idx({ 1, 4, 5, 10 })
             .dense(dt, { 10 * s, cout }, hfill(271), true)
             .pi("M", 10).pi("cin", cin).pi("cout", cout).pi("stride", s).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Lanes 2, 0, 1, 3 of four ten-row lanes, overlap 4, seven channels (a Philox block of
             * four straddles rows): FIRST | JOIN over a whole window; the same flags on an
             * unconditional branch, which moves nothing; JOIN alone over a window of 3 rows,
             * shorter than the overlap; FIRST alone over 6 rows. */
            static const uint32_t kSeeds[8] = { 0x9e3779b9u, 1u, 7u, 0u, 3u, 0xfffffffeu, 11u,
                                                0x80000001u };
            Hand h(("every flag combination, an unconditional entry, a short window" + sfx).c_str(),
                   "win_prep", 0.0);
            h.idx({ 17, 17, 16, 1 })
             .dense(RAD_F32, { 4 }, [](int64_t i) { return 0.7f - 0.13f * (float)i; })
             .raw(RAD_U32, { 8 }, kSeeds)
             .idx({ 0, 5, 2, 9 })
             .idx({ 2, 0, 1, 3 })
             .idx({ 1, -2, -1, -1 })
             .idx({ 0, 10, 20, 23, 29 })
             .dense(RAD_BF16, { 16, 7 }, hfill(272))
             .dense(RAD_BF16, { 16, 7 }, hfill(273), true)
             .dense(RAD_BF16, { 40, 7 }, hfill(274), true)
             .pi("M", 29).pi("n", 7).pi("stride", 10).pi("ov", 4).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* LAST | JOIN over a whole window; LAST over 7 rows (rows - 2 ov < 0, so the kept
             * overlap starts at row 0); LAST on an unconditional branch; JOIN without LAST. */
            Hand h(("LAST with and without JOIN, a short window, two skipped" + sfx).c_str(),
                   "win_finish", 0.0);
            h.idx({ 24, 8, 8, 16 })
             .idx({ -1, -1, -2, 0 })
             .idx({ 1, 3, 2, 0 })
             .idx({ 0, 10, 17, 20, 25 })
             .dense(dt, { 40, 5 }, hfill(275))
             .dense(RAD_BF16, { 40, 7 }, hfill(276), true)
             .dense(RAD_BF16, { 16, 7 }, hfill(277), true)
             .dense(dt, { 16, 5 }, hfill(278), true)
             .pi("M", 25).pi("n", 7).pi("n_cond", 5).pi("stride", 10).pi("ov", 4)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Lane 1 twice, the second an unconditional branch (zeros for its conditioning). */
            Hand h(("an unconditional entry beside its conditional one" + sfx).c_str(),
                   "win_input", 0.0);
            h.dense(dt, { 24, 9 }, hfill(279))
             .dense(dt, { 24, 5 }, hfill(280))
             .idx({ 0, 6, 12, 15 })
             .idx({ 1, 1, 0 })
             .idx({ 1, -2, -1 })
             .dense(dt, { 15, 16 }, hfill(281), true)
             .pi("M", 15).pi("n_lat", 9).pi("n_zero", 2).pi("n_cond", 5).pi("stride", 8)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Four lanes, two branch rows each: entry 0 at frame 0 (moves past the prompt only),
             * entry 1 on an ended lane, entries 2 and 3 stored. */
            Hand h(("frame 0, an ended lane, two stored" + sfx).c_str(), "frame_mix_store", 0.0);
            h.dense(dt, { 3 * 8, 37 }, hfill(282, 4.0f))
             .dense(RAD_F32, { 3 }, [](int64_t l) { return l == 0 ? 0.3f : l == 1 ? -1.2f : 2.0f; })
             .dense(RAD_F32, { 1 }, [](int64_t) { return 1.7f; })
             .idx({ 2, 0, 3, 1 })
             .idx({ 0, 3, 2, 5 })
             .idx({ -1, 2, 5, 7 })
             .dense(dt, { 20, 37 }, hfill(283), true)
             .pi("M", 4).pi("n_embd", 37).pi("L", 3).pi("stride", 5).pi("rep", 2)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Codes past either table: row 1's first code (its row is not written), row 2's
             * second (skipped in the sum). Code-major [3, 4]. */
            static const int32_t kCodes[12] = { 3, 10, 0, -2,   4, 1, -15, 7,   9, 2, 5, 0 };
            Hand h(("codes past either table" + sfx).c_str(), "rvq_embed", 0.0);
            h.raw(RAD_I32, { 3, 4 }, kCodes)
             .dense(RAD_BF16, { 12, 9 }, hfill(284))
             .dense(RAD_BF16, { 20, 9 }, hfill(285))
             .dense(dt, { 4, 9 }, hfill(286), true)
             .pi("M", 4).pi("n_embd", 9).pi("n_codes", 3).pi("vocab", 10).pi("off0", 2)
             .pf("scale", 0.75).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (int round : { 1, 0 }) {
            /* A trained projection's scale: angles to about 25 radians. */
            Hand h(("trained frequencies, round " + std::to_string(round) + sfx).c_str(),
                   "fourier_embed", 0.0);
            h.dense(RAD_F32, { 5 }, [](int64_t i) { return i == 4 ? 1.0f : 0.23f * (float)i; })
             .dense(RAD_F32, { 32 }, hfill(287, 4.0f))
             .dense(dt, { 5, 64 }, zero, true)
             .pi("M", 5).pi("n", 64).pi("round", round).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            Hand h(("ten rows down to four" + sfx).c_str(), "resample_nearest", 0.0);
            h.dense(dt, { 10, 7 }, hfill(288))
             .dense(dt, { 4, 7 }, zero, true)
             .pi("M", 4).pi("n", 7).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            Hand h(("three groups of input rows at a wider pitch" + sfx).c_str(), "chan_split",
                   0.0);
            h.view(dt, { 5, 21 }, { 25, 1 }, 125, hfill(289))
             .dense(dt, { 15, 7 }, zero, true)
             .pi("M", 5).pi("n", 7).pi("parts", 3).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (int lg : { 0, 1 }) {
            /* SnakeBeta over x in [-8, 8]: alpha and beta as a codec stores them (logarithms in
             * [-1, 1], `log` 1) or as positive values. */
            const int64_t n = 37;
            const std::function<float(int64_t)> al =
                lg ? hfill(291) : std::function<float(int64_t)>([](int64_t c) {
                    return 0.3f + 0.05f * (float)c;
                });
            const std::function<float(int64_t)> be =
                lg ? hfill(292) : std::function<float(int64_t)>([](int64_t c) {
                    return 0.2f + 0.03f * (float)c;
                });
            Hand h(("SnakeBeta, alpha and beta as " + std::string(lg ? "logarithms" : "values") +
                    sfx).c_str(), "snake", norm_tol);
            h.dense(dt, { 4, n }, hfill(293, 8.0f))
             .dense(RAD_F32, { n }, al)
             .dense(dt, { 4, n }, zero, true)
             .dense(RAD_F32, { n }, be)
             .pi("M", 4).pi("n", n).pi("log", lg).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* Two segments from row 1, with a bias; row 0 keeps its bytes. */
            const int64_t C = 21;
            Hand h(("two segments from row 1, with a bias" + sfx).c_str(), "conv1d_dw", norm_tol);
            h.dense(dt, { 12, C }, hfill(294))
             .dense(dt, { C, 1, 5 }, hfill(295, 0.4f))
             .idx({ 1, 6, 12 })
             .dense(dt, { 12, C }, zero, true)
             .dense(dt, { C }, hfill(296, 0.5f))
             .pi("M", 12).pi("n", C).pi("k", 5).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (int dil : { 1, 3 }) {
            /* Causal with an even kernel (a codec's), two sequences, a bias. */
            const int64_t cin = 20, cout = 13;
            Hand h(("causal, k 4, dilation " + std::to_string(dil) + sfx).c_str(), "conv1d",
                   norm_tol);
            h.dense(dt, { 16, cin }, hfill(296 + dil))
             .dense(dt, { cout, cin, 4 }, hfill(300, 0.3f))
             .idx({ 0, 7, 16 })
             .none()
             .dense(dt, { 16, cout }, zero, true)
             .dense(dt, { cout }, hfill(301, 0.5f))
             .pi("M", 16).pi("cin", cin).pi("cout", cout).pi("k", 4).pi("dil", dil)
             .pi("causal", 1).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const int64_t s : { 5, 3 }) {
            /* A codec's odd strides at kernel 2 stride, causal, over sequences of 3, 1 and 5 rows
             * from row 1. */
            const int64_t cin = 12, cout = 7;
            Hand h(("odd stride " + std::to_string(s) + ", causal" + sfx).c_str(), "conv_t1d",
                   norm_tol);
            h.dense(dt, { 10, cin }, hfill(314 + s))
             .dense(dt, { cin, cout, 2 * s }, hfill(320, 0.3f))
             .dense(dt, { cout }, hfill(321, 0.5f))
             .idx({ 1, 4, 5, 10 })
             .dense(dt, { 10 * s, cout }, hfill(322), true)
             .pi("M", 10).pi("cin", cin).pi("cout", cout).pi("stride", s).pi("pad", 0)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const int64_t s : { 5, 2 }) {
            /* BigVGAN's upsamplers (kernel 9 at stride 5, pad 2; kernel 12 at stride 2, its
             * default pad 5 past a stride) over sequences of 3, 1 and 5 rows from row 1. */
            const int64_t cin = 12, cout = 7, K = s == 5 ? 9 : 12;
            Hand h(("kernel " + std::to_string(K) + " at stride " + std::to_string(s) +
                    ", three uneven sequences" + sfx).c_str(), "conv_t1d", norm_tol);
            h.dense(dt, { 10, cin }, hfill(330 + s))
             .dense(dt, { cin, cout, K }, hfill(336, 0.3f))
             .dense(dt, { cout }, hfill(337, 0.5f))
             .idx({ 1, 4, 5, 10 })
             .dense(dt, { 10 * s, cout }, hfill(338), true)
             .pi("M", 10).pi("cin", cin).pi("cout", cout).pi("stride", s).pi("k", K)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        for (const int64_t k : { 12, 7 }) {
            /* The alias-free SnakeBeta over sequences of 5, 1 and 9 rows from row 1 (row 0 and
             * rows past 16 keep their bytes), at a trained scale: alpha and beta logarithms,
             * x in [-4, 4], a smooth low-pass pair of filters. */
            const int64_t n = 21;
            const auto lowpass = [k](int64_t j) {
                const float t = ((float)j + 0.5f) / (float)k - 0.5f;
                return (1.0f - 4.0f * t * t) / (float)k * 1.5f;
            };
            Hand h(("three uneven sequences, filters of " + std::to_string(k) + sfx).c_str(),
                   "snake_aa", norm_tol);
            h.dense(dt, { 18, n }, hfill(340, 4.0f))
             .dense(RAD_F32, { n }, hfill(341, 0.8f))
             .dense(RAD_F32, { n }, hfill(342, 0.8f))
             .dense(RAD_F32, { 1, 1, k }, lowpass)
             .dense(RAD_F32, { 1, 1, k }, lowpass)
             .idx({ 1, 6, 7, 16 })
             .dense(dt, { 18, n }, hfill(343), true)
             .pi("M", 18).pi("n", n).pi("ratio", 2).pi("k", k).pi("log", 1).ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* A kernel of one stride at an odd stride: each output row reads one input row. */
            const int64_t cin = 12, cout = 7, s = 3;
            Hand h(("kernel 3 at stride 3" + sfx).c_str(), "conv_t1d", norm_tol);
            h.dense(dt, { 10, cin }, hfill(323))
             .dense(dt, { cin, cout, s }, hfill(324, 0.3f))
             .dense(dt, { cout }, hfill(325, 0.5f))
             .idx({ 1, 4, 5, 10 })
             .dense(dt, { 10 * s, cout }, hfill(326), true)
             .pi("M", 10).pi("cin", cin).pi("cout", cout).pi("stride", s).pi("k", s)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* The causal transposed form over sequences of 3, 1 and 5 rows from row 1. */
            const int64_t cin = 12, cout = 7, s = 4;
            Hand h(("the causal form over three uneven sequences" + sfx).c_str(), "conv_t1d",
                   norm_tol);
            h.dense(dt, { 10, cin }, hfill(302))
             .dense(dt, { cin, cout, 2 * s }, hfill(303, 0.3f))
             .dense(dt, { cout }, hfill(304, 0.5f))
             .idx({ 1, 4, 5, 10 })
             .dense(dt, { 10 * s, cout }, hfill(305), true)
             .pi("M", 10).pi("cin", cin).pi("cout", cout).pi("stride", s).pi("pad", 0)
             .ps("dtype", dn);
            hs.push_back(std::move(h));
        }
        {
            /* One rounding of the whole sum, with codes past either table (row 1's first code
             * leaves its row unwritten, row 2's second adds nothing). Code-major [3, 4]. */
            static const int32_t kCodes1[12] = { 3, 10, 0, -2,   4, 1, -15, 7,   9, 2, 5, 0 };
            Hand h(("one rounding, codes past either table" + sfx).c_str(), "rvq_embed", 0.0);
            h.raw(RAD_I32, { 3, 4 }, kCodes1)
             .dense(RAD_BF16, { 12, 9 }, hfill(306))
             .dense(RAD_BF16, { 20, 9 }, hfill(307))
             .dense(dt, { 4, 9 }, hfill(308), true)
             .pi("M", 4).pi("n_embd", 9).pi("n_codes", 3).pi("vocab", 10).pi("off0", 2)
             .pf("scale", 0.75).pi("round", 1).ps("dtype", dn);
            hs.push_back(std::move(h));

            /* The codes as a block of a wider plane (a lane's history over a run of frames): rows
             * at pitch 11, read at the operand's strides. */
            Hand g(("one rounding, codes a block of a wider plane" + sfx).c_str(), "rvq_embed",
                   0.0);
            g.view(RAD_I32, { 3, 5 }, { 11, 1 }, 33, [](int64_t i) { return (float)(i % 9); })
             .dense(RAD_BF16, { 12, 9 }, hfill(309))
             .dense(RAD_BF16, { 20, 9 }, hfill(310))
             .dense(dt, { 5, 9 }, zero, true)
             .pi("M", 5).pi("n_embd", 9).pi("n_codes", 3).pi("vocab", 10).pi("off0", 2)
             .pf("scale", 1.0).pi("round", 1).ps("dtype", dn);
            hs.push_back(std::move(g));
        }
    }
    {
        /* Logits on a grid of 0.05 over [-1, 1]: ties throughout, at the k-th place and at the
         * conditional's cond_k-th. Three lanes, two branches each. */
        static const uint32_t kSeeds[6] = { 5u, 0u, 0xdeadbeefu, 7u, 1u, 1u };
        const auto grid = [](int64_t i) { return std::round(hfill(290)(i) * 20.0f) / 20.0f; };
        Hand h("ties at the cut and in the guided order", "cfg_topk_sample", 0.0);
        h.dense(RAD_F32, { 6, 40 }, grid)
         .raw(RAD_U32, { 6 }, kSeeds)
         .idx({ 0, 17, 123456 })
         .dense(RAD_I32, { 6 }, zero, true)
         .pi("M", 6).pi("n_vocab", 40).pi("k", 12).pi("rep", 2).pi("cond_k", 9).pf("cfg", 1.5)
         .pi("stream", 2).ps("dtype", "f32");
        hs.push_back(std::move(h));

        Hand g("one branch, every entry a candidate, a temperature", "cfg_topk_sample", 0.0);
        g.dense(RAD_F32, { 3, 40 }, grid)
         .raw(RAD_U32, { 6 }, kSeeds)
         .idx({ 4, 4, 9 })
         .dense(RAD_I32, { 3 }, zero, true)
         .pi("M", 3).pi("n_vocab", 40).pi("k", 40).pi("rep", 1).pf("cfg", 1.0).pf("temp", 0.6)
         .pi("stream", 0).ps("dtype", "f32");
        hs.push_back(std::move(g));
    }
    {
        /* A speech model's frame: three lanes with their own guidance, each lane's history of
         * earlier picks penalised (lane 2 at frame 0, its row cleared first), reserved ids
         * suppressed; the picks recorded in the history. */
        static const uint32_t kSeeds[6] = { 5u, 0u, 0xdeadbeefu, 7u, 1u, 1u };
        Hand h("guidance a lane, a history, reserved ids", "cfg_topk_sample", 0.0);
        h.dense(RAD_F32, { 6, 40 }, hfill(311, 3.0f))
         .raw(RAD_U32, { 6 }, kSeeds)
         .idx({ 4, 0, 9 })
         .dense(RAD_I32, { 6 }, zero, true)
         .dense(RAD_F32, { 3 }, [](int64_t i) { return 1.0f + 1.5f * (float)i; })
         .idx({ 1, 2, 3 })
         .dense(RAD_I32, { 4, 40 }, [](int64_t i) { return (float)((i * 7) % 3 == 0); }, true)
         .pi("M", 6).pi("n_vocab", 40).pi("k", 12).pi("rep", 2).pf("cfg", 1.0).pf("temp", 0.9)
         .pi("stream", 0).pi("sup_lo", 36).pi("sup_hi", 39).pf("penalty", 1.1)
         .ps("dtype", "f32");
        hs.push_back(std::move(h));

        /* One lane twice, in order: logits 4 - 0.5 j at temperature 0.01, so each entry takes its
         * best candidate; the first takes id 0, which the second sees penalised (4 / 1.5 falls
         * below id 1's 3.5). */
        static const uint32_t kSeeds2[4] = { 9u, 0u, 9u, 0u };
        Hand g("one lane twice: the second entry sees the first's pick", "cfg_topk_sample", 0.0);
        g.dense(RAD_F32, { 2, 40 }, [](int64_t i) { return 4.0f - 0.5f * (float)(i % 40); })
         .raw(RAD_U32, { 4 }, kSeeds2)
         .idx({ 3, 4 })
         .dense(RAD_I32, { 2 }, zero, true)
         .none()
         .idx({ 1, 1 })
         .dense(RAD_I32, { 2, 40 }, zero, true)
         .pi("M", 2).pi("n_vocab", 40).pi("k", 3).pi("rep", 1).pf("cfg", 1.0).pf("temp", 0.01)
         .pi("stream", 0).pf("penalty", 1.5).ps("dtype", "f32")
         .expect(3, { 0, 1 });
        hs.push_back(std::move(g));

        /* Refusals before anything is written: a lane past the history, every id suppressed, and
         * the conditional's best ids all suppressed. */
        Hand r1("a lane past the history", "cfg_topk_sample", 0.0);
        r1.dense(RAD_F32, { 2, 40 }, hfill(312))
          .raw(RAD_U32, { 2 }, kSeeds)
          .idx({ 0 })
          .dense(RAD_I32, { 2 }, zero, true)
          .none()
          .idx({ 5 })
          .dense(RAD_I32, { 4, 40 }, zero, true)
          .pi("M", 2).pi("n_vocab", 40).pi("k", 5).pi("rep", 2).pf("cfg", 2.0)
          .pi("stream", 0).pf("penalty", 1.1).ps("dtype", "f32")
          .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(r1));
        Hand r2("every id suppressed", "cfg_topk_sample", 0.0);
        r2.dense(RAD_F32, { 1, 40 }, hfill(313))
          .raw(RAD_U32, { 2 }, kSeeds)
          .idx({ 2 })
          .dense(RAD_I32, { 1 }, zero, true)
          .pi("M", 1).pi("n_vocab", 40).pi("k", 5).pi("rep", 1).pf("cfg", 1.0)
          .pi("stream", 0).pi("sup_lo", 0).pi("sup_hi", 40).ps("dtype", "f32")
          .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(r2));
        Hand r3("the conditional's best ids all suppressed", "cfg_topk_sample", 0.0);
        r3.dense(RAD_F32, { 2, 40 }, [](int64_t i) {
              const int64_t j = i % 40;
              return j >= 10 && j < 15 ? 5.0f : -0.01f * (float)j;
          })
          .raw(RAD_U32, { 2 }, kSeeds)
          .idx({ 2 })
          .dense(RAD_I32, { 2 }, zero, true)
          .pi("M", 2).pi("n_vocab", 40).pi("k", 5).pi("rep", 2).pi("cond_k", 5).pf("cfg", 1.5)
          .pi("stream", 0).pi("sup_lo", 10).pi("sup_hi", 20).ps("dtype", "f32")
          .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(r3));
    }
    {
        /* Four entries, codes a block of a wider plane (pitch 11, each lane's two branch columns):
         * entry 0 at frame 0 under skip 1 (moves past the prompt), entry 3 on an ended lane,
         * entries 1 and 2 stored. */
        Hand h("frame 0 under skip, an ended lane, codes a block", "frame_store", 0.0);
        h.view(RAD_I32, { 3, 8 }, { 11, 1 }, 33, [](int64_t i) { return (float)(100 + i); })
         .idx({ 2, 0, 3, 1 })
         .idx({ 0, 3, 2, 5 })
         .idx({ 1, -1, 4, 2 })
         .dense(RAD_I32, { 12, 6 }, [](int64_t) { return -7.0f; }, true)
         .pi("M", 4).pi("n_codes", 3).pi("stride", 6).pi("rep", 2).pi("skip", 1);
        hs.push_back(std::move(h));

        /* Refusals before anything is written: a frame past its lane's row, a lane past the
         * status. */
        Hand r1("a frame past its lane's row", "frame_store", 0.0);
        r1.idx({ 1, 2, 3, 4, 5, 6 })
          .idx({ 0, 1 })
          .idx({ 7, 2 })
          .idx({ 0, 0 })
          .dense(RAD_I32, { 6, 6 }, [](int64_t) { return -7.0f; }, true)
          .pi("M", 2).pi("n_codes", 3).pi("stride", 6).pi("rep", 1).pi("skip", 1)
          .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(r1));
        Hand r2("a lane past the status", "frame_store", 0.0);
        r2.idx({ 1, 2, 3 })
          .idx({ 4 })
          .idx({ 2 })
          .idx({ 0, 0, 0, 0 })
          .dense(RAD_I32, { 12, 6 }, [](int64_t) { return -7.0f; }, true)
          .pi("M", 1).pi("n_codes", 3).pi("stride", 6).pi("rep", 1).pi("skip", 0)
          .refuses(RAD_E_SHAPE);
        hs.push_back(std::move(r2));
    }
    {
        /* Lane 2 ends (its pick is `end`), lanes 0 and 1 continue, lane 3 is not in the frame. */
        Hand h("an ending lane among continuing ones", "end_status", 0.0);
        h.idx({ 5, 5, 3, 3, 5, 5 })
         .idx({ 2, 0, 1 })
         .idx({ 7, 7, 7, 7 }, true)
         .pi("M", 3).pi("end", 5).pi("rep", 2)
         .expect(2, { 3, -1, -1, 7 });
        hs.push_back(std::move(h));
    }
}

}  // namespace

bool hand_cases(const Check& ck) {
    std::vector<Hand> hs;
    shape_cases(hs);
    format_cases(hs);
    anchor_cases(hs);
    packed_cases(hs);
    gen_cases(hs);
    video_cases(hs);
    upscaler_cases(hs);
    vsa_cases(hs);
    qsa_cases(hs);
    encoder_cases(hs);
    audio_cases(hs);
    Broken broken;
    planes_cases(ck, hs, broken);

    bool ok = true;
    int run = 0, fails = 0, absent = 0;
    for (Hand& h : hs) {
        if (!ck.only_op.empty() && ck.only_op != h.op) continue;
        ++run;
        switch (h.run(ck)) {
            case Hand::PASS:   break;
            case Hand::FAIL:   ok = false; ++fails; break;
            case Hand::ABSENT: ++absent; break;
        }
    }
    if (ck.only_op.empty() || ck.only_op == "gdn_recurrent_update") {
        /* libr4d's layout over 3-row steps, and a 16 x 16 head up to its 7-token capacity. */
        static const int kSched64[4] = { 1, 2, 3, 1 }, kSched16[4] = { 1, 7, 4, 2 };
        for (const bool small : { false, true }) {
            ++run;
            const bool good = small ? anchor_replays_per_candidate(ck, 16, 16, 7, kSched16)
                                    : anchor_replays_per_candidate(ck, 64, 64, 3, kSched64);
            if (!good) { ok = false; ++fails; }
        }
    }
    if (const int cf = codebook_cases(ck, &run)) {
        ok = false;
        fails += cf;
    }
    if (ck.only_op.empty() || ck.only_op == "gemm_nt_q" || ck.only_op == "logits_gemm") {
        int lay = 0;
        if (!planes_refusals(ck, &lay)) { ok = false; ++fails; }
        run += lay > 0;
    }
    for (const std::string& w : broken.why) {
        std::printf("  FAIL a joined weight could not be built: %s\n", w.c_str());
        ok = false;
        ++fails;
        ++run;
    }
#ifndef AVX_CHECK_CORE_DECODER
    std::printf("  joined weights: no core decoder in this build; libavx is compared with libref "
                "only\n");
#endif
    if (!run) return true;
    std::printf("  hand-built: %d checked, %d failed", run - absent, fails);
    if (absent) std::printf(", %d for ops %s does not serve", absent, ck.lib->info->name);
    std::printf("\n");
    return ok;
}
