/* tests/dflash_tree_gpu_test.cpp -- DEVICE-vs-HOST byte comparison for `dflash_tree_select`.
 *
 * The device row (`mine_dflash_tree_select`) and the host oracle (`ref_dflash_tree_select`) are two
 * independent implementations of one op, and the output is a discrete structure: identical bytes or
 * a disagreement. So the comparator is a memcmp, with no tolerance anywhere.
 *
 * WHAT IS COMPARED. Both sides start from THE SAME HOST IMAGE. The oracle runs on that image, the
 * device row runs on freshly uploaded device copies of it, and the two output planes are compared
 * byte for byte. The device row's output buffer is framed in a sentinel pattern and every byte the
 * op may not write is checked against it afterwards -- so a kernel that scribbles past the row, or
 * into a column past `count`, fails.
 *
 * EACH CASE IS A PROPERTY THE OP IS SUPPOSED TO HAVE, not just a random draw:
 *
 *   plain            three levels, four candidates, width 2 -- the ordinary tree
 *   hp-nontrivial    a dense, sign-changing hidden projection at R = 8
 *   hp-absent        the bigram form (no hidden projection at all)
 *   ties             every candidate of a level scores identically; the tie rule must be the same
 *                    on both sides, which is exactly what a device-side argmax gets wrong
 *   invalid-ids      candidates of -1 and of n_vocab, with the highest unary terms
 *   invalid-anchor   count must be 0, on both sides
 *   budget-1         the tightest budget: only one draft node survives
 *   budget-15        the largest tree the verifier can hold, three levels, width 16
 *   beta-negative    a depth penalty that prefers shallow nodes
 *   penalty-overflow finite beta overflows at depth 2; -inf ties still retain ancestors first
 *   multi-sequence   four sequences with different anchors in one launch
 *   reduction-order  a scene where the lane fold and an ascending sum rank two candidates
 *                    oppositely, so the device's summation order is actually exercised
 *   duplicate-ids    repeated candidate ids with differing unary terms, so the per-parent dedupe
 *                    and its "keep the best-scoring instance" rule are exercised
 *   anchor-stride    the pass's token_ids plane, stride 4
 *   odd-rank-frontier rank 65, seven candidates and width 5 across strided sequence anchors;
 *                    the third element of lane 0 dominates, and hp reverses its ordering by row
 *
 * NO SKIP IS A PASS. A missing device, a missing row, a refused launch or a zero-case run returns
 * nonzero. This must not be run while training owns the cards; compiling is free.
 *
 * AND A PRODUCTION CHECK, because an oracle and a kernel that agree may both be wrong: the last
 * case pins libr4d's `dflash_select` row BY NAME, runs the shipping kernel on the same bytes, and
 * requires the tree's width-1 chain to be its token sequence. R is a multiple of 32 there, which is
 * what that row accepts.
 *
 * ONE THING THIS COMPARATOR CANNOT ADJUDICATE, stated rather than hidden: at a score that lands
 * within an ulp of a tie the two sides can order two candidates differently. The tie cases below
 * use EXACTLY equal bytes so that this cannot happen; everywhere else the comparison is byte
 * equality. A case deliberately constructed inside the ulp band would have to be reported as
 * inconclusive rather than as a pass or a failure.
 */
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"      /* rad_f32_to_bf16 and rad_bf16_to_f32: the ABI's own narrowing */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/* THE ROW IS 1 + 2*capacity INTS. The engine declares the drafter tree with 16 node slots, so a
 * row is 33 columns: count, 16 tokens, then the anchor's -1 sentinel and 15 draft parents.
 * The parent column of node i is `1 + kCap + i`, which is what makes the two blocks line up for
 * any capacity rather than only for 16. */
constexpr int kCap = 16;
constexpr int kCols = 1 + 2 * kCap;             /* 33 */
constexpr size_t kPad = 64;
constexpr uint8_t kSentinel = 0xa5;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }
void require(bool ok, const std::string& what) { if (!ok) fail(what); }
void check(int status, const std::string& what) {
    if (status == RAD_OK) return;
    const char* detail = rad_dev_last_error();
    fail(what + ": " + rad_strerror(status) + "; " + (detail ? detail : ""));
}

uint16_t to_bf(float v) { return rad_f32_to_bf16(v); }

/* One case: the host image of every operand plus the parameter list. */
struct Case {
    std::string name;
    int64_t M = 1, steps = 3, K = 4, R = 3, V = 16, astr = 1, budget = 15, width = 2;
    double beta = 0.0;
    bool hp = true;
    std::vector<int32_t> cand;
    std::vector<float>   unary;
    std::vector<uint16_t> hpv;
    std::vector<int32_t> anchor;
    std::vector<uint16_t> pred, succ;
    std::vector<int32_t> expect;   /* the oracle's answer, filled before upload */

    int32_t& C(int64_t m, int64_t l, int64_t j) { return cand[size_t((m * steps + l) * K + j)]; }
    float&   U(int64_t m, int64_t l, int64_t j) { return unary[size_t((m * steps + l) * K + j)]; }

    void alloc() {
        cand.assign(size_t(M * steps * K), 0);
        unary.assign(size_t(M * steps * K), 0.0f);
        if (hp) hpv.assign(size_t(M * steps * R), to_bf(1.0f));
        else hpv.clear();
        anchor.assign(size_t(M * astr), 0);
        pred.assign(size_t(V * R), 0);
        succ.assign(size_t(V * R), 0);
        expect.assign(size_t(M * kCols), kSentinel);
    }

    std::vector<RadParam> params() const {
        return { RAD_INT("M", M), RAD_INT("steps", steps), RAD_INT("top_k", K),
                 RAD_INT("rank", R), RAD_INT("n_vocab", V), RAD_INT("anchor_stride", astr),
                 RAD_INT("tree_budget", budget), RAD_INT("tree_width", width),
                 RAD_F64("depth_penalty", beta), RAD_STR("dtype", "bf16") };
    }
};

/* Deterministic noise, narrowed so no side ever sees a NaN or a denormal it could resolve
 * differently. */
uint32_t g_rng = 0x2468ace1u;
float rnd() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return float((g_rng >> 9) & 0x3fff) / 16384.0f - 0.5f;
}

void fill_codebooks(Case& c) {
    for (int64_t v = 0; v < c.V; ++v)
        for (int64_t r = 0; r < c.R; ++r) {
            c.pred[size_t(v * c.R + r)] = to_bf(rnd());
            c.succ[size_t(v * c.R + r)] = to_bf(rnd());
        }
}

void fill_candidates(Case& c) {
    for (int64_t m = 0; m < c.M; ++m)
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t j = 0; j < c.K; ++j) {
                c.C(m, l, j) = int32_t((m * 5 + l * 3 + j * 2 + 1) % c.V);
                c.U(m, l, j) = rnd();
            }
}

struct Session {
    RadStream stream = nullptr;
    Session() {
        check(rad_dev_set(0), "select GPU 0");
        RadDeviceProps props{};
        check(rad_dev_props(0, &props), "GPU properties");
        require(!props.is_host_backend,
                "dflash_tree_select parity needs a real device; a host backend is not a pass");
        check(rad_stream_create(&stream, 0), "create comparator stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        if (bytes) data = rad_dev_alloc(int64_t(bytes), RAD_MEM_DEVICE);
        require(bytes == 0 || data != nullptr, "device allocation failed");
    }
    ~DeviceBuffer() { if (data) rad_dev_free(data, RAD_MEM_DEVICE); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/* One framed device buffer per operand. The output's frame is sentinels; every other buffer is
 * upload-only, and a kernel that writes one of them is caught by the candidate-vs-image check. */
struct Side {
    std::vector<std::unique_ptr<DeviceBuffer>> buf;

    void upload(const Session& s, const Case& c) {
        auto add = [&](const void* src, size_t bytes) {
            const size_t total = kPad + bytes + kPad;
            buf.emplace_back(new DeviceBuffer(total));
            std::vector<uint8_t> framed(total, kSentinel);
            if (src && bytes) std::memcpy(framed.data() + kPad, src, bytes);
            check(rad_memcpy_async(buf.back()->data, framed.data(), int64_t(total), s.stream),
                  "upload operand");
        };
        add(c.cand.data(), c.cand.size() * 4);
        add(c.unary.data(), c.unary.size() * 4);
        /* An absent optional operand is a null `data`, never a shorter list: allocate nothing and
         * bind a null pointer below. */
        if (!c.hpv.empty()) add(c.hpv.data(), c.hpv.size() * 2);
        add(c.anchor.data(), c.anchor.size() * 4);
        add(c.pred.data(), c.pred.size() * 2);
        add(c.succ.data(), c.succ.size() * 2);
        /* The output's payload is filled with the sentinel too. The op writes all 33 columns of
         * every row, so the oracle comparison below is what catches a column left alone; the frame
         * check then catches a write past the row. */
        add(nullptr, size_t(c.M * kCols) * 4);
        check(rad_stream_sync(s.stream), "finish upload");
    }

    static void* at(const DeviceBuffer& b) {
        return static_cast<uint8_t*>(b.data) + kPad;
    }

    std::vector<RadTensor> bind(const Case& c) const {
        std::vector<RadTensor> t(7);
        size_t i = 0;
        auto shape2 = [](RadTensor& x, uint32_t dt, int64_t a, int64_t b) {
            x.dtype = dt; x.rank = 2; x.shape[0] = a; x.shape[1] = b;
            x.stride[0] = b; x.stride[1] = 1;
        };
        shape2(t[0], RAD_I32, c.M * c.steps, c.K);
        t[0].data = at(*buf[i++]);
        shape2(t[1], RAD_F32, c.M * c.steps, c.K);
        t[1].data = at(*buf[i++]);
        shape2(t[2], RAD_BF16, c.M * c.steps, c.R);
        t[2].data = c.hpv.empty() ? nullptr : at(*buf[i++]);
        t[3].dtype = RAD_I32; t[3].rank = 1; t[3].shape[0] = c.M * c.astr; t[3].stride[0] = 1;
        t[3].data = at(*buf[i++]);
        shape2(t[4], RAD_BF16, c.V, c.R);
        t[4].data = at(*buf[i++]);
        shape2(t[5], RAD_BF16, c.V, c.R);
        t[5].data = at(*buf[i++]);
        shape2(t[6], RAD_I32, c.M, kCols);
        t[6].data = at(*buf[i++]);
        return t;
    }
};

rad::Geometry geometry_of(const Case& c) {
    rad::Geometry g;
    const std::vector<RadParam> p = c.params();
    for (const auto& e : p) {
        if (e.kind == RAD_P_INT) g.set_i(e.key, e.ival);
        else if (e.kind == RAD_P_F64) g.set_f(e.key, e.dval);
        else if (e.kind == RAD_P_STR) g.set_s(e.key, e.sval);
        else fail("an unresolved test parameter reached the geometry");
    }
    return g;
}

/* The oracle runs FIRST, on host memory, and its bytes are the expectation. It is resolved through
 * the registry's HOST domain rather than called by symbol: this binary loads the plugins as .so
 * files and links none of them, so a direct call would not link at all -- and a resolved host row
 * is also the identity check the comparison wants (it must be mine_ref). */
void run_oracle(rad::Registry& reg, Case& c) {
    RadTensor t[7]{};
    auto shape2 = [](RadTensor& x, uint32_t dt, int64_t a, int64_t b) {
        x.dtype = dt; x.rank = 2; x.shape[0] = a; x.shape[1] = b;
        x.stride[0] = b; x.stride[1] = 1;
    };
    shape2(t[0], RAD_I32, c.M * c.steps, c.K); t[0].data = c.cand.data();
    shape2(t[1], RAD_F32, c.M * c.steps, c.K); t[1].data = c.unary.data();
    shape2(t[2], RAD_BF16, c.M * c.steps, c.R); t[2].data = c.hpv.empty() ? nullptr : c.hpv.data();
    t[3].dtype = RAD_I32; t[3].rank = 1; t[3].shape[0] = c.M * c.astr; t[3].stride[0] = 1;
    t[3].data = c.anchor.data();
    shape2(t[4], RAD_BF16, c.V, c.R); t[4].data = c.pred.data();
    shape2(t[5], RAD_BF16, c.V, c.R); t[5].data = c.succ.data();
    shape2(t[6], RAD_I32, c.M, kCols); t[6].data = c.expect.data();

    const rad::Geometry g = geometry_of(c);
    rad::SelectionTrace why;
    rad::Resolved host = reg.resolve("dflash_tree_select", g, RAD_DOMAIN_HOST, &why);
    require(host.row != nullptr, "no host oracle for dflash_tree_select: " + why.reason);
    require(host.row->plugin == "mine_ref", "host row came from plugin '" + host.row->plugin + "'");
    check(reg.init_instance(host, 0, 1), "init host instance");

    RadArgs a{};
    a.t = t;
    a.n_t = 7;
    a.p = host.geom.params();
    a.n_p = host.geom.n_params();
    a.instance = host.instance;
    a.rank = 0;
    a.world_size = 1;
    check(host.row->info->launch(&a, nullptr), "host oracle " + c.name);
}

size_t run_case(rad::Registry& reg, const Session& s, Case& c) {
    run_oracle(reg, c);

    const rad::Geometry g = geometry_of(c);
    std::string error;
    check(reg.validate("dflash_tree_select", g.params(), g.n_params(), &error),
          "schema: " + c.name + ": " + error);

    rad::SelectionTrace why;
    rad::Resolved dev = reg.resolve("dflash_tree_select", g, RAD_DOMAIN_DEVICE, &why);
    require(dev.row != nullptr, "no device row for dflash_tree_select: " + why.reason);
    require(dev.row->plugin == "mine", "device row came from plugin '" + dev.row->plugin + "'");
    require(dev.row->info && dev.row->info->name &&
            std::strcmp(dev.row->info->name, "mine_dflash_tree_select") == 0,
            "device row is not mine_dflash_tree_select");
    check(reg.init_instance(dev, 0, 1), "init device instance");

    Side side;
    side.upload(s, c);
    std::vector<RadTensor> t = side.bind(c);

    RadArgs a{};
    a.t = t.data();
    a.n_t = int(t.size());
    a.p = dev.geom.params();
    a.n_p = dev.geom.n_params();
    a.instance = dev.instance;
    a.rank = 0;
    a.world_size = 1;
    if (dev.row->info->scratch) {
        const int64_t bytes = dev.row->info->scratch(&a);
        require(bytes == 0, "dflash_tree_select declares scratch this comparator cannot supply");
    }
    check(dev.row->info->launch(&a, s.stream), "device launch " + c.name);

    std::vector<uint8_t> got(kPad + size_t(c.M * kCols) * 4 + kPad);
    check(rad_memcpy_async(got.data(), side.buf.back()->data, int64_t(got.size()), s.stream),
          "download output");
    check(rad_stream_sync(s.stream), "finish download");

    const int32_t* ref = c.expect.data();
    const int32_t* out = reinterpret_cast<const int32_t*>(got.data() + kPad);
    for (int64_t m = 0; m < c.M; ++m)
        for (int i = 0; i < kCols; ++i) {
            const int64_t at = m * kCols + i;
            if (ref[at] == out[at]) continue;
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "%s: mismatch at row %lld col %d: device=%d host=%d", c.name.c_str(),
                          (long long)m, i, out[at], ref[at]);
            fail(msg);
        }

    /* THE FRAME. Every byte the op may not write still holds the sentinel. */
    for (size_t i = 0; i < got.size(); ++i) {
        const bool inside = i >= kPad && i < got.size() - kPad;
        if (inside || got[i] == kSentinel) continue;
        char msg[256];
        std::snprintf(msg, sizeof msg, "%s: wrote frame byte %zu (0x%02x)", c.name.c_str(), i,
                      got[i]);
        fail(msg);
    }

    /* THE VALIDATOR'S OWN CHECKS, reproduced here so a producer `dloop_tree_from_packed` would
     * REJECT fails in this comparator too, naming the rule rather than a byte offset. The row is
     * capacity-generic: [0] count, [1+i] node i's token, [1+kCap+i] node i's parent (0 = anchor),
     * and [1+kCap] == -1 is the anchor's sentinel. At kCap = 16 that is column 17.
     *
     * COUNT 0 IS NOT A VALIDATOR REJECTION HERE, it is this op's own "this row produced no tree"
     * answer (an anchor outside the vocabulary, a non-finite term on a live candidate). The engine
     * never hands such a row to the validator -- it reads count to decide whether a row holds a
     * tree at all -- so the checks below apply only to rows that do, and a zero row is required to
     * be otherwise untouched (-1 everywhere, from the fill). */
    for (int64_t m = 0; m < c.M; ++m) {
        const int32_t* r = out + m * kCols;
        if (r[0] == 0) {
            for (int i = 1; i < kCols; ++i)
                if (r[i] != -1) {
                    char msg[192];
                    std::snprintf(msg, sizeof msg,
                                  "%s: row %lld has count 0 but column %d is %d, not -1",
                                  c.name.c_str(), (long long)m, i, r[i]);
                    fail(msg);
                }
            continue;
        }
        if (r[0] < 1 || r[0] > kCap) {
            char msg[192];
            std::snprintf(msg, sizeof msg, "%s: row %lld has count %d", c.name.c_str(),
                          (long long)m, r[0]);
            fail(msg);
        }
        if (r[1 + kCap] != -1) {
            char msg[192];
            std::snprintf(msg, sizeof msg, "%s: row %lld's root parent is %d, not the -1 sentinel",
                          c.name.c_str(), (long long)m, r[1 + kCap]);
            fail(msg);
        }
        for (int i = 1; i < r[0]; ++i) {
            const int parent = r[1 + kCap + i];
            if (parent < 0 || parent >= i) {
                char msg[256];
                std::snprintf(msg, sizeof msg,
                              "%s: row %lld node %d has parent %d, which is not an earlier "
                              "root-inclusive node", c.name.c_str(), (long long)m, i, parent);
                fail(msg);
            }
        }
        for (int j = 1; j < r[0]; ++j)
            for (int i = 1; i < j; ++i)
                if (r[1 + kCap + i] == r[1 + kCap + j] && r[1 + i] == r[1 + j]) {
                    char msg[256];
                    std::snprintf(msg, sizeof msg,
                                  "%s: row %lld nodes %d and %d are siblings proposing token %d",
                                  c.name.c_str(), (long long)m, i, j, r[1 + i]);
                    fail(msg);
                }
    }
    std::printf("TREE_PASS %-16s nodes=%d bytes=%zu\n", c.name.c_str(), ref[0],
                size_t(c.M * kCols) * 4);
    return size_t(c.M * kCols) * 4;
}

std::vector<Case> cases() {
    std::vector<Case> v;

    auto base = [&](const char* name) {
        Case c;
        c.name = name;
        v.push_back(c);
    };

    base("plain"); {
        Case& c = v.back();
        c.steps = 3; c.K = 4; c.R = 3; c.V = 16; c.width = 2; c.budget = 15;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 1;
    }
    base("hp-nontrivial"); {
        Case& c = v.back();
        c.steps = 3; c.K = 8; c.R = 8; c.V = 24; c.width = 3; c.budget = 12;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 7;
        for (auto& x : c.hpv) x = to_bf(rnd() * 4.0f);      /* sign-changing, not ones */
    }
    base("hp-absent"); {
        Case& c = v.back();
        c.steps = 3; c.K = 6; c.R = 8; c.V = 20; c.width = 2; c.budget = 10;
        c.hp = false;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 3;
    }
    base("ties"); {
        Case& c = v.back();
        c.steps = 2; c.K = 4; c.R = 3; c.V = 8; c.width = 4; c.budget = 15;
        c.alloc();
        c.anchor[0] = 2;
        for (int64_t v2 = 0; v2 < c.V; ++v2)
            for (int64_t r = 0; r < c.R; ++r) {
                c.pred[size_t(v2 * c.R + r)] = to_bf(0.5f);
                c.succ[size_t(v2 * c.R + r)] = to_bf(0.25f);
            }
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t j = 0; j < c.K; ++j) {
                c.C(0, l, j) = int32_t(1 + j);
                c.U(0, l, j) = 0.75f;              /* identical, exactly representable */
            }
    }
    base("penalty-overflow"); {
        /* All path scores are exactly zero. Depth 1 remains finite; depths 2 and 3 tie at -inf.
         * Budget 6 keeps all four shallow nodes and the first two old ids at depth 2. */
        Case& c = v.back();
        c.steps = 3; c.K = 4; c.R = 3; c.V = 8; c.width = 4; c.budget = 6;
        c.beta = -2.0e38;
        c.alloc();
        c.anchor[0] = 0;
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t j = 0; j < c.K; ++j)
                c.C(0, l, j) = int32_t(1 + j);
    }
    base("invalid-ids"); {
        Case& c = v.back();
        c.steps = 2; c.K = 6; c.R = 3; c.V = 12; c.width = 3; c.budget = 8;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 4;
        c.C(0, 0, 0) = -1;      c.U(0, 0, 0) = 50.0f;
        c.C(0, 1, 1) = int32_t(c.V); c.U(0, 1, 1) = 50.0f;
    }
    base("invalid-anchor"); {
        Case& c = v.back();
        c.steps = 2; c.K = 4; c.R = 3; c.V = 12; c.width = 2; c.budget = 8;
        c.alloc(); fill_codebooks(c); fill_candidates(c);
        c.anchor[0] = int32_t(c.V);
    }
    base("budget-1"); {
        Case& c = v.back();
        c.steps = 3; c.K = 8; c.R = 4; c.V = 16; c.width = 4; c.budget = 1;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 0;
    }
    base("budget-15"); {
        Case& c = v.back();
        c.steps = 3; c.K = 16; c.R = 4; c.V = 48; c.width = 16; c.budget = 15;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 11;
    }
    base("beta-negative"); {
        Case& c = v.back();
        c.steps = 3; c.K = 6; c.R = 4; c.V = 16; c.width = 3; c.budget = 6;
        c.beta = -0.5;
        c.alloc(); fill_codebooks(c); fill_candidates(c); c.anchor[0] = 5;
    }
    base("multi-sequence"); {
        Case& c = v.back();
        c.M = 4; c.steps = 2; c.K = 6; c.R = 4; c.V = 32; c.width = 3; c.budget = 10;
        c.alloc(); fill_codebooks(c);
        for (int64_t m = 0; m < c.M; ++m) {
            c.anchor[size_t(m)] = int32_t(3 * m + 1);
            for (int64_t l = 0; l < c.steps; ++l)
                for (int64_t j = 0; j < c.K; ++j) {
                    c.C(m, l, j) = int32_t((m * 7 + l * 5 + j) % c.V);
                    c.U(m, l, j) = rnd() + 0.1f * float(m);
                }
        }
    }
    base("duplicate-ids"); {
        /* THE CONSUMER'S CONSTRAINT, CHECKED ON THE CARD. The scheduler rejects a tree with two
         * siblings proposing the same token, so repeated candidate ids must collapse to one node
         * each, keeping the best-scoring instance. Here every level's candidate row repeats ids
         * with DIFFERING unary terms, so an implementation that kept the first occurrence and one
         * that kept the best would emit different tokens -- and the oracle and the kernel must
         * agree on the best-scoring choice. */
        Case& c = v.back();
        c.steps = 2; c.K = 6; c.R = 4; c.V = 16; c.width = 4; c.budget = 12;
        c.alloc(); fill_codebooks(c);
        c.anchor[0] = 3;
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t j = 0; j < c.K; ++j) {
                /* three distinct ids a level, each twice, with rising and falling unary terms. */
                c.C(0, l, j) = int32_t((l * 2 + j / 2 + 1) % c.V);
                c.U(0, l, j) = (j % 2 == 0) ? (0.5f + 0.25f * float(j))
                                            : (2.0f - 0.5f * float(j));
            }
    }
    base("reduction-order"); {
        /* THE ARITHMETIC'S ORDER, CHECKED ON THE CARD. Candidate 0's succ plane is 1 at r = 0..62
         * and 2^24 at r = 63, so the lane fold loses the one in lane 31 and totals 2^24 + 32,
         * while an ascending sum would keep it and total 2^24 + 64. Candidate 1 is a constant
         * 2^24 + 40, between the two. The device and the oracle must both answer with the fold's
         * ranking; an implementation that summed ascending would disagree here even though it
         * agreed everywhere else. */
        Case& c = v.back();
        c.steps = 1; c.K = 2; c.R = 64; c.V = 8; c.width = 1; c.budget = 15;
        c.alloc();
        c.anchor[0] = 0;
        const float big = 16777216.0f;
        for (int64_t v2 = 0; v2 < c.V; ++v2)
            for (int64_t r = 0; r < c.R; ++r) {
                c.pred[size_t(v2 * c.R + r)] = to_bf(1.0f);
                c.hpv[size_t(r)] = to_bf(1.0f);
                c.succ[size_t(v2 * c.R + r)] = 0;
            }
        for (int64_t r = 0; r < c.R; ++r) {
            c.succ[size_t(0 * c.R + r)] = to_bf(r == 63 ? big : 1.0f);
            c.succ[size_t(1 * c.R + r)] = to_bf(big + 40.0f);
        }
        c.C(0, 0, 0) = 0;
        c.C(0, 0, 1) = 1;
        c.U(0, 0, 0) = 0.0f;
        c.U(0, 0, 1) = 0.0f;
    }
    base("anchor-stride"); {
        Case& c = v.back();
        c.M = 3; c.astr = 4; c.steps = 3; c.K = 4; c.R = 3; c.V = 16; c.width = 2; c.budget = 9;
        c.alloc(); fill_codebooks(c); fill_candidates(c);
        c.anchor.assign(size_t(c.M * c.astr), 0);
        for (int64_t m = 0; m < c.M; ++m) c.anchor[size_t(m * c.astr)] = int32_t(m + 2);
    }
    base("odd-rank-frontier"); {
        /* K and frontier*K do not divide the four scoring waves. R crosses two wave strides
         * with just lane 0 active on the last stride. Unary alone prefers the opposite winner. */
        Case& c = v.back();
        c.M = 3; c.astr = 4; c.steps = 3; c.K = 7; c.R = 65; c.V = 32;
        c.width = 5; c.budget = 15;
        c.alloc(); fill_codebooks(c); fill_candidates(c);
        std::fill(c.anchor.begin(), c.anchor.end(), -1);
        for (int64_t v2 = 0; v2 < c.V; ++v2) {
            c.pred[size_t(v2 * c.R + 64)] = to_bf(float(1 + v2 % 3));
            c.succ[size_t(v2 * c.R + 64)] = to_bf(16.0f * float(v2));
        }
        for (auto& x : c.hpv) x = to_bf(rnd() * 0.25f);
        for (int64_t m = 0; m < c.M; ++m) {
            c.anchor[size_t(m * c.astr)] = int32_t(3 * m + 1);
            for (int64_t l = 0; l < c.steps; ++l) {
                const bool negative = (m + l) % 2 != 0;
                c.hpv[size_t((m * c.steps + l) * c.R + 64)] = to_bf(negative ? -2.0f : 2.0f);
                for (int64_t j = 0; j < c.K; ++j)
                    c.U(m, l, j) = (negative ? 8.0f : -8.0f) * float(c.C(m, l, j));
            }
        }
    }
    return v;
}

/* ======================= against the SHIPPING selector =======================
 *
 * The oracle cases above prove the device row agrees with a host implementation of the SAME op.
 * That is necessary and not sufficient: if both were built on a misreading of `dflash_select`, they
 * would agree with each other and both be wrong.
 *
 * This comparison pins libr4d's `dflash_select` row BY NAME -- never through resolve(), which could
 * answer with this tree op -- and runs it on the same bytes at `tree_width = 1`. The shipping
 * kernel emits `steps` token ids in a `I32[M, steps]` plane; the tree emits the anchor plus a chain
 * in the packed layout. The claim being checked is that the tree's draft tokens, in order, ARE the
 * shipping walk's.
 *
 * THE TWO OPS DO NOT SHARE A SCHEMA, so this binds each side's own operand list against one host
 * image, exactly as dflash2_fusion_gpu_test.cpp does for its fusions. */
const rad::KernelRow* pin_row(rad::Registry& reg, const char* op, const char* kernel,
                              const char* plugin) {
    const auto* rows = reg.rows_for(op);
    require(rows != nullptr && !rows->empty(), std::string("no row at all for op '") + op + "'");
    const rad::KernelRow* hit = nullptr;
    for (const rad::KernelRow* r : *rows) {
        if (!r->info || !r->info->name || std::strcmp(r->info->name, kernel) != 0) continue;
        require(r->plugin == plugin, std::string("kernel '") + kernel + "' is published by '" +
                                    r->plugin + "', expected '" + plugin + "'");
        require(hit == nullptr, std::string("duplicate row named '") + kernel + "'");
        hit = r;
    }
    require(hit != nullptr, std::string("missing pinned row ") + plugin + "/" + kernel +
                            " for op '" + op + "'");
    require(hit->info->launch != nullptr, std::string("pinned row ") + kernel + " has no launcher");
    require(hit->info->domain == RAD_DOMAIN_DEVICE,
            std::string("pinned row ") + kernel + " is not a device row");
    std::printf("ROW role=pinned plugin=%s kernel=%s op=%s\n", hit->plugin.c_str(),
                hit->info->name, op);
    return hit;
}

/* One scene, shared by both sides. `R` is a multiple of 32 because that is what the shipping row
 * accepts; anything else would be refused there and the comparison would be vacuous. */
struct ShipScene {
    int64_t M = 2, steps = 3, K = 8, R = 64, V = 20, astr = 1, budget = 15;
    std::vector<int32_t> cand, anchor;
    std::vector<float> unary;
    std::vector<uint16_t> hp, pred, succ;
    std::vector<int32_t> ship;     /* [M, steps]  the shipping walk */
    std::vector<int32_t> tree;     /* [M, 33]     this op, width 1 */
};

size_t compare_against_shipping_selector(rad::Registry& reg, const Session& s) {
    ShipScene c;
    c.cand.assign(size_t(c.M * c.steps * c.K), 0);
    c.unary.assign(size_t(c.M * c.steps * c.K), 0.0f);
    c.anchor.assign(size_t(c.M * c.astr), 0);
    c.hp.assign(size_t(c.M * c.steps * c.R), 0);
    c.pred.assign(size_t(c.V * c.R), 0);
    c.succ.assign(size_t(c.V * c.R), 0);
    c.ship.assign(size_t(c.M * c.steps), -77);

    /* Deterministic, and chosen so the two sides cannot agree by accident: every codebook entry
     * differs, and the unary terms are large enough to compete with the edge sums. */
    g_rng = 0x9e3779b9u;
    for (int64_t v = 0; v < c.V; ++v)
        for (int64_t r = 0; r < c.R; ++r) {
            c.pred[size_t(v * c.R + r)] = to_bf(rnd());
            c.succ[size_t(v * c.R + r)] = to_bf(rnd());
        }
    for (int64_t m = 0; m < c.M; ++m) {
        c.anchor[size_t(m)] = int32_t(2 * m + 1);
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t j = 0; j < c.K; ++j) {
                c.cand[size_t((m * c.steps + l) * c.K + j)] = int32_t((m * 5 + l * 3 + j * 2) % c.V);
                c.unary[size_t((m * c.steps + l) * c.K + j)] = rnd() * 2.0f;
            }
        for (int64_t l = 0; l < c.steps; ++l)
            for (int64_t r = 0; r < c.R; ++r)
                c.hp[size_t((m * c.steps + l) * c.R + r)] = to_bf(rnd() * 3.0f);
    }
    /* A REPEATED ID IN THE FIRST LEVEL OF EVERY ROW, with a DIFFERENT unary term. The shipping
     * walk scores each instance independently and picks the best token; this op must collapse the
     * repeat and still pick the same token, which is only true because the retained instance is the
     * best-scoring one. The slot replaced is the last, so the walk's favourite is unlikely to be the
     * removed instance by accident. */
    for (int64_t m = 0; m < c.M; ++m) {
        const int64_t last = c.K - 1;
        c.cand[size_t((m * c.steps + 0) * c.K + last)] =
            c.cand[size_t((m * c.steps + 0) * c.K + 0)];
        c.unary[size_t((m * c.steps + 0) * c.K + last)] += 0.75f;
    }

    /* ---- the shipping row ---- */
    rad::Geometry sg;
    sg.set_i("M", c.M); sg.set_i("steps", c.steps); sg.set_i("top_k", c.K);
    sg.set_i("rank", c.R); sg.set_i("n_vocab", c.V); sg.set_i("anchor_stride", c.astr);
    sg.set_s("dtype", "bf16");
    const rad::KernelRow* shipRow = pin_row(reg, "dflash_select", "dflash_select_bf16", "libr4d");
    rad::Resolved sr;
    sr.row = shipRow;
    sr.geom = sg;
    check(reg.init_instance(sr, 0, 1), "init shipping selector");

    std::vector<RadTensor> st(7);
    auto shape2 = [](RadTensor& x, uint32_t dt, int64_t a, int64_t b) {
        x.dtype = dt; x.rank = 2; x.shape[0] = a; x.shape[1] = b;
        x.stride[0] = b; x.stride[1] = 1;
    };
    shape2(st[0], RAD_I32, c.M * c.steps, c.K);
    shape2(st[1], RAD_F32, c.M * c.steps, c.K);
    shape2(st[2], RAD_BF16, c.M * c.steps, c.R);
    st[3].dtype = RAD_I32; st[3].rank = 1; st[3].shape[0] = c.M * c.astr; st[3].stride[0] = 1;
    shape2(st[4], RAD_BF16, c.V, c.R);
    shape2(st[5], RAD_BF16, c.V, c.R);
    st[6].dtype = RAD_I32; st[6].rank = 2;
    st[6].shape[0] = c.M; st[6].shape[1] = c.steps;
    st[6].stride[0] = c.steps; st[6].stride[1] = 1;

    const std::vector<const void*> host = { c.cand.data(), c.unary.data(), c.hp.data(),
                                             c.anchor.data(), c.pred.data(), c.succ.data(),
                                             c.ship.data() };
    std::vector<size_t> bytes = { c.cand.size() * 4, c.unary.size() * 4, c.hp.size() * 2,
                                  c.anchor.size() * 4, c.pred.size() * 2, c.succ.size() * 2,
                                  c.ship.size() * 4 };
    std::vector<std::unique_ptr<DeviceBuffer>> sbuf;
    std::vector<RadTensor> stDev = st;
    for (size_t i = 0; i < st.size(); ++i) {
        sbuf.emplace_back(new DeviceBuffer(bytes[i]));
        stDev[i].data = sbuf.back()->data;
        if (bytes[i]) check(rad_memcpy_async(sbuf.back()->data, host[i], int64_t(bytes[i]), s.stream),
                            "upload shipping operand");
    }
    check(rad_stream_sync(s.stream), "finish shipping upload");

    RadArgs sa{};
    sa.t = stDev.data(); sa.n_t = 7;
    sa.p = sr.geom.params(); sa.n_p = sr.geom.n_params(); sa.instance = sr.instance;
    sa.rank = 0; sa.world_size = 1;
    check(shipRow->info->launch(&sa, s.stream), "launch the shipping dflash_select");
    check(rad_memcpy_async(c.ship.data(), sbuf.back()->data, int64_t(bytes[6]), s.stream),
          "download the shipping tokens");
    check(rad_stream_sync(s.stream), "finish shipping download");

    /* ---- this op at width 1 ---- */
    Case t;
    t.name = "shipping-equivalence";
    t.M = c.M; t.steps = c.steps; t.K = c.K; t.R = c.R; t.V = c.V; t.astr = c.astr;
    t.budget = 15; t.width = 1;
    t.cand = c.cand; t.unary = c.unary; t.hpv = c.hp; t.anchor = c.anchor;
    t.pred = c.pred; t.succ = c.succ;
    t.expect.assign(size_t(c.M * kCols), 0);
    run_case(reg, s, t);

    /* Each row's tree must be a CHAIN whose draft tokens are the shipping walk's, in order. */
    for (int64_t m = 0; m < c.M; ++m) {
        const int32_t* tr = t.expect.data() + m * kCols;
        const int32_t* sh = c.ship.data() + m * c.steps;
        if (tr[0] < 1) fail("shipping-equivalence: the tree row is invalid");
        if (tr[0] - 1 > c.steps) fail("shipping-equivalence: more draft nodes than steps");
        for (int i = 1; i < tr[0]; ++i) {
            if (tr[1 + kCap + i] != i - 1)
                fail("shipping-equivalence: width 1 did not produce a chain");
            if (tr[1 + i] != sh[i - 1]) {
                char msg[256];
                std::snprintf(msg, sizeof msg,
                              "shipping-equivalence: row %lld node %d: tree=%d shipping=%d",
                              (long long)m, i, tr[1 + i], sh[i - 1]);
                fail(msg);
            }
        }
    }
    std::printf("SHIPPING_PASS rows=%lld steps=%lld rank=%lld tokens=%lld\n",
                (long long)c.M, (long long)c.steps, (long long)c.R,
                (long long)(c.M * c.steps));
    return size_t(c.M * c.steps) * 4;
}

}  /* namespace */

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "--help") {
            std::printf("usage: %s --mine PATH --ref PATH --libr4d PATH\n"
                        "  --libr4d is REQUIRED: the single-path equivalence is checked against\n"
                        "  the shipping dflash_select kernel itself, not only the host oracle.\n",
                        argv[0]);
            return 0;
        }
        const char* mine = argc >= 3 && std::string(argv[1]) == "--mine" ? argv[2] : nullptr;
        const char* ref = argc >= 5 && std::string(argv[3]) == "--ref" ? argv[4] : nullptr;
        const char* r4d = argc >= 7 && std::string(argv[5]) == "--libr4d" ? argv[6] : nullptr;
        require(mine && ref && r4d,
                "the comparator needs --mine <plugin.so> --ref <plugin.so> --libr4d <plugin.so>");

        rad::Registry reg;
        reg.set_code_device(0);
        check(reg.load_plugin(mine, 0), "load mine");
        check(reg.load_plugin(ref, 1), "load mine_ref");
        check(reg.load_plugin(r4d, 2), "load libr4d");
        check(reg.check_complete(), "complete plugin schemas");
        require(reg.plugin("mine") && reg.plugin("mine_ref") && reg.plugin("libr4d"),
                "expected mine, mine_ref and libr4d plugin identities");

        Session session;
        size_t total = 0;
        const std::vector<Case> all = cases();
        for (const Case& scene : all) {
            Case c = scene;
            total += run_case(reg, session, c);
        }
        require(!all.empty() && total > 0, "zero cases compared");

        /* THE PRODUCTION COMPARISON, which the oracle-vs-device cases above cannot substitute for:
         * two implementations of one op by the same author agreeing proves consistency, not that
         * the SHIPPING kernel computes the same tokens. */
        total += compare_against_shipping_selector(reg, session);

        std::printf("ALL_PASS cases=%zu bytes=%zu\n", all.size() + 1, total);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
