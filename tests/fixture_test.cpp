/* fixture_test.cpp -- the recorded reference itself: the draw, the reduction and the file.
 *
 * Every kernel correctness number is relative to this code. rad-kbench compares a kernel against
 * an answer libref recorded elsewhere and earlier; that is valid only if both sides drew
 * byte-identical inputs from a stored seed and reduced outputs the same way. A drift shows up as a
 * page of kernel failures, or worse as agreement, since two all-zero sides agree perfectly.
 *
 * The cases pin the properties the comparison rests on:
 *
 *   - a weight's bytes depend on the operand's identity, not the case, so an M sweep draws one
 *     weight once and two libraries are comparable;
 *   - the blocked draw (large operands) and the whole draw (small ones) are one stream;
 *   - the reduction is independent of thread count, or a recorded norm is machine-specific;
 *   - an input lies in the domain its op is handed, or the comparison measures the test's
 *     condition number; a sampler row is a request, or a stage gated on it compares two copies of
 *     its input;
 *   - every dtype a recorded case uses can be both drawn and read back;
 *   - a weight's planes are drawn within what each can hold, the reference reads their decoded
 *     values, and a kernel gets what its own layout hooks make of them.
 */
#include "rad_test.h"
#include "kfixture.h"
#include "opshapes.h"
#include "rad_plugin.h"
#include "rad_sample.h"
#include "plugin/registry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>

using namespace rad;

namespace {

/* rad_narrow of `src` into `dt`, then rad_widen of those bytes: whether the narrow wrote any
 * non-zero byte and the widen read any non-zero value back (one per element, checked). */
struct NarrowWiden { bool wrote = false, read = false; };
NarrowWiden narrow_widen(const float* src, uint32_t dt, int64_t n) {
    NarrowWiden r;
    std::vector<uint8_t> packed((size_t)rad_dtype_bytes(dt, n), 0);
    rad_narrow(src, dt, n, packed.data());
    for (uint8_t b : packed) r.wrote |= b != 0;
    std::vector<float> back;
    rad_widen(packed.data(), dt, n, &back);
    CHECK_EQ((int64_t)back.size(), n);
    for (float x : back) r.read |= x != 0.0f;
    return r;
}

/* Overwrites `n` bytes of the file at `off`; false when it cannot be opened. */
bool poke(const std::string& path, long off, const void* bytes, size_t n) {
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) return false;
    std::fseek(f, off, SEEK_SET);
    std::fwrite(bytes, 1, n, f);
    std::fclose(f);
    return true;
}

/* An f32 operand of `shape` with the given role and draw law. */
KfOperand opd_f32(std::vector<int64_t> shape, int role = RAD_OPD_IN, int fill = KF_FILL_NORMAL) {
    KfOperand d;
    d.role = role;
    d.dtype = RAD_F32;
    d.shape = std::move(shape);
    d.fill = fill;
    return d;
}

/* Selects the planes of d->enc named `roles` (every plane when empty) over a [rows, cols] weight,
 * each with its own extents. */
void select_planes(KfOperand* d, int64_t rows, int64_t cols,
                   const std::vector<const char*>& roles = {}) {
    const int n = roles.empty() ? d->enc.n_planes : (int)roles.size();
    for (int j = 0; j < n; ++j) {
        const int i = roles.empty() ? j : rad_enc_find(&d->enc, roles[(size_t)j]);
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&d->enc.plane[i], rows, cols, &pr, &pc);
        d->sel.push_back(i);
        d->sel_rows.push_back(pr);
        d->sel_cols.push_back(pc);
    }
}

/* A weight operand of a block-fp8 [rows, cols] weight selecting `roles`, as a declaration records
 * it: one plane takes its own dtype and extents, several take f32 and the logical extents. */
KfOperand fp8_weight(int64_t rows, int64_t cols, std::vector<const char*> roles) {
    KfOperand d;
    d.role = RAD_OPD_WEIGHT;
    d.has_enc = true;
    rad_enc_clear(&d.enc);
    rad_enc_copy_str(d.enc.scheme, "affine");
    rad_enc_add_plane(&d.enc, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&d.enc, "scale", RAD_BF16, 128, 128);
    select_planes(&d, rows, cols, roles);
    if (roles.size() == 1) {
        d.dtype = d.enc.plane[d.sel[0]].dtype;
        d.shape = { d.sel_rows[0], d.sel_cols[0] };
    } else {
        d.dtype = RAD_F32;
        d.shape = { rows, cols };
    }
    return d;
}

int64_t numel(const KfOperand& d) {
    int64_t n = 1;
    for (int64_t x : d.shape) n *= x;
    return n;
}

std::vector<float> draw_f32(const std::vector<KfOperand>& opd, size_t k, int64_t M, uint64_t seed) {
    std::vector<uint8_t> bytes;
    kf_draw(opd, k, M, kf_vocab_rows(opd), seed, &bytes);
    std::vector<float> out;
    rad_widen(bytes.data(), opd[k].dtype, numel(opd[k]), &out);
    return out;
}

}  /* namespace */

/* ================================================================== the draw's block contract */

/* RAD_FILL_BLOCK is part of the draw contract. kf_draw streams an operand larger than one block
 * through rad_fill_normal_block, so the float intermediate stays small; smaller ones go through
 * rad_fill_normal in one pass. Both must produce the same stream, or a weight's values would depend
 * on its size. */
TEST(a_blocked_draw_and_a_whole_one_are_the_same_stream) {
    const int64_t n = RAD_FILL_BLOCK * 3 + 517;   /* three full blocks and a partial one */
    std::vector<float> whole((size_t)n, 0.0f);
    rad_fill_normal(whole.data(), n, 0xC0FFEEull, 0.02f);

    std::vector<float> piecewise((size_t)n, 0.0f);
    for (int64_t lo = 0; lo < n; lo += RAD_FILL_BLOCK) {
        const int64_t len = (n - lo) < RAD_FILL_BLOCK ? (n - lo) : RAD_FILL_BLOCK;
        rad_fill_normal_block(piecewise.data() + lo, len, 0xC0FFEEull, lo / RAD_FILL_BLOCK, 0.02f);
    }
    CHECK_EQ(std::memcmp(whole.data(), piecewise.data(), (size_t)n * sizeof(float)), 0);
}

/* An operand spanning several blocks is drawn by threads, round-robin over the machine's cores.
 * A draw that depended on the core count would make every recorded reference machine-specific, so
 * the threaded draw must equal one thread drawing the blocks in order. */
TEST(a_draw_does_not_change_between_runs) {
    const int64_t n = RAD_FILL_BLOCK * 5;
    std::vector<float> a((size_t)n, 0.0f), b((size_t)n, 0.0f);
    rad_fill_normal(a.data(), n, 12345ull, 1.0f);
    for (int64_t blk = 0; blk < 5; ++blk)
        rad_fill_normal_block(b.data() + blk * RAD_FILL_BLOCK, RAD_FILL_BLOCK, 12345ull, blk, 1.0f);
    CHECK_EQ(std::memcmp(a.data(), b.data(), (size_t)n * sizeof(float)), 0);
    /* The fill must be a unit-variance normal; an all-zero draw would pass the equality above. */
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double)a[(size_t)i] * (double)a[(size_t)i];
    CHECK_NEAR(std::sqrt(ss / (double)n), 1.0, 0.02);
}

TEST(a_case_seed_separates_the_op_the_band_the_domain_and_the_operand) {
    const uint64_t base = 7;
    const uint64_t s = rad_case_seed(base, "rmsnorm", 0, 0, 0);
    CHECK(s != rad_case_seed(base, "softmax", 0, 0, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 1, 0, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 0, 1, 0));
    CHECK(s != rad_case_seed(base, "rmsnorm", 0, 0, 1));
    CHECK_EQ(s, rad_case_seed(base, "rmsnorm", 0, 0, 0));
    CHECK(rad_case_seed(base, "rmsnorm", 0, 0, 0) != 0);
}

/* ================================================================== narrow and widen */

/* rad_narrow writes what a kernel reads and rad_widen reads what a kernel wrote, so a dtype missing
 * from either draws or compares zeros, and two zeroed sides pass. Covers the core's integer and
 * float types; packed weight formats (W4, W2, MXFP4) are absent from both on purpose, since only
 * the kernel's layout hook knows their bytes.
 *
 * This checks completeness, not identity: a drawn i8 is a quantised unit normal with a scale, so
 * writing 5.0 and reading 127 back is correct. Identity is checked below where a value is its own
 * meaning. */
TEST(every_plain_dtype_the_harness_writes_it_can_read_back) {
    struct Row { uint32_t dt; const char* name; };
    const Row rows[] = {
        { RAD_F32, "f32" }, { RAD_BF16, "bf16" }, { RAD_F16, "f16" },
        { RAD_F8E4M3, "f8e4m3" }, { RAD_F8E5M2, "f8e5m2" },
        { RAD_I8, "i8" }, { RAD_U8, "u8" }, { RAD_I16, "i16" },
        { RAD_I32, "i32" }, { RAD_I64, "i64" }, { RAD_U32, "u32" }, { RAD_BOOL, "bool" },
    };
    const float src[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);

    for (const Row& r : rows) {
        const NarrowWiden t = narrow_widen(src, r.dt, n);
        if (!t.wrote || !t.read)
            std::fprintf(stderr, "    %s: narrow wrote %s, widen read %s\n", r.name,
                         t.wrote ? "bytes" : "NOTHING", t.read ? "values" : "NOTHING");
        CHECK(t.wrote);
        CHECK(t.read);
    }
}

/* Integer operands carry token ids, slot numbers, lengths and table sizes, so they round trip
 * exactly; any scaling or rounding would yield a different index. i8 and u8 are excluded: an i8
 * operand here is a quantised activation, not an index. */
TEST(an_integer_operand_round_trips_exactly) {
    const uint32_t dts[] = { RAD_I16, RAD_I32, RAD_I64, RAD_U32 };
    const float src[] = { 0.0f, 1.0f, 17.0f, 4095.0f, 65535.0f, 7.0f, 250.0f, 3.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);
    for (uint32_t dt : dts) {
        std::vector<uint8_t> packed((size_t)rad_dtype_bytes(dt, n), 0);
        rad_narrow(src, dt, n, packed.data());
        std::vector<float> back;
        rad_widen(packed.data(), dt, n, &back);
        if ((int64_t)back.size() != n) { CHECK_EQ((int64_t)back.size(), n); continue; }
        for (int64_t i = 0; i < n; ++i) {
            /* the largest probe exceeds i16's range; everything else is exact */
            if (dt == RAD_I16 && src[i] > 32767.0f) continue;
            if (back[(size_t)i] != src[i])
                std::fprintf(stderr, "    %s: %g came back %g\n", rad_dtype_name(dt), src[i],
                             back[(size_t)i]);
            CHECK_EQ(back[(size_t)i], src[i]);
        }
    }
    /* bool keeps one bit of each value. */
    const float flags[] = { 0.0f, 1.0f, 0.0f, 9.0f };
    std::vector<uint8_t> packed(4, 0xEE);
    rad_narrow(flags, RAD_BOOL, 4, packed.data());
    std::vector<float> back;
    rad_widen(packed.data(), RAD_BOOL, 4, &back);
    CHECK_EQ((int64_t)back.size(), (int64_t)4);
    for (int i = 0; i < 4 && i < (int)back.size(); ++i)
        CHECK_EQ(back[(size_t)i], flags[i] != 0.0f ? 1.0f : 0.0f);
}

/* Float formats round trip to their own precision, so a recorded bf16 reference is comparable
 * with a bf16 output. */
TEST(a_float_dtype_round_trips_to_its_own_precision) {
    struct Row { uint32_t dt; double tol; };
    const Row rows[] = { { RAD_F32, 0.0 }, { RAD_BF16, 0.05 }, { RAD_F16, 0.005 },
                         { RAD_F8E4M3, 0.0 }, { RAD_F8E5M2, 1.0 } };
    const float src[] = { 0.0f, 1.0f, 2.0f, 5.0f, 9.0f, -3.0f, 7.0f, -4.0f };
    const int64_t n = (int64_t)(sizeof src / sizeof src[0]);
    for (const Row& r : rows) {
        std::vector<uint8_t> packed((size_t)rad_dtype_bytes(r.dt, n), 0);
        rad_narrow(src, r.dt, n, packed.data());
        std::vector<float> back;
        rad_widen(packed.data(), r.dt, n, &back);
        if ((int64_t)back.size() != n) { CHECK_EQ((int64_t)back.size(), n); continue; }
        for (int64_t i = 0; i < n; ++i)
            CHECK_NEAR(back[(size_t)i], src[i], r.tol);
    }
}

TEST(an_operand_seed_is_distinct_per_operand_and_stable) {
    const uint64_t c = 0xABCDEF01ull;
    std::set<uint64_t> seen;
    for (int k = 0; k < 32; ++k) {
        const uint64_t s = kf_operand_seed(c, k);
        CHECK(seen.insert(s).second);
        CHECK_EQ(s, kf_operand_seed(c, k));
    }
    CHECK(kf_operand_seed(c, 0) != kf_operand_seed(c + 1, 0));
}

/* A weight's seed is its identity, not its case. An M sweep thus draws a large weight once, and
 * the caller caches the drawn bytes under that key; keyed on the case, the cache would never hit
 * and the weight would differ at every point of the sweep. */
TEST(a_weight_is_drawn_from_its_own_identity_and_not_the_case) {
    std::vector<KfOperand> opd = { opd_f32({ 64, 32 }, RAD_OPD_WEIGHT) };

    std::vector<uint8_t> a, b;
    kf_draw(opd, 0, /*M=*/1, kf_vocab_rows(opd), kf_operand_seed(111, 0), &a);
    kf_draw(opd, 0, /*M=*/512, kf_vocab_rows(opd), kf_operand_seed(999, 0), &b);
    CHECK_EQ(a.size(), b.size());
    CHECK_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);

    /* Identity is position, dtype and extents; changing any gives a different weight, so two
     * unrelated operands never share a cache entry. */
    const uint64_t s = kf_weight_seed(opd[0], 0);
    CHECK_EQ(s, kf_weight_seed(opd[0], 0));
    CHECK(s != kf_weight_seed(opd[0], 1));
    KfOperand other = opd[0];
    other.dtype = RAD_BF16;
    CHECK(s != kf_weight_seed(other, 0));
    other = opd[0];
    other.shape = { 32, 64 };
    CHECK(s != kf_weight_seed(other, 0));
}

/* A weight is drawn at a checkpoint's scale (rms 0.02), not unit variance; at unit variance an fp8
 * quantiser's amax would be orders of magnitude from a real model's. */
TEST(a_weight_is_drawn_at_the_scale_a_checkpoint_holds) {
    std::vector<KfOperand> opd = { opd_f32({ 256, 256 }, RAD_OPD_WEIGHT) };
    const std::vector<float> w = draw_f32(opd, 0, 1, 0);
    double ss = 0;
    for (float x : w) ss += (double)x * (double)x;
    CHECK_NEAR(std::sqrt(ss / (double)w.size()), 0.02, 0.002);
}

/* ================================================================== sampling positions */

TEST(a_small_operand_is_sampled_exhaustively) {
    std::vector<int64_t> pos;
    kf_sample_positions(42, 300, &pos);
    CHECK_EQ((int64_t)pos.size(), 300);
    for (int64_t i = 0; i < 300; ++i) CHECK_EQ(pos[(size_t)i], i);
    /* Below the sample count nothing is drawn, so the seed does not matter. */
    std::vector<int64_t> other;
    kf_sample_positions(43, 300, &other);
    CHECK(pos == other);
}

TEST(a_large_operand_is_sampled_at_positions_both_sides_can_derive) {
    const int64_t n = 1 << 20;
    std::vector<int64_t> a, b;
    kf_sample_positions(0x5EEDull, n, &a);
    kf_sample_positions(0x5EEDull, n, &b);
    CHECK_EQ((int64_t)a.size(), (int64_t)KF_SAMPLES);
    CHECK(a == b);
    for (int64_t p : a) CHECK(p >= 0 && p < n);
    std::vector<int64_t> c;
    kf_sample_positions(0x5EEEull, n, &c);
    CHECK(a != c);
}

TEST(an_empty_operand_is_sampled_at_no_position) {
    std::vector<int64_t> pos = { 9 };
    kf_sample_positions(1, 0, &pos);
    CHECK(pos.empty());
    kf_sample_positions(1, -4, &pos);
    CHECK(pos.empty());
}

/* ================================================================== the reduction */

/* The reduction is threaded over blocks and the partials are added in block order, so the sum is
 * independent of core count. The expected value is a serial blocked sum, not a left-to-right one,
 * which differs in floating point. */
TEST(the_reduction_adds_its_blocks_back_in_order) {
    const int64_t n = KF_REDUCE_BLOCK * 3 + 1000;   /* past the threaded threshold */
    std::vector<float> v((size_t)n, 0.0f);
    rad_fill_normal(v.data(), n, 0xBEEFull, 1.0f);

    double got_ss = 0;
    int64_t got_nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), n, 7, &got_ss, &got_nf, &sample);

    double want = 0;
    for (int64_t lo = 0; lo < n; lo += KF_REDUCE_BLOCK) {
        const int64_t hi = std::min<int64_t>(lo + KF_REDUCE_BLOCK, n);
        double part = 0;
        for (int64_t i = lo; i < hi; ++i) part += (double)v[(size_t)i] * (double)v[(size_t)i];
        want += part;
    }
    CHECK_EQ(got_ss, want);
    CHECK_EQ(got_nf, (int64_t)0);

    double again_ss = 0;
    int64_t again_nf = 0;
    std::vector<float> again;
    kf_reduce(v.data(), n, 7, &again_ss, &again_nf, &again);
    CHECK_EQ(again_ss, got_ss);
    CHECK(again == sample);
}

/* A NaN on one side and a number on the other fails before any tolerance, so the non-finite count
 * must be exact; non-finite elements stay out of the norm, or one NaN would poison the sum. */
TEST(a_non_finite_element_is_counted_and_kept_out_of_the_norm) {
    std::vector<float> v(10, 2.0f);
    v[3] = std::nanf("");
    v[7] = INFINITY;
    double ss = 0;
    int64_t nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), (int64_t)v.size(), 1, &ss, &nf, &sample);
    CHECK_EQ(nf, (int64_t)2);
    CHECK_EQ(ss, 8.0 * 4.0);          /* the eight finite 2.0s */
    CHECK_EQ((int64_t)sample.size(), (int64_t)v.size());
    CHECK(std::isnan(sample[3]));
    CHECK(std::isinf(sample[7]));
}

TEST(the_reduction_samples_the_positions_a_reader_will_ask_for) {
    const int64_t n = (int64_t)KF_SAMPLES * 4;
    std::vector<float> v((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)i;
    double ss = 0;
    int64_t nf = 0;
    std::vector<float> sample;
    kf_reduce(v.data(), n, 0x1234ull, &ss, &nf, &sample);

    std::vector<int64_t> pos;
    kf_sample_positions(0x1234ull, n, &pos);
    CHECK_EQ(sample.size(), pos.size());
    for (size_t i = 0; i < pos.size() && i < sample.size(); ++i)
        CHECK_EQ(sample[i], (float)pos[i]);
}

/* ================================================================== the input-domain laws
 *
 * Each law keeps an operand inside the domain its op is handed in practice. Outside it, the two
 * implementations disagree in proportion to the input's ill-conditioning, and the report blames
 * the kernel.
 */

/* Cumulative sequence lengths are the batch's shape, not a draw: every gated-delta-net kernel reads
 * cu[n] before any null test, so a wrong value faults. */
TEST(a_cumulative_length_operand_is_the_batch_shape_not_a_draw) {
    KfOperand d = opd_f32({ 5 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_cu = true;
    const std::vector<float> v = draw_f32({ d }, 0, /*M=*/100, 0xAAAA);
    CHECK_EQ((int64_t)v.size(), (int64_t)5);
    CHECK_EQ(v[0], 0.0f);
    CHECK_EQ(v[4], 100.0f);
    for (size_t i = 1; i < v.size(); ++i) CHECK(v[i] >= v[i - 1]);
}

/* A length operand is the recorded constant. A random context shorter than the query makes causal
 * attention read a nonexistent prefix, and both implementations agree on garbage. */
TEST(a_length_operand_is_the_constant_it_was_recorded_with) {
    KfOperand d = opd_f32({ 6 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_const = 4096;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xBBBB);
    for (float x : v) CHECK_EQ(x, 4096.0f);
}

/* A scatter's destinations are distinct: two tokens on one slot race, and the implementations
 * resolve the race differently. */
TEST(a_scatter_destination_is_drawn_without_repeats) {
    KfOperand d = opd_f32({ 8 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_unique = true;
    d.idx_max = 16;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xCCCC);
    std::set<int> seen;
    for (float x : v) {
        CHECK(x >= 0.0f && x < 16.0f);
        CHECK(seen.insert((int)x).second);
    }

    /* Destinations beyond the range are marked -1 rather than repeated, which would race. */
    d.idx_max = 4;
    const std::vector<float> w = draw_f32({ d }, 0, 8, 0xCCCC);
    std::set<int> first;
    for (size_t i = 0; i < 4; ++i) CHECK(first.insert((int)w[i]).second);
    for (size_t i = 4; i < w.size(); ++i) CHECK_EQ(w[i], -1.0f);
}

/* An index indexes the operand before it, and its leading extent is the only legal range. */
TEST(an_index_operand_stays_inside_the_operand_it_indexes) {
    std::vector<KfOperand> opd = { opd_f32({ 7, 3 }, RAD_OPD_WEIGHT), opd_f32({ 32 }) };
    opd[1].dtype = RAD_I32;
    opd[1].is_index = true;
    const std::vector<float> v = draw_f32(opd, 1, 32, 0xDDDD);
    bool any_past_zero = false;
    for (float x : v) {
        CHECK(x >= 0.0f && x < 7.0f);
        if (x > 0.0f) any_past_zero = true;
    }
    CHECK(any_past_zero);          /* drawn, not zeroed */

    /* Where the range is not an operand's leading extent (a KV slot indexes
     * n_blocks * block_size), the recipe names it and the named range wins. */
    opd[1].idx_max = 200;
    const std::vector<float> w = draw_f32(opd, 1, 32, 0xDDDD);
    bool past_seven = false;
    for (float x : w) {
        CHECK(x >= 0.0f && x < 200.0f);
        if (x >= 7.0f) past_seven = true;
    }
    CHECK(past_seven);
}

TEST(a_sigmoid_gate_is_drawn_inside_zero_to_one) {
    const std::vector<float> v = draw_f32({ opd_f32({ 64 }, RAD_OPD_IN, KF_FILL_SIGMOID) },
                                          0, 64, 0xEEEE);
    bool spread = false;
    for (float x : v) {
        CHECK(x > 0.0f && x < 1.0f);
        if (x > 0.6f || x < 0.4f) spread = true;
    }
    CHECK(spread);
}

/* The chunked delta rule's intra-chunk matrix is I + tril(diag(beta) K K^T, -1): unit lower
 * triangular with a bounded inverse. A normal draw on the diagonal can put an element near zero,
 * and gdn_kkt_solve then returns values around 1e20. */
TEST(an_intra_chunk_matrix_is_drawn_unit_lower_triangular) {
    KfOperand d = opd_f32({ 8, 2, 4 }, RAD_OPD_IN, KF_FILL_TRI_LOWER);
    d.fill_chunk = 4;
    const std::vector<float> v = draw_f32({ d }, 0, 8, 0xF00D);
    const int64_t heads = 2, ch = 4, rows = 8;
    for (int64_t t = 0; t < rows; ++t)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t j = 0; j < ch; ++j) {
                const float x = v[(size_t)((t * heads + h) * ch + j)];
                const int64_t diag = t % ch;
                if (j > diag)       CHECK_EQ(x, 0.0f);
                else if (j == diag) CHECK_EQ(x, 1.0f);
                else                CHECK(std::fabs(x) < 4.0f);
            }
}

/* The gated-delta-net key comes from the model's qk_norm with unit rows. Plain normals over a
 * 128-wide head have norm ~11, making the solve's off-diagonals ~100x too large. */
TEST(a_qk_normed_key_is_drawn_with_unit_rows) {
    const int64_t r = 6, w = 128;
    const std::vector<float> v = draw_f32({ opd_f32({ r, w }, RAD_OPD_IN, KF_FILL_UNIT_ROW) },
                                          0, r, 0x1010);
    for (int64_t i = 0; i < r; ++i) {
        double ss = 0;
        for (int64_t j = 0; j < w; ++j) {
            const double x = (double)v[(size_t)(i * w + j)];
            ss += x * x;
        }
        CHECK_NEAR(std::sqrt(ss), 1.0, 1e-5);
    }
}

/* The decay plane is an intra-chunk cumulative sum of log-decays that resets at every chunk
 * boundary, as chunk_local_cumsum produces and the scan assumes. Plain normals overflow the
 * reference. */
TEST(a_gate_cumsum_runs_down_a_chunk_and_resets_at_its_boundary) {
    const int64_t rows = 8, heads = 2;
    KfOperand d = opd_f32({ rows, heads }, RAD_OPD_IN, KF_FILL_GATE_CUMSUM);
    KfOperand plain = d;
    plain.fill_chunk = 0;                    /* one chunk over the whole plane: no reset */
    d.fill_chunk = 4;

    const std::vector<float> chunked = draw_f32({ d }, 0, rows, 0x2020);
    const std::vector<float> whole   = draw_f32({ plain }, 0, rows, 0x2020);

    for (int64_t h = 0; h < heads; ++h) {
        /* Strictly decreasing within a chunk: each step subtracts log1p(exp(x)) > 0 from an
         * accumulator starting at zero, so every value is negative. */
        for (int64_t t = 0; t < rows; ++t) {
            const float x = chunked[(size_t)(t * heads + h)];
            CHECK(x < 0.0f);
            if (t % 4 != 0) CHECK(x < chunked[(size_t)((t - 1) * heads + h)]);
        }
        /* The first chunk is identical either way. */
        for (int64_t t = 0; t < 4; ++t)
            CHECK_EQ(chunked[(size_t)(t * heads + h)], whole[(size_t)(t * heads + h)]);
        /* At the boundary the accumulator restarts: the chunked value equals the unchunked
         * step alone. */
        const double step = (double)whole[(size_t)(4 * heads + h)] -
                            (double)whole[(size_t)(3 * heads + h)];
        CHECK_NEAR(chunked[(size_t)(4 * heads + h)], step, 1e-5);
    }
}

TEST(an_output_operand_is_sized_and_left_alone) {
    const std::vector<float> v = draw_f32({ opd_f32({ 16, 4 }, RAD_OPD_OUT) }, 0, 16, 0x3030);
    CHECK_EQ((int64_t)v.size(), (int64_t)64);
    for (float x : v) CHECK_EQ(x, 0.0f);
}

TEST(an_absent_operand_draws_nothing) {
    KfOperand d = opd_f32({ 16 });
    d.absent = true;
    std::vector<uint8_t> bytes;
    kf_draw({ d }, 0, 16, 16, 1, &bytes);
    for (uint8_t b : bytes) CHECK_EQ((int)b, 0);
}

/* The fallback range for an index operand with nothing to index. Both sides derive it from the
 * case, not the model, so recorder and reader draw the same token ids from a seed. */
TEST(the_fallback_index_range_is_the_widest_leading_extent) {
    CHECK_EQ(kf_vocab_rows({ opd_f32({ 4, 9 }), opd_f32({ 128, 2 }), opd_f32({ 7 }) }),
             (int64_t)128);
    CHECK_EQ(kf_vocab_rows({}), (int64_t)0);
}

/* ================================================================== the file */

TEST(a_fixture_survives_a_save_and_a_load) {
    Kfixture fx;
    fx.source = "/models/thing.rad";
    fx.oracle = "libref@0.1.0";
    fx.recorded = "2026-01-02";

    KfCase c;
    c.op = "rmsnorm";
    c.seed = 0xFEEDFACEDEADBEEFull;
    c.m = 517;
    c.band = "M in (64, 512]";
    c.params.push_back(KfParam{ "M", RAD_P_INT, 517, 0, "" });
    c.params.push_back(KfParam{ "eps", RAD_P_F64, 0, 1e-6, "" });
    c.params.push_back(KfParam{ "dtype", RAD_P_STR, 0, 0, "bf16" });

    KfOperand in = opd_f32({ 517, 4096 }, RAD_OPD_IN, KF_FILL_GATE_CUMSUM);
    in.fill_chunk = 64;
    in.is_index = true;
    in.idx_unique = true;
    in.idx_cu = true;
    in.idx_prefix = true;
    in.idx_history = true;
    in.idx_max = 99;
    in.idx_const = 12;
    c.opd.push_back(in);

    KfOperand params = opd_f32({ 517, (int64_t)(sizeof(RadSampleParams) / 4) }, RAD_OPD_IN,
                               KF_FILL_SAMPLE_PARAMS);
    params.fill_chunk = 64;
    params.idx_max = 517;
    c.opd.push_back(params);

    KfOperand out = opd_f32({ 517, 4096 }, RAD_OPD_OUT);
    out.dtype = RAD_BF16;
    out.have_ref = true;
    out.ref_n = 517 * 4096;
    out.ref_sumsq = 1234.5;
    out.ref_nonfinite = 3;
    out.sample.assign(KF_SAMPLES, 0.0f);
    for (int i = 0; i < KF_SAMPLES; ++i) out.sample[(size_t)i] = (float)i * 0.5f;
    out.sketch.assign(KF_SKETCH, 0.0);
    for (int j = 0; j < KF_SKETCH; ++j) out.sketch[(size_t)j] = -3.25 * j;
    c.opd.push_back(out);

    KfOperand gone = opd_f32({ 1 });
    gone.absent = true;
    c.opd.push_back(gone);

    c.opd.push_back(fp8_weight(256, 384, { "codes", "scale" }));
    fx.cases.push_back(c);

    const std::string path = radtest::tmp_path("roundtrip.rkb");
    CHECK_OK(fx.save(path));

    Kfixture rd;
    CHECK_OK(rd.load(path));
    CHECK_EQ(rd.version, fx.version);
    CHECK_EQ(rd.source, fx.source);
    CHECK_EQ(rd.oracle, fx.oracle);
    CHECK_EQ(rd.recorded, fx.recorded);
    REQUIRE_EQ(rd.cases.size(), (size_t)1);

    const KfCase& g = rd.cases[0];
    CHECK_EQ(g.op, c.op);
    CHECK_EQ(g.seed, c.seed);
    CHECK_EQ(g.m, c.m);
    CHECK_EQ(g.band, c.band);
    CHECK_EQ(g.params.size(), c.params.size());
    for (size_t i = 0; i < g.params.size() && i < c.params.size(); ++i) {
        CHECK_EQ(g.params[i].key, c.params[i].key);
        CHECK_EQ(g.params[i].kind, c.params[i].kind);
        CHECK_EQ(g.params[i].i, c.params[i].i);
        CHECK_EQ(g.params[i].d, c.params[i].d);
        CHECK_EQ(g.params[i].s, c.params[i].s);
    }
    CHECK_EQ(g.opd.size(), c.opd.size());
    for (size_t i = 0; i < g.opd.size() && i < c.opd.size(); ++i) {
        const KfOperand& a = g.opd[i];
        const KfOperand& b = c.opd[i];
        CHECK_EQ(a.role, b.role);
        CHECK_EQ(a.dtype, b.dtype);
        CHECK(a.shape == b.shape);
        CHECK_EQ(a.absent, b.absent);
        CHECK_EQ(a.fill, b.fill);
        CHECK_EQ(a.is_index, b.is_index);
        CHECK_EQ(a.idx_unique, b.idx_unique);
        CHECK_EQ(a.idx_cu, b.idx_cu);
        CHECK_EQ(a.idx_prefix, b.idx_prefix);
        CHECK_EQ(a.idx_history, b.idx_history);
        CHECK_EQ(a.idx_max, b.idx_max);
        CHECK_EQ(a.idx_const, b.idx_const);
        CHECK_EQ(a.fill_chunk, b.fill_chunk);
        CHECK_EQ(a.have_ref, b.have_ref);
        CHECK_EQ(a.ref_n, b.ref_n);
        CHECK_EQ(a.ref_sumsq, b.ref_sumsq);
        CHECK_EQ(a.ref_nonfinite, b.ref_nonfinite);
        CHECK(a.sample == b.sample);
        CHECK(a.sketch == b.sketch);
        CHECK_EQ(a.undrawn, false);
        CHECK_EQ(a.has_enc, b.has_enc);
        CHECK(rad_enc_equal(&a.enc, &b.enc));
        CHECK(a.sel == b.sel);
        CHECK(a.sel_rows == b.sel_rows);
        CHECK(a.sel_cols == b.sel_cols);
    }
}

TEST(a_fixture_that_is_not_there_is_not_found) {
    Kfixture fx;
    CHECK_EQ(fx.load(radtest::tmp_path("no_such_file.rkb")), RAD_E_NOTFOUND);
}

/* An unrecognised file is refused; misreading a changed struct would yield confident wrong numbers
 * instead of an error. */
TEST(a_fixture_the_reader_does_not_understand_is_refused) {
    Kfixture fx;
    fx.source = "x";
    fx.cases.push_back(KfCase{});
    const std::string path = radtest::tmp_path("bad.rkb");
    CHECK_OK(fx.save(path));

    const char wrong_magic[4] = { 'R', 'K', 'B', '9' };
    CHECK(poke(path, 0, wrong_magic, 4));
    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);

    const char good_magic[4] = { 'R', 'K', 'B', '1' };
    CHECK(poke(path, 0, good_magic, 4));
    const uint32_t wrong_version = 99;
    CHECK(poke(path, 4, &wrong_version, 4));
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);

    /* Sample positions derive from the count, so a file recorded at another count would compare
     * every sample against the wrong element. */
    const uint32_t version = KF_VERSION;
    CHECK(poke(path, 4, &version, 4));
    const uint32_t wrong_samples = 512;
    CHECK(poke(path, 8, &wrong_samples, 4));
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
}

/* A file whose weights carry no encoding (version 3) is refused. Kernel layout hooks cannot take
 * such operands, and a replay would hand every relayout a plain plane instead of the encoding the
 * reference used. */
TEST(a_fixture_without_encodings_is_refused) {
    Kfixture fx;
    fx.source = "x";
    fx.cases.push_back(KfCase{});
    const std::string path = radtest::tmp_path("v3.rkb");
    CHECK_OK(fx.save(path));
    const uint32_t v3 = 3;
    REQUIRE(poke(path, 4, &v3, 4));
    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
}

TEST(a_truncated_fixture_is_refused) {
    Kfixture fx;
    fx.source = "x";
    KfCase c;
    c.op = "rmsnorm";
    c.opd.push_back(opd_f32({ 4 }));
    fx.cases.push_back(c);
    const std::string path = radtest::tmp_path("short.rkb");
    CHECK_OK(fx.save(path));

    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fclose(f);
    CHECK(n > 24);
    CHECK_EQ(truncate(path.c_str(), n - 8), 0);

    Kfixture rd;
    CHECK_EQ(rd.load(path), RAD_E_FORMAT);
}

/* ================================================================== a weight's planes */

namespace {

/* Select every plane of `e` over a [rows, cols] weight, as a declaration taking all of them. */
KfOperand all_planes(const RadEncoding& e, int64_t rows, int64_t cols) {
    KfOperand d;
    d.role = RAD_OPD_WEIGHT;
    d.has_enc = true;
    d.enc = e;
    d.dtype = RAD_F32;
    d.shape = { rows, cols };
    select_planes(&d, rows, cols);
    return d;
}

}  /* namespace */

/* A drawn plane holds only what its role allows: a code past its codebook reads past a table, a
 * permutation repeating a column is a different weight per reader, and an exponent scale over the
 * full range overflows the reference. */
TEST(a_weights_planes_are_drawn_inside_what_each_can_hold) {
    const int64_t R = 8, C = 64;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "perm");
    rad_enc_add_plane(&e, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_E8M0, 1, 32);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 10);
    rad_enc_add_table(&e, "t.perm", RAD_I32, 1, C);
    REQUIRE(rad_enc_valid(&e));
    const KfOperand d = all_planes(e, R, C);

    std::vector<std::vector<uint8_t>> a, b, x;
    kf_draw_planes(d, 7, &a);
    kf_draw_planes(d, 7, &b);
    kf_draw_planes(d, 8, &x);
    REQUIRE_EQ(a.size(), (size_t)4);
    CHECK(a == b);                                   /* a seed determines the planes */
    CHECK(a[0] != x[0]);
    for (int64_t i = 0; i < R * C; ++i) CHECK(a[0][(size_t)i] < 10);
    for (uint8_t s8 : a[1]) CHECK(s8 >= 123 && s8 <= 131);
    std::vector<int32_t> perm((size_t)C);
    std::memcpy(perm.data(), a[3].data(), (size_t)C * 4);
    std::sort(perm.begin(), perm.end());
    for (int64_t j = 0; j < C; ++j) CHECK_EQ(perm[(size_t)j], (int32_t)j);
}

/* Every plane dtype a weight's codes can take draws values: finite, and not all zero. The draw
 * stores floats through rad_store_f32 and bit patterns through rad_store_bits, so a dtype the
 * first does not encode (the sub-byte floats) would come back zeroed, and every case reading such
 * a weight would compare zeros with zeros. */
TEST(every_code_dtype_a_weight_can_hold_is_drawn_finite_and_nonzero) {
    struct Row { uint32_t dt; const char* name; };
    const Row rows[] = {
        { RAD_F32, "f32" }, { RAD_BF16, "bf16" }, { RAD_F16, "f16" },
        { RAD_F8E4M3, "f8e4m3" }, { RAD_F8E5M2, "f8e5m2" },
        { RAD_FP4E2M1, "fp4e2m1" }, { RAD_FP6E2M3, "fp6e2m3" }, { RAD_FP6E3M2, "fp6e3m2" },
        { RAD_I8, "i8" }, { RAD_U8, "u8" }, { RAD_I4, "i4" }, { RAD_U4, "u4" },
        { RAD_I2, "i2" }, { RAD_U2, "u2" }, { RAD_E8M0, "e8m0" },
    };
    const int64_t R = 4, C = 64;
    for (const Row& r : rows) {
        RadEncoding e;
        rad_enc_clear(&e);
        rad_enc_copy_str(e.scheme, "plain");
        rad_enc_add_plane(&e, "codes", r.dt, 1, 1);
        REQUIRE(rad_enc_valid(&e));
        const KfOperand d = all_planes(e, R, C);
        std::vector<std::vector<uint8_t>> pl;
        kf_draw_planes(d, 11, &pl);
        REQUIRE_EQ(pl.size(), (size_t)1);
        int64_t nonzero = 0, nonfinite = 0;
        for (int64_t i = 0; i < R * C; ++i) {
            const float v = rad_load_f32(pl[0].data(), r.dt, i);
            nonzero += v != 0.0f;
            nonfinite += !std::isfinite(v);
        }
        if (nonzero < R * C / 2 || nonfinite)
            std::fprintf(stderr, "    %s: %lld of %lld drawn values non-zero, %lld non-finite\n",
                         r.name, (long long)nonzero, (long long)(R * C), (long long)nonfinite);
        CHECK(nonzero >= R * C / 2);
        CHECK_EQ(nonfinite, (int64_t)0);
    }
}

/* The reference reads what the planes decode to: codes times their block's scale for the whole
 * encoding, or one plane's own values, widened when the operand is wider. */
TEST(the_reference_reads_what_the_planes_decode_to) {
    const int64_t R = 256, C = 384;
    KfOperand both = fp8_weight(R, C, { "codes", "scale" });
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(both, 3, &pl);
    std::vector<uint8_t> out;
    std::string why;
    REQUIRE_EQ(kf_planes_logical(both, pl, &out, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)(R * C * 4));
    const float* f = (const float*)out.data();
    for (int64_t r : { (int64_t)0, (int64_t)130, R - 1 })
        for (int64_t c : { (int64_t)0, (int64_t)200, C - 1 }) {
            const float code = rad_load_f32(pl[0].data(), RAD_F8E4M3, r * C + c);
            const float scale = rad_load_f32(pl[1].data(), RAD_BF16, (r / 128) * 3 + c / 128);
            CHECK_EQ(f[r * C + c], code * scale);
        }

    KfOperand sc = fp8_weight(R, C, { "scale" });
    sc.dtype = RAD_F32;                              /* a reference reading the scales in f32 */
    std::vector<std::vector<uint8_t>> sp;
    kf_draw_planes(sc, 3, &sp);
    REQUIRE_EQ(kf_planes_logical(sc, sp, &out, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)(2 * 3 * 4));
    for (int i = 0; i < 6; ++i)
        CHECK_EQ(((const float*)out.data())[i], rad_load_f32(sp[0].data(), RAD_BF16, i));
}

/* A scale selected with the level above it but without the codes, as in the scale view of a
 * w4nl64a8h expert: an E4M3 scale per (row, 64 of K) under one f32 per weight. The reference
 * reads scale times level in the scale plane's extents. Refused: a chain not starting at its first
 * level, a level finer than the scale it multiplies, and a reader expecting another count. */
TEST(the_reference_reads_a_scale_times_the_level_above_it) {
    const int64_t R = 32, C = 256;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "fwht128");
    rad_enc_add_plane(&e, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F8E4M3, 1, 64);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 16);
    REQUIRE(rad_enc_valid(&e));
    auto view = [&](const RadEncoding& enc, std::vector<const char*> roles) {
        KfOperand d;
        d.role = RAD_OPD_WEIGHT;
        d.has_enc = true;
        d.enc = enc;
        d.dtype = RAD_F32;
        select_planes(&d, R, C, roles);
        d.shape = { d.sel_rows[0], d.sel_cols[0] };
        return d;
    };
    const KfOperand d = view(e, { "scale", "scale.1" });
    REQUIRE_EQ(numel(d), R * C / 64);
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(d, 9, &pl);
    REQUIRE_EQ(pl.size(), (size_t)2);
    std::vector<uint8_t> out;
    std::string why;
    REQUIRE_EQ(kf_planes_logical(d, pl, &out, &why), RAD_OK);
    const float level = rad_load_f32(pl[1].data(), RAD_F32, 0);
    const float* f = (const float*)out.data();
    for (int64_t i = 0; i < R * C / 64; ++i)
        CHECK_EQ(f[i], rad_load_f32(pl[0].data(), RAD_F8E4M3, i) * level);

    CHECK_EQ(kf_planes_logical(view(e, { "scale.1", "scale" }), pl, &out, &why), RAD_E_SHAPE);
    KfOperand more = d;
    more.shape = { R, C / 32 };
    CHECK_EQ(kf_planes_logical(more, pl, &out, &why), RAD_E_SHAPE);
    RadEncoding fine = e;
    fine.plane[rad_enc_find(&e, "scale.1")].block[1] = 32;
    const KfOperand df = view(fine, { "scale", "scale.1" });
    std::vector<std::vector<uint8_t>> pf;
    kf_draw_planes(df, 9, &pf);
    CHECK_EQ(kf_planes_logical(df, pf, &out, &why), RAD_E_SHAPE);
    CHECK(why.find("does not cover") != std::string::npos);
}

namespace {

int lay_identity(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
                 RadLayout*) { return RAD_E_UNSUPPORTED; }
int lay_refuse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int,
               RadLayout*) { return RAD_E_DTYPE; }
int lay_reverse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor* pl,
                int n, RadLayout* out) {
    if (n != 1) return RAD_E_SHAPE;
    out->dtype = pl[0].dtype;
    out->bytes = rad_dtype_bytes(pl[0].dtype, pl[0].shape[0] * pl[0].shape[1]);
    return RAD_OK;
}
int relay_reverse(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor* pl,
                  int, void* dst, int64_t bytes) {
    const uint8_t* s = (const uint8_t*)pl[0].data;
    for (int64_t i = 0; i < bytes; ++i) ((uint8_t*)dst)[i] = s[bytes - 1 - i];
    return RAD_OK;
}

}  /* namespace */

/* A kernel is handed what its own layout hooks make of the planes. A hook has three distinct
 * answers: read as stored, not this encoding, or these bytes. */
TEST(a_kernel_is_handed_what_its_own_layout_makes) {
    const KfOperand d = fp8_weight(128, 256, { "codes" });
    std::vector<std::vector<uint8_t>> pl;
    kf_draw_planes(d, 5, &pl);
    RadKernelInfo ki{};
    RadLayout lay{};
    std::vector<uint8_t> out;

    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_UNSUPPORTED);
    ki.layout = lay_identity;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_UNSUPPORTED);
    ki.layout = lay_refuse;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_DTYPE);
    ki.layout = lay_reverse;
    CHECK_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_E_STATE);  /* no relayout */
    ki.relayout = relay_reverse;
    REQUIRE_EQ(kf_lay_weight(&ki, nullptr, 0, 0, d, pl, &lay, &out), RAD_OK);
    REQUIRE_EQ(out.size(), pl[0].size());
    CHECK(std::equal(out.begin(), out.end(), pl[0].rbegin()));
    CHECK_EQ(lay.dtype, (uint32_t)RAD_F8E4M3);
}

/* ================================================================== the artifact in the tree
 *
 * The checked-in data/kernels.rkb is the replay's entire coverage, so it is validated, not trusted.
 * An unreadable fixture, lost cases or an undrawable dtype would otherwise make rad-kbench
 * silently check less than the fixture records.
 */
namespace {

/* Where the build copies the checked-in fixture `leaf`. */
std::string checked_in_path(const char* leaf) { return home() + "/" + leaf; }

/* RAD_OK; RAD_E_NOTFOUND when the file is absent, which the case reports as a missing artifact;
 * or the load's error, a failed check here: a stale recording means the replay checks nothing. */
int load_checked_in(const char* leaf, Kfixture* fx) {
    const std::string path = checked_in_path(leaf);
    const int rc = fx->load(path);
    if (rc == RAD_E_NOTFOUND) return rc;
    if (rc != RAD_OK)
        std::fprintf(stderr, "    %s does not load (%s) -- record it again with rad-kbench -m "
                             "<model.rad> --max-cases 1 --record\n", path.c_str(),
                     rad_strerror(rc));
    CHECK_OK(rc);
    return rc;
}

}  /* namespace */

TEST(the_checked_in_fixtures_load_and_describe_real_cases) {
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        const int rc = load_checked_in(leaf, &fx);
        if (rc == RAD_E_NOTFOUND) ARTIFACT_MISSING(checked_in_path(leaf));
        if (rc != RAD_OK) continue;
        /* The version this tree records, or one it still reads: 4 (no sketch, weight tables
         * recorded undrawn) or 5 (no sampler laws). A fixture recorded by an older tool fails here
         * rather than as a page of kernel failures. */
        CHECK(fx.version >= 4 && fx.version <= KF_VERSION);
        CHECK(!fx.cases.empty());
        CHECK(!fx.source.empty());
        CHECK(!fx.oracle.empty());
        int with_ref = 0, with_enc = 0;
        for (const KfCase& c : fx.cases) {
            CHECK(!c.op.empty());
            CHECK(!c.opd.empty());
            for (const KfOperand& d : c.opd) {
                if (d.absent) continue;
                if (d.role == RAD_OPD_WEIGHT) {
                    /* every weight a recording draws carries its encoding */
                    CHECK(d.has_enc);
                    if (d.has_enc) ++with_enc;
                }
                CHECK(!d.shape.empty());
                /* Only what a kernel writes has a reference; a reference on an input would be
                 * compared with bytes no kernel produced. */
                if (d.have_ref) CHECK(d.role == RAD_OPD_OUT || d.role == RAD_OPD_INOUT);
                /* A version-5 recording drew every weight table from an encoding. */
                if (fx.version >= 5 && d.role == RAD_OPD_WTAB) CHECK(d.has_enc && !d.undrawn);
                if (d.have_ref) {
                    ++with_ref;
                    CHECK(d.ref_n > 0);
                    CHECK((int64_t)d.sample.size() ==
                          (d.ref_n < KF_SAMPLES ? d.ref_n : (int64_t)KF_SAMPLES));
                    if (fx.version >= 5) CHECK_EQ(d.sketch.size(), (size_t)KF_SKETCH);
                }
            }
        }
        CHECK(with_ref > 0);
        CHECK(with_enc > 0);
    }
}

/* Every dtype a recorded case uses must be writable and readable by the harness. A plain operand
 * goes through rad_narrow and an output through rad_widen: a dtype the first misses reaches the
 * kernel as zeros, one the second misses compares zeros on both sides and passes. A weight is
 * drawn as its encoding's planes and decoded instead, so each op, weight format and operand's
 * encoding and plane selection is drawn here over a 256 x 256 weight (the draw costs per element,
 * and what comes out as zeros is a plane's dtype, not its size) and must decode to something other
 * than zeros. None of these raises an error, so the fixtures are swept here. */
TEST(the_checked_in_fixtures_use_no_dtype_the_harness_cannot_draw_or_read) {
    const float probe[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        const int rc = load_checked_in(leaf, &fx);
        if (rc == RAD_E_NOTFOUND) ARTIFACT_MISSING(checked_in_path(leaf));
        if (rc != RAD_OK) continue;
        std::set<uint32_t> seen;
        std::map<std::string, std::pair<const KfOperand*, size_t>> weights;
        for (const KfCase& c : fx.cases) {
            std::string variant;
            for (const KfParam& p : c.params)
                if (p.key == "dtype" && p.kind == RAD_P_STR) variant = "[" + p.s + "]";
            for (size_t k = 0; k < c.opd.size(); ++k) {
                const KfOperand& d = c.opd[k];
                if (d.absent) continue;
                if (!d.has_enc) { seen.insert(d.dtype); continue; }
                if (d.role == RAD_OPD_WTAB || d.undrawn) continue;   /* drawn per expert */
                weights.emplace(c.op + variant + "/" + std::to_string(k), std::make_pair(&d, k));
            }
        }

        for (uint32_t dt : seen) {
            /* Plugin-private dtypes are opaque to the harness and excluded. */
            if (dt >= RAD_DT_PLUGIN_BASE) continue;
            const NarrowWiden t = narrow_widen(probe, dt, 4);
            if (!t.wrote || !t.read)
                std::fprintf(stderr, "    %s: dtype %s (%u) narrow=%s widen=%s\n", leaf,
                             rad_dtype_name(dt), dt, t.wrote ? "yes" : "NO", t.read ? "yes" : "NO");
            CHECK(t.wrote);
            CHECK(t.read);
        }
        for (const auto& [what, w] : weights) {
            KfOperand t = *w.first;
            t.sel_rows.clear();
            t.sel_cols.clear();
            for (int i : t.sel) {
                int64_t pr = 0, pc = 0;
                rad_enc_plane_dims(&t.enc.plane[i], 256, 256, &pr, &pc);
                t.sel_rows.push_back(pr);
                t.sel_cols.push_back(pc);
            }
            t.shape = t.sel.size() == 1 ? std::vector<int64_t>{ t.sel_rows[0], t.sel_cols[0] }
                                        : std::vector<int64_t>{ 256, 256 };
            std::vector<std::vector<uint8_t>> planes;
            kf_draw_planes(t, kf_weight_seed(t, w.second), &planes);
            std::vector<uint8_t> logical;
            std::string why;
            const int lrc = kf_planes_logical(t, planes, &logical, &why);
            bool any = false;
            for (uint8_t b : logical) any = any || b != 0;
            if (lrc != RAD_OK || !any)
                std::fprintf(stderr, "    %s: the weight %s (%s) decodes to %s\n", leaf,
                             what.c_str(), rad_dtype_name(w.first->dtype),
                             lrc != RAD_OK ? why.c_str() : "zeros");
            CHECK_EQ(lrc, RAD_OK);
            CHECK(any);
        }
    }
}

/* Every output of every op in a checked-in fixture has a case whose reference is not identically
 * zero: a zero reference only checks that a kernel wrote nothing. A version-4 fixture recorded its
 * weight tables undrawn, so its grouped-GEMM references are zero; those ops are named and wait for
 * the next recording. */
TEST(every_checked_in_output_has_a_case_with_a_nonzero_reference) {
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        const int rc = load_checked_in(leaf, &fx);
        if (rc == RAD_E_NOTFOUND) ARTIFACT_MISSING(checked_in_path(leaf));
        if (rc != RAD_OK) continue;
        /* "op[dtype]/operand index" -> seen a non-zero one. Keyed by the case's dtype too: one
         * op serves several weight formats, and the formats' cases are drawn differently. */
        std::map<std::string, bool> nonzero;
        std::set<std::string> undrawn_ops;
        for (const KfCase& c : fx.cases) {
            bool undrawn = false;
            for (const KfOperand& d : c.opd) undrawn = undrawn || d.undrawn;
            if (undrawn) { undrawn_ops.insert(c.op); continue; }
            std::string variant;
            for (const KfParam& p : c.params)
                if (p.key == "dtype" && p.kind == RAD_P_STR) variant = "[" + p.s + "]";
            for (size_t k = 0; k < c.opd.size(); ++k) {
                const KfOperand& d = c.opd[k];
                if (d.absent || !d.have_ref) continue;
                bool& seen = nonzero[c.op + variant + "/" + std::to_string(k)];
                seen = seen || d.ref_sumsq > 0 || d.ref_nonfinite > 0;
            }
        }
        for (const std::string& op : undrawn_ops)
            radtest::skip("%s: %s was recorded with undrawn weight tables -- record the fixture "
                          "again", leaf, op.c_str());
        for (const auto& [what, seen] : nonzero) {
            if (!seen) std::fprintf(stderr, "    %s: every reference of %s is zero\n", leaf,
                                    what.c_str());
            CHECK(seen);
        }
    }
}

/* ================================================================== drawing, threaded */

/* kf_draw hands each RAD_FILL_BLOCK block of a large plain operand to its own thread. The bytes must
 * be those of one thread drawing the blocks in order and narrowing each, or every recording would
 * depend on the recording machine's core count. */
TEST(a_threaded_draw_is_byte_identical_to_one_thread) {
    for (uint32_t dt : { (uint32_t)RAD_BF16, (uint32_t)RAD_F8E4M3, (uint32_t)RAD_F32 }) {
        for (int role : { (int)RAD_OPD_IN, (int)RAD_OPD_WEIGHT, (int)RAD_OPD_WTAB }) {
            KfOperand d;
            d.role = role;
            d.dtype = dt;
            d.shape = { 3, RAD_FILL_BLOCK + 517 };
            std::vector<uint8_t> got;
            kf_draw({ d }, 0, 1, 0, 0xABCDEFull, &got);
            const bool weight = role != RAD_OPD_IN;
            const uint64_t seed = weight ? kf_weight_seed(d, 0) : 0xABCDEFull;
            const float sigma = weight ? 0.02f : 1.0f;
            const int64_t n = numel(d);
            std::vector<uint8_t> want((size_t)rad_dtype_bytes(dt, n));
            std::vector<float> blk;
            for (int64_t lo = 0; lo < n; lo += RAD_FILL_BLOCK) {
                const int64_t len = std::min<int64_t>(RAD_FILL_BLOCK, n - lo);
                blk.resize((size_t)len);
                rad_fill_normal_block(blk.data(), len, seed, lo / RAD_FILL_BLOCK, sigma);
                rad_narrow(blk.data(), dt, len, want.data() + rad_dtype_bytes(dt, lo));
            }
            if (got != want)
                std::fprintf(stderr, "    %s role %d differs from the serial draw\n",
                             rad_dtype_name(dt), role);
            CHECK(got == want);
        }
    }
}

/* ================================================================== weight tables */

/* A weight table (a MoE layer's experts) left as zeros makes every grouped-GEMM reference zero,
 * which a kernel writing nothing passes. It is drawn like a weight: from its identity, not the
 * case. */
TEST(a_weight_table_is_drawn_like_a_weight) {
    KfOperand t;
    t.role = RAD_OPD_WTAB;
    t.dtype = RAD_F8E4M3;
    t.shape = { 4, 32, 64 };
    std::vector<uint8_t> a, b;
    kf_draw({ t }, 0, 1, 0, 111, &a);
    kf_draw({ t }, 0, 512, 0, 999, &b);
    CHECK(a == b);
    size_t nz = 0;
    for (uint8_t x : a) nz += x != 0;
    CHECK(nz > a.size() / 2);
    /* A version-4 recording drew its tables as zeros and its references were computed on them,
     * so a replay of it draws the same zeros. */
    KfOperand old = t;
    old.undrawn = true;
    std::vector<uint8_t> z;
    kf_draw({ old }, 0, 1, 0, 111, &z);
    CHECK_EQ(z.size(), a.size());
    for (uint8_t x : z) CHECK_EQ((int)x, 0);
}

/* With the encoding a recording takes from the run's first expert, each expert's planes are drawn
 * from the table's identity and its own index: codes and scales as a quantiser writes them, the
 * scales positive, and no two experts alike. */
TEST(a_weight_tables_experts_are_drawn_from_its_encoding) {
    KfOperand codes = fp8_weight(256, 256, { "codes" });
    codes.role = RAD_OPD_WTAB;
    codes.shape.insert(codes.shape.begin(), 3);
    KfOperand scale = fp8_weight(256, 256, { "scale" });
    scale.role = RAD_OPD_WTAB;
    scale.shape.insert(scale.shape.begin(), 3);
    std::vector<uint8_t> cb, sb;
    std::string why;
    REQUIRE_EQ(kf_draw_table(codes, 0, &cb, &why), RAD_OK);
    REQUIRE_EQ(kf_draw_table(scale, 1, &sb, &why), RAD_OK);
    REQUIRE_EQ(cb.size(), (size_t)3 * 256 * 256);
    REQUIRE_EQ(sb.size(), (size_t)3 * 2 * 2 * 2);
    std::vector<float> s;
    rad_widen(sb.data(), RAD_BF16, 12, &s);
    for (float x : s) CHECK(x > 0.0f);
    CHECK(std::memcmp(cb.data(), cb.data() + 65536, 65536) != 0);
    std::vector<uint8_t> via_draw;
    kf_draw({ codes }, 0, 1, 0, 5, &via_draw);
    CHECK(via_draw == cb);
}

/* ================================================================== routing */

/* moe_scatter's offsets are a prefix array: non-decreasing from 0 to the row count. A uniform draw
 * is a malformed histogram that two correct grouped GEMMs may read differently. */
TEST(routing_offsets_are_drawn_as_a_valid_histogram) {
    KfOperand d = opd_f32({ 9 });
    d.dtype = RAD_I32;
    d.is_index = true;
    d.idx_prefix = true;
    d.idx_max = 41;
    const std::vector<float> v = draw_f32({ d }, 0, 1, 0x77);
    REQUIRE_EQ(v.size(), (size_t)9);
    CHECK_EQ(v[0], 0.0f);
    CHECK_EQ(v[8], 40.0f);
    std::set<float> steps;
    for (size_t i = 1; i < v.size(); ++i) {
        CHECK(v[i] >= v[i - 1]);
        steps.insert(v[i] - v[i - 1]);
    }
    CHECK(steps.size() > 1);              /* buckets of different sizes, not an even split */
}

namespace {

std::string libref_path() { return home() + "/kernels/libref.so"; }

/* The registry with libref loaded, or null when it does not load. */
Registry* libref_registry() {
    static Registry reg;
    static int state = 0;
    if (state == 0) {
        state = -1;
        if (reg.load_plugin(libref_path(), 0) >= 0 && reg.check_complete() >= 0) state = 1;
    }
    return state == 1 ? &reg : nullptr;
}

/* libref's description of `op` at `g`. */
bool ref_describe(const std::string& op, const Geometry& g, std::vector<OpdShape>* sh,
                  const RadOpSchema** schema) {
    Registry* reg = libref_registry();
    if (!reg) return false;
    *schema = reg->schema(op);
    const KernelRow* row = reg->select_in("libref", op, g, RAD_DOMAIN_HOST);
    if (!*schema || !row) return false;
    return rad_operand_shapes_hook(row->info, g, (*schema)->n_operands, sh) == RAD_OK;
}

int operand_index(const RadOpSchema* s, const char* name) {
    for (int i = 0; i < s->n_operands; ++i)
        if (s->operands[i].name && std::strcmp(s->operands[i].name, name) == 0) return i;
    return -1;
}

}  /* namespace */

/* libref describes the grouped GEMMs' routing with those laws, so a recording draws one
 * moe_scatter could have produced: `sorted_tok` a permutation of [0, M*top_k) and `expert_offset`
 * a prefix array ending at M*top_k. */
TEST(the_reference_describes_a_valid_routing_for_the_grouped_gemms) {
    if (!libref_registry()) ARTIFACT_MISSING(libref_path() + " (does not load)");
    struct C { const char* op; const char* dtype; };
    for (const C& cs : { C{ "moe_gemm", "bf16" }, C{ "moe_gemm_q", "fp8a8" },
                         C{ "moe_gemm_q_mod4", "fp8a8" } }) {
        Geometry g;
        g.set_i("M", 5); g.set_i("N", 128); g.set_i("K", 128); g.set_i("n_expert", 8);
        g.set_i("top_k", 3); g.set_i("group", 128); g.set_s("dtype", cs.dtype);
        std::vector<OpdShape> sh;
        const RadOpSchema* schema = nullptr;
        REQUIRE(ref_describe(cs.op, g, &sh, &schema));
        const int st = operand_index(schema, "sorted_tok");
        const int eo = operand_index(schema, "expert_offset");
        REQUIRE(st >= 0 && eo >= 0);
        CHECK(sh[(size_t)st].idx_unique);
        CHECK(sh[(size_t)eo].idx_prefix);
        std::vector<KfOperand> opd(sh.size());
        for (size_t k = 0; k < sh.size(); ++k) {
            opd[k].role = schema->operands[k].role;
            opd[k].dtype = sh[k].dtype;
            opd[k].shape = sh[k].shape;
            opd[k].absent = sh[k].absent;
            opd[k].is_index = sh[k].is_index;
            opd[k].idx_unique = sh[k].idx_unique;
            opd[k].idx_prefix = sh[k].idx_prefix;
            opd[k].idx_max = sh[k].idx_max;
        }
        const std::vector<float> tok = draw_f32(opd, (size_t)st, 5, 21);
        std::vector<float> sorted = tok;
        std::sort(sorted.begin(), sorted.end());
        REQUIRE_EQ(sorted.size(), (size_t)15);
        for (size_t i = 0; i < sorted.size(); ++i) CHECK_EQ(sorted[i], (float)i);
        const std::vector<float> off = draw_f32(opd, (size_t)eo, 5, 22);
        REQUIRE_EQ(off.size(), (size_t)9);
        CHECK_EQ(off.front(), 0.0f);
        CHECK_EQ(off.back(), 15.0f);
        for (size_t i = 1; i < off.size(); ++i) CHECK(off[i] >= off[i - 1]);
    }
}

/* Under state_form=anchor the recurrence reads six state_idx columns a sequence (docs/OPS.md):
 * the anchor's three and the factor block's three. libref describes a table at least six wide,
 * whatever the step's length, every entry a distinct slot inside the state, and one accepted token
 * to replay, so a recording draws an input libr4d reads inside its operands and the same way as the
 * host libraries. A step longer than the factor block holds is not described. */
TEST(the_reference_describes_six_distinct_state_columns_for_the_anchor_form) {
    if (!libref_registry()) ARTIFACT_MISSING(libref_path() + " (does not load)");
    for (const char* op : { "gdn_recurrent_update", "gdn_conv_recurrent_update" })
        for (int64_t q_len : { 1, 3, 8, 16, 17 }) {
            Geometry g;
            g.set_i("q_len", q_len); g.set_i("n_seq", 2); g.set_i("head_k", 128);
            g.set_i("head_v", 128); g.set_i("n_head_k", 1); g.set_i("n_head_v", 2);
            g.set_i("conv_width", 4); g.set_s("act", "silu"); g.set_s("dtype", "bf16");
            g.set_s("state_form", "anchor");
            std::vector<OpdShape> sh;
            const RadOpSchema* schema = nullptr;
            const bool described = ref_describe(op, g, &sh, &schema);
            if (q_len > 16) {                 /* past the sixteen tokens a 128 x 128 block holds */
                CHECK(!described);
                continue;
            }
            REQUIRE(described);
            const int st = operand_index(schema, "state");
            const int si = operand_index(schema, "state_idx");
            const int na = operand_index(schema, "num_accepted");
            REQUIRE(st >= 0 && si >= 0 && na >= 0);
            const int64_t cols = std::max<int64_t>(q_len, 6);
            REQUIRE_EQ(sh[(size_t)si].shape.size(), (size_t)2);
            CHECK_EQ(sh[(size_t)si].shape[0], (int64_t)2);
            CHECK_EQ(sh[(size_t)si].shape[1], cols);
            CHECK_EQ(sh[(size_t)st].shape[0], 2 * cols);
            CHECK(sh[(size_t)si].idx_unique);
            CHECK_EQ(sh[(size_t)na].idx_const, (int64_t)1);
            std::vector<KfOperand> opd(sh.size());
            for (size_t k = 0; k < sh.size(); ++k) {
                opd[k].role = schema->operands[k].role;
                opd[k].dtype = sh[k].dtype;
                opd[k].shape = sh[k].shape;
                opd[k].absent = sh[k].absent;
                opd[k].is_index = sh[k].is_index;
                opd[k].idx_unique = sh[k].idx_unique;
                opd[k].idx_cu = sh[k].idx_cu;
                opd[k].idx_max = sh[k].idx_max;
                opd[k].idx_const = sh[k].idx_const;
            }
            const std::vector<float> idx = draw_f32(opd, (size_t)si, 2 * q_len, 31);
            std::vector<float> sorted = idx;
            std::sort(sorted.begin(), sorted.end());
            REQUIRE_EQ(sorted.size(), (size_t)(2 * cols));
            for (size_t i = 0; i < sorted.size(); ++i) CHECK_EQ(sorted[i], (float)i);
            for (float v : draw_f32(opd, (size_t)na, 2 * q_len, 32)) CHECK_EQ(v, 1.0f);
        }
}

/* ================================================================== the sampler's requests
 *
 * A sampler stage acts on a row only when the row's flag is set and its window or mask row is
 * real, so a parameter plane drawn as numbers leaves most stages comparing two copies of their
 * input. RAD_FILL_SAMPLE_PARAMS draws requests and RAD_OPD_F_IDX_HISTORY a history that repeats;
 * these cases pin both laws, and that libref's stages then do work on most rows.
 */
namespace {

constexpr int64_t kParamWords = (int64_t)(sizeof(RadSampleParams) / 4);

/* `rows` request rows over a history plane `hist` wide and `mask_rows` bitmask rows. */
std::vector<RadSampleParams> draw_requests(int64_t rows, int64_t hist, int64_t mask_rows,
                                           uint64_t seed) {
    KfOperand d = opd_f32({ rows, kParamWords }, RAD_OPD_IN, KF_FILL_SAMPLE_PARAMS);
    d.fill_chunk = hist;
    d.idx_max = mask_rows;
    std::vector<uint8_t> b;
    kf_draw({ d }, 0, rows, 0, seed, &b);
    std::vector<RadSampleParams> p((size_t)rows);
    CHECK_EQ(b.size(), p.size() * sizeof(RadSampleParams));
    std::memcpy(p.data(), b.data(), std::min(b.size(), p.size() * sizeof(RadSampleParams)));
    return p;
}

/* A [rows, w] history of ids in [0, range), drawn by the history law or uniformly. */
std::vector<int32_t> draw_tokens(int64_t rows, int64_t w, int64_t range, bool history,
                                 uint64_t seed) {
    KfOperand d;
    d.role = RAD_OPD_IN;
    d.dtype = RAD_I32;
    d.shape = { rows, w };
    d.is_index = true;
    d.idx_history = history;
    d.idx_max = range;
    std::vector<uint8_t> b;
    kf_draw({ d }, 0, rows, 0, seed, &b);
    std::vector<int32_t> v((size_t)(rows * w));
    std::memcpy(v.data(), b.data(), std::min(b.size(), v.size() * 4));
    return v;
}

/* Whether h[0, n) ends in a repeat DRY penalises at its default allowed length of 2: its last two
 * tokens occur together earlier. */
bool ends_in_repeat(const int32_t* h, int64_t n) {
    for (int64_t j = 1; j + 1 < n; ++j)
        if (h[j - 1] == h[n - 2] && h[j] == h[n - 1]) return true;
    return false;
}

}  /* namespace */

/* Every row is a request a kernel can act on: fields in a request's ranges, flags agreeing with
 * them as the engine's staging makes them agree, a window inside the history plane and a mask row
 * inside the bitmask. Row 0 switches every stage on; on the others each stage is on in most rows
 * and at its identity in some, so both paths run, and a minority is greedy. */
TEST(sampler_rows_are_drawn_as_requests_every_stage_can_act_on) {
    const int64_t R = 2048, hist = 64, mask_rows = 37;
    const std::vector<RadSampleParams> rows = draw_requests(R, hist, mask_rows, 0x5A3D);
    int greedy = 0, lp = 0, pen = 0, dry = 0, xtc = 0, mask = 0;
    int temp = 0, topk = 0, topp = 0, minp = 0, typical = 0;
    for (int64_t r = 0; r < R; ++r) {
        const RadSampleParams& p = rows[(size_t)r];
        const int f = p.flags;
        CHECK_EQ(f & ~0x3F, 0);
        /* A greedy row runs the argmax at temperature 0; a greedy request asking for logprobs is
         * staged at top_k 1 without the flag instead. */
        const bool g = (f & RAD_SP_GREEDY) != 0;
        if (g) CHECK(p.temp == 0.0f && !(f & RAD_SP_LOGPROBS));
        if (p.top_k == 1) CHECK(p.temp == 0.0f && (f & RAD_SP_LOGPROBS) && !g);
        if (p.temp == 0.0f) CHECK(g || p.top_k == 1);
        else                CHECK(p.temp >= 0.5f && p.temp <= 1.5f);
        CHECK(p.top_k == 0 || p.top_k == 1 || (p.top_k >= 20 && p.top_k <= 100));
        CHECK(p.top_p >= 0.8f && p.top_p <= 1.0f);
        CHECK(p.min_p == 0.0f || (p.min_p >= 0.01f && p.min_p <= 0.1f));
        CHECK(p.typical_p >= 0.7f && p.typical_p <= 1.0f);

        /* Penalties are on exactly when a window counts and a value is not the identity. */
        CHECK(p.rep_penalty >= 1.0f && p.rep_penalty <= 1.3f);
        CHECK(p.freq_penalty >= 0.0f && p.freq_penalty <= 0.5f);
        CHECK(p.pres_penalty >= 0.0f && p.pres_penalty <= 0.5f);
        CHECK(p.penalty_last_n >= -1 && p.penalty_last_n <= hist);
        const bool pen_on = p.penalty_last_n != 0 && (p.rep_penalty != 1.0f ||
                                                      p.freq_penalty != 0.0f ||
                                                      p.pres_penalty != 0.0f);
        CHECK_EQ((f & RAD_SP_PENALTY) != 0, pen_on);

        /* DRY is on exactly when its multiplier is positive; a breaker's repeat limit is never
         * below the allowed length. */
        const bool d = (f & RAD_SP_DRY) != 0;
        CHECK_EQ(d, p.dry_multiplier > 0.0f);
        if (d) {
            CHECK(p.dry_multiplier <= 1.0f);
            CHECK(p.dry_base >= 1.5f && p.dry_base < 2.0f);
            CHECK(p.dry_allowed_length >= 1 && p.dry_allowed_length <= 3);
            CHECK(p.dry_penalty_last_n == -1 ||
                  (p.dry_penalty_last_n >= 8 && p.dry_penalty_last_n <= hist));
            CHECK(p.dry_rep_limit == 0 || (p.dry_rep_limit >= p.dry_allowed_length &&
                                           p.dry_rep_limit <= p.dry_allowed_length + 6));
        } else {
            CHECK_EQ(p.dry_rep_limit, 0);
        }

        const bool x = (f & RAD_SP_XTC) != 0;
        CHECK_EQ(x, p.xtc_probability > 0.0f);
        if (x) CHECK(p.xtc_probability >= 0.75f && p.xtc_probability <= 1.0f &&
                     p.xtc_threshold >= 1e-4f && p.xtc_threshold < 1e-2f);

        /* The window lies inside the plane and covers at least half of what follows its start;
         * a row that reads no history has none. */
        if (f & (RAD_SP_PENALTY | RAD_SP_DRY)) {
            CHECK(p.hist_off >= 0 && p.hist_len >= 1 && p.hist_off + p.hist_len <= hist);
            CHECK(2 * p.hist_len >= hist - p.hist_off);
        } else {
            CHECK(p.hist_off == 0 && p.hist_len == 0);
        }
        CHECK_EQ((f & RAD_SP_MASK) != 0, p.mask_row >= 0);
        CHECK(p.mask_row >= -1 && p.mask_row < mask_rows);

        if (r == 0) continue;
        greedy  += g;
        lp      += (f & RAD_SP_LOGPROBS) != 0;
        pen     += pen_on;
        dry     += d;
        xtc     += x;
        mask    += (f & RAD_SP_MASK) != 0;
        temp    += p.temp > 0.0f && p.temp != 1.0f;
        topk    += p.top_k > 1;
        topp    += p.top_p < 1.0f;
        minp    += p.min_p > 0.0f;
        typical += p.typical_p < 1.0f;
    }

    /* Row 0 switches every stage on, so a one-row case does work in each. */
    const RadSampleParams& z = rows[0];
    CHECK_EQ(z.flags, RAD_SP_PENALTY | RAD_SP_MASK | RAD_SP_XTC | RAD_SP_DRY | RAD_SP_LOGPROBS);
    CHECK(z.temp > 0.0f && z.temp != 1.0f);
    CHECK(z.top_k >= 20 && z.top_p < 1.0f && z.min_p > 0.0f && z.typical_p < 1.0f);
    CHECK(z.xtc_probability == 1.0f && z.xtc_threshold < 1e-3f);
    CHECK(z.hist_off == 0 && z.hist_len == hist);

    /* The rest: a minority greedy, logprobs on most, and every stage on in most rows (seven in
     * eight by the law, fewer for the stages greedy rows skip) while some keep its identity. */
    auto frac = [&](int c) { return (double)c / (double)(R - 1); };
    CHECK(frac(greedy) > 0.08 && frac(greedy) < 0.17);
    CHECK(frac(lp) > 0.65);
    for (int c : { pen, dry, xtc, mask, temp, topk, topp, minp, typical })
        CHECK(frac(c) > 0.65 && frac(c) < 0.95);
}

/* The rows are a function of the seed: two draws agree, another seed differs, and a row does not
 * depend on how many follow it, so the first rows of every point of an M sweep are the same
 * requests. Without a history or a bitmask plane the window is empty and no row is masked. */
TEST(sampler_rows_depend_on_the_seed_alone) {
    auto bytes = [](const std::vector<RadSampleParams>& v) {
        const uint8_t* b = (const uint8_t*)v.data();
        return std::vector<uint8_t>(b, b + v.size() * sizeof(RadSampleParams));
    };
    const std::vector<RadSampleParams> a = draw_requests(256, 64, 256, 7);
    CHECK(bytes(a) == bytes(draw_requests(256, 64, 256, 7)));
    CHECK(bytes(a) != bytes(draw_requests(256, 64, 256, 8)));
    const std::vector<RadSampleParams> head = draw_requests(5, 64, 256, 7);
    CHECK_EQ(std::memcmp(head.data(), a.data(), 5 * sizeof(RadSampleParams)), 0);
    for (const RadSampleParams& p : draw_requests(256, 0, 0, 9)) {
        CHECK(p.hist_off == 0 && p.hist_len == 0);
        CHECK(p.mask_row == -1 && !(p.flags & RAD_SP_MASK));
    }
}

/* A history repeats itself inside its range: at most 16 distinct ids a row, spread over the range;
 * every row ends in a pair of tokens seen together earlier, the repeat DRY penalises, and most
 * windows ending earlier do too. A uniform draw over a vocabulary has no such pair. */
TEST(a_token_history_repeats_itself_inside_its_range) {
    const int64_t R = 512, W = 64, V = 248320;
    const std::vector<int32_t> h = draw_tokens(R, W, V, true, 0x4157);
    int repeats = 0, earlier = 0;
    int32_t top = 0;
    for (int64_t r = 0; r < R; ++r) {
        const int32_t* row = h.data() + r * W;
        std::set<int32_t> ids;
        for (int64_t i = 0; i < W; ++i) {
            CHECK(row[i] >= 0 && row[i] < V);
            top = std::max(top, row[i]);
            ids.insert(row[i]);
        }
        CHECK(ids.size() <= 16);
        repeats += ends_in_repeat(row, W);
        earlier += ends_in_repeat(row, W - 1 - r % 24);
    }
    CHECK(top >= V / 2);
    CHECK_EQ(repeats, (int)R);
    CHECK(2 * earlier > R);
    CHECK(draw_tokens(R, W, V, true, 0x4157) == h);
    CHECK(draw_tokens(R, W, V, true, 0x4158) != h);

    const std::vector<int32_t> u = draw_tokens(R, W, V, false, 0x4157);
    int uniform_repeats = 0;
    for (int64_t r = 0; r < R; ++r) uniform_repeats += ends_in_repeat(u.data() + r * W, W);
    CHECK_EQ(uniform_repeats, 0);
}

namespace {

/* libref's `op` at `g`, run on operands drawn from `seed` as a recording draws them. `before` holds
 * each operand as drawn, an output filled with 0x7F bytes (a value no stage writes), and `after`
 * as the launch left it. False when libref does not describe or run the case. */
bool ref_run(const std::string& op, const Geometry& g, uint64_t seed, std::vector<KfOperand>* opd,
             std::vector<std::vector<uint8_t>>* before, std::vector<std::vector<uint8_t>>* after) {
    std::vector<OpdShape> sh;
    const RadOpSchema* schema = nullptr;
    if (!ref_describe(op, g, &sh, &schema)) return false;
    const KernelRow* row = libref_registry()->select_in("libref", op, g, RAD_DOMAIN_HOST);
    if (!row || !row->info->launch) return false;
    const size_t n = sh.size();
    opd->assign(n, KfOperand{});
    for (size_t k = 0; k < n; ++k) {
        KfOperand& d = (*opd)[k];
        d.role = schema->operands[k].role;
        d.dtype = sh[k].dtype;
        d.shape = sh[k].shape;
        d.absent = sh[k].absent;
        d.fill = sh[k].fill;
        d.is_index = sh[k].is_index;
        d.idx_unique = sh[k].idx_unique;
        d.idx_cu = sh[k].idx_cu;
        d.idx_prefix = sh[k].idx_prefix;
        d.idx_history = sh[k].idx_history;
        d.idx_max = sh[k].idx_max;
        d.idx_const = sh[k].idx_const;
        d.fill_chunk = sh[k].fill_chunk;
    }
    long long M = 1;
    g.get_i("M", &M);
    before->assign(n, {});
    std::vector<RadTensor> t(n);
    for (size_t k = 0; k < n; ++k) {
        const KfOperand& d = (*opd)[k];
        if (d.absent) continue;
        kf_draw(*opd, k, M, kf_vocab_rows(*opd), kf_operand_seed(seed, (int)k), &(*before)[k]);
        if (d.role == RAD_OPD_OUT) std::fill((*before)[k].begin(), (*before)[k].end(), 0x7F);
    }
    *after = *before;
    for (size_t k = 0; k < n; ++k) {
        const KfOperand& d = (*opd)[k];
        if (d.absent) continue;
        t[k].data = (*after)[k].data();
        t[k].dtype = d.dtype;
        t[k].rank = (uint32_t)d.shape.size();
        for (size_t i = 0; i < d.shape.size(); ++i) t[k].shape[i] = d.shape[i];
        rad_tensor_pack(&t[k]);
    }
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)n;
    a.p = g.params();
    a.n_p = g.n_params();
    a.world_size = 1;
    std::vector<uint8_t> scratch;
    if (row->info->scratch) {
        const int64_t need = row->info->scratch(&a);
        if (need > 0) {
            scratch.assign((size_t)need, 0);
            a.scratch = scratch.data();
            a.scratch_bytes = need;
        }
    }
    return row->info->launch(&a, nullptr) == RAD_OK;
}

}  /* namespace */

/* libref's stages do work on the rows libref describes: the penalties, DRY, temperature and the
 * grammar mask change the logits, the narrowing stages drop candidates, logprobs writes its
 * summary and the pick a token, each on most rows of a batch and on row 0, which is all a one-row
 * case has. Under a numeric draw the mask, DRY and XTC change nothing and the penalties act on an
 * eighth of the rows; a kernel is checked only on the rows a stage changes.
 *
 * DRY on row 0 is likely, not certain: the history's closing loop offers three repeats, and each
 * one's continuation is exempt when the breaker plane marks it, which it does at half the
 * positions. */
TEST(every_sampling_stage_does_work_on_most_drawn_rows) {
    if (!libref_registry()) ARTIFACT_MISSING(libref_path() + " (does not load)");
    struct C {
        const char* op;
        const char* checked;   /* the operand a stage's work shows in */
        std::vector<std::pair<const char*, int64_t>> kv;
    };
    const int64_t M = 64;
    const std::vector<C> cases = {
        { "sample_penalties", "logits",   { { "M", M }, { "n_vocab", 4096 } } },
        { "sample_dry",       "logits",   { { "M", M }, { "n_vocab", 4096 } } },
        { "sample_temp",      "logits",   { { "M", M }, { "n_vocab", 4096 } } },
        { "sample_mask",      "logits",   { { "M", M }, { "n_vocab", 4096 } } },
        { "sample_topp",      "cand_idx", { { "M", M }, { "n_cand", 1024 } } },
        { "sample_minp",      "cand_idx", { { "M", M }, { "n_cand", 1024 } } },
        { "sample_typical",   "cand_idx", { { "M", M }, { "n_cand", 1024 } } },
        { "sample_xtc",       "cand_idx", { { "M", M }, { "n_cand", 1024 } } },
        { "sample_logprobs",  "stats",    { { "M", M }, { "n_vocab", 4096 }, { "k", 5 } } },
        { "sample_pick",      "token",    { { "M", M }, { "n_cand", 64 } } },
    };
    for (const C& c : cases) {
        Geometry g;
        for (const auto& [k, v] : c.kv) g.set_i(k, v);
        std::vector<KfOperand> opd;
        std::vector<std::vector<uint8_t>> before, after;
        const bool ran = ref_run(c.op, g, 0xC0DE, &opd, &before, &after);
        if (!ran) std::fprintf(stderr, "    %s: libref did not run it\n", c.op);
        CHECK(ran);
        if (!ran) continue;
        const int k = operand_index(libref_registry()->schema(c.op), c.checked);
        REQUIRE(k >= 0);
        const size_t rb = before[(size_t)k].size() / (size_t)M;
        int changed = 0;
        bool first = false;
        for (int64_t r = 0; r < M; ++r) {
            const bool ch = std::memcmp(before[(size_t)k].data() + r * rb,
                                        after[(size_t)k].data() + r * rb, rb) != 0;
            changed += ch;
            if (r == 0) first = ch;
        }
        const bool row0 = first || std::strcmp(c.op, "sample_dry") == 0;
        if (2 * changed <= M || !row0)
            std::fprintf(stderr, "    %s: %d of %lld rows changed, row 0 %s\n", c.op, changed,
                         (long long)M, first ? "changed" : "unchanged");
        CHECK(2 * changed > M);
        CHECK(row0);
    }
}

/* ================================================================== the projection sketch */

/* The sketch is summed per block, in double, and the blocks added in order, so it does not depend
 * on how many threads summed them. The serial sum below uses the same per-element signs; the claim
 * pinned is the order of the additions. Non-finite elements stay out, as in the norm. */
TEST(the_sketch_is_independent_of_the_thread_count) {
    const int64_t n = KF_REDUCE_BLOCK * 2 + 333;
    std::vector<float> v((size_t)n);
    rad_fill_normal(v.data(), n, 0x5EED, 1.0f);
    v[7] = std::nanf("");
    std::vector<double> got;
    kf_sketch(v.data(), n, 99, &got);
    REQUIRE_EQ(got.size(), (size_t)KF_SKETCH);
    std::vector<double> want(KF_SKETCH, 0.0);
    const uint64_t key = 99ull ^ 0xD6E8FEB86659FD93ull;
    for (int64_t lo = 0; lo < n; lo += KF_REDUCE_BLOCK) {
        double acc[KF_SKETCH] = {};
        for (int64_t i = lo; i < std::min<int64_t>(lo + KF_REDUCE_BLOCK, n); ++i) {
            if (!std::isfinite(v[(size_t)i])) continue;
            uint64_t z = key + (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            for (int j = 0; j < KF_SKETCH; ++j)
                acc[j] += ((z >> j) & 1u) ? -(double)v[(size_t)i] : (double)v[(size_t)i];
        }
        for (int j = 0; j < KF_SKETCH; ++j) want[(size_t)j] += acc[j];
    }
    CHECK(got == want);
    /* Each projection is a random-sign sum: its square estimates the sum of squares. */
    double ss = 0, est = 0;
    for (int64_t i = 0; i < n; ++i)
        if (std::isfinite(v[(size_t)i])) ss += (double)v[(size_t)i] * v[(size_t)i];
    for (double p : got) est += p * p;
    CHECK(est / KF_SKETCH > 0.5 * ss && est / KF_SKETCH < 1.5 * ss);
}

/* ================================================================== the loader */

/* A corrupt but parseable file is refused at load, before anything sized by it is allocated: an
 * extent of 2^40 or below one, an unknown role, a dtype the core cannot size, a plane selection
 * without extents, a sample or sketch count unlike the operand's. */
TEST(a_fixture_with_an_impossible_operand_is_refused) {
    const KfOperand good = [] {
        KfOperand d = opd_f32({ 8, 4 }, RAD_OPD_OUT);
        d.have_ref = true;
        d.ref_n = 32;
        d.sample.assign(32, 1.0f);
        return d;
    }();
    struct Bad { const char* what; KfOperand d; };
    std::vector<Bad> bads;
    { KfOperand d = good; d.shape = { (int64_t)1 << 40, 2 }; d.have_ref = false; d.sample.clear();
      bads.push_back({ "2^41 elements", d }); }
    { KfOperand d = good; d.shape = { 8, -1 }; bads.push_back({ "a negative extent", d }); }
    { KfOperand d = good; d.shape = { 8, 0 }; bads.push_back({ "a zero extent", d }); }
    { KfOperand d = good; d.role = 9; bads.push_back({ "role 9", d }); }
    { KfOperand d = good; d.dtype = 999; bads.push_back({ "dtype 999", d }); }
    { KfOperand d = good; d.dtype = RAD_DT_PLUGIN_BASE + 3; bads.push_back({ "a private dtype", d }); }
    { KfOperand d = good; d.sample.resize(31); bads.push_back({ "31 samples of 32", d }); }
    { KfOperand d = good; d.ref_n = 33; bads.push_back({ "ref_n unlike the shape", d }); }
    { KfOperand d = good; d.sketch.assign(5, 0.0); bads.push_back({ "a sketch of 5", d }); }
    { KfOperand d = good; d.fill = 77; bads.push_back({ "fill 77", d }); }
    {
        KfOperand d = fp8_weight(256, 384, { "codes", "scale" });
        d.sel_rows[1] = 0;
        bads.push_back({ "a plane with no rows", d });
    }
    for (const Bad& b : bads) {
        Kfixture fx;
        fx.source = "x";
        KfCase c;
        c.op = "add";
        c.opd.push_back(b.d);
        fx.cases.push_back(c);
        const std::string path = radtest::tmp_path("bad_operand.rkb");
        REQUIRE_EQ(fx.save(path), RAD_OK);
        Kfixture rd;
        const int rc = rd.load(path);
        if (rc != RAD_E_FORMAT) std::fprintf(stderr, "    %s loaded: %s\n", b.what, rad_strerror(rc));
        CHECK_EQ(rc, RAD_E_FORMAT);
    }
    /* The same operand, valid, loads. */
    Kfixture fx;
    fx.source = "x";
    KfCase c;
    c.op = "add";
    c.opd.push_back(good);
    fx.cases.push_back(c);
    const std::string path = radtest::tmp_path("good_operand.rkb");
    REQUIRE_EQ(fx.save(path), RAD_OK);
    Kfixture rd;
    CHECK_OK(rd.load(path));
}

/* Version 4 still loads: no sketch, a per-case flag byte no recording set, and weight tables the
 * reader marks undrawn so a replay draws the zeros their references were computed on. Written here
 * field by field as the version-4 writer did. */
TEST(a_version_4_fixture_loads_with_its_weight_tables_undrawn) {
    std::vector<uint8_t> b;
    auto raw = [&](const void* p, size_t n) {
        b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n);
    };
    auto u8 = [&](uint8_t v) { raw(&v, 1); };
    auto u32 = [&](uint32_t v) { raw(&v, 4); };
    auto i64 = [&](int64_t v) { raw(&v, 8); };
    auto str = [&](const std::string& s) { u32((uint32_t)s.size()); raw(s.data(), s.size()); };
    raw("RKB1", 4);
    u32(4);
    u32((uint32_t)KF_SAMPLES);
    str("src");
    str("libref");
    str("2026-10-01");
    u32(1);                                    /* cases */
    str("moe_gemm_q");
    i64(77);                                   /* seed */
    i64(3);                                    /* m */
    str("band");
    u8(1);                                     /* a per-case flag byte */
    u32(0);                                    /* params */
    u32(2);                                    /* operands */
    /* a weight table, no reference */
    u32(RAD_OPD_WTAB); u32(RAD_F8E4M3); u32(3); i64(2); i64(4); i64(8);
    u8(0); u32(0); u8(0); i64(0); i64(-1); i64(0); u8(0); u8(0);
    /* the output, with a reference */
    u32(RAD_OPD_OUT); u32(RAD_BF16); u32(2); i64(3); i64(4);
    u8(0); u32(0); u8(0); i64(0); i64(-1); i64(0); u8(0); u8(1);
    i64(12);                                   /* ref_n */
    { const double ss = 0.0; raw(&ss, 8); }
    i64(0);
    u32(12);
    for (int i = 0; i < 12; ++i) { const float z = 0.0f; raw(&z, 4); }
    const std::string path = radtest::tmp_path("v4.rkb");
    FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    Kfixture fx;
    REQUIRE_EQ(fx.load(path), RAD_OK);
    CHECK_EQ(fx.version, (uint32_t)4);
    REQUIRE_EQ(fx.cases.size(), (size_t)1);
    REQUIRE_EQ(fx.cases[0].opd.size(), (size_t)2);
    CHECK(fx.cases[0].opd[0].undrawn);
    CHECK(!fx.cases[0].opd[1].undrawn);
    CHECK(fx.cases[0].opd[1].sketch.empty());
    CHECK_EQ(fx.cases[0].opd[1].ref_n, (int64_t)12);
}

namespace {

/* A file of one case with one [4, 24] f32 input, written field by field as versions 5 and 6 lay
 * it out, the input carrying fill law `fill` and index-flag byte `flags`. */
std::string one_input_fixture(uint32_t version, uint32_t fill, uint8_t flags) {
    std::vector<uint8_t> b;
    auto raw = [&](const void* p, size_t n) {
        b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n);
    };
    auto u8 = [&](uint8_t v) { raw(&v, 1); };
    auto u32 = [&](uint32_t v) { raw(&v, 4); };
    auto i64 = [&](int64_t v) { raw(&v, 8); };
    auto str = [&](const std::string& s) { u32((uint32_t)s.size()); raw(s.data(), s.size()); };
    raw("RKB1", 4);
    u32(version);
    u32((uint32_t)KF_SAMPLES);
    str("src");
    str("libref");
    str("2026-10-16");
    u32(1);                                    /* cases */
    str("sample_temp");
    i64(5);                                    /* seed */
    i64(4);                                    /* m */
    str("band");
    u32(0);                                    /* params */
    u32(1);                                    /* operands */
    u32(RAD_OPD_IN); u32(RAD_F32); u32(2); i64(4); i64(24);
    u8(0);                                     /* absent */
    u32(fill);
    u8(flags);
    i64(4); i64(-1); i64(64);                  /* idx_max, idx_const, fill_chunk */
    u8(0); u8(0);                              /* no encoding, no reference */
    const std::string path = radtest::tmp_path("one_input_v" + std::to_string(version) + "_" +
                                               std::to_string(fill) + "_" +
                                               std::to_string(flags) + ".rkb");
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return path;
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    return path;
}

}  /* namespace */

/* The sampler laws arrived with version 6. A version-6 file carrying them loads; a version-5 file
 * still loads, but one claiming a law that version did not have is corrupt; an index flag or a
 * version this reader does not know is refused rather than drawn as something else. */
TEST(a_law_newer_than_its_file_or_unknown_to_the_reader_is_refused) {
    Kfixture fx;
    REQUIRE_EQ(fx.load(one_input_fixture(6, KF_FILL_SAMPLE_PARAMS, 0)), RAD_OK);
    REQUIRE(fx.cases.size() == 1 && fx.cases[0].opd.size() == 1);
    CHECK_EQ(fx.cases[0].opd[0].fill, (int)KF_FILL_SAMPLE_PARAMS);
    CHECK_EQ(fx.cases[0].opd[0].fill_chunk, (int64_t)64);
    CHECK_EQ(fx.cases[0].opd[0].idx_max, (int64_t)4);
    REQUIRE_EQ(fx.load(one_input_fixture(6, KF_FILL_NORMAL, 1 | 16)), RAD_OK);
    CHECK(fx.cases[0].opd[0].is_index && fx.cases[0].opd[0].idx_history);
    CHECK_EQ(fx.load(one_input_fixture(5, KF_FILL_NORMAL, 1)), RAD_OK);
    CHECK_EQ(fx.version, (uint32_t)5);

    CHECK_EQ(fx.load(one_input_fixture(5, KF_FILL_SAMPLE_PARAMS, 0)), RAD_E_FORMAT);
    CHECK_EQ(fx.load(one_input_fixture(5, KF_FILL_NORMAL, 1 | 16)), RAD_E_FORMAT);
    /* The block-order law is version 7's; a version-6 file claiming it held a law it did not. */
    CHECK_EQ(fx.load(one_input_fixture(6, KF_FILL_NORMAL, 1 | 32)), RAD_E_FORMAT);
    REQUIRE_EQ(fx.load(one_input_fixture(7, KF_FILL_NORMAL, 1 | 32)), RAD_OK);
    CHECK(fx.cases[0].opd[0].is_index && fx.cases[0].opd[0].idx_ordered);
    CHECK_EQ(fx.version, (uint32_t)7);
    CHECK_EQ(fx.load(one_input_fixture(6, KF_FILL_SAMPLE_PARAMS + 1, 0)), RAD_E_FORMAT);
    CHECK_EQ(fx.load(one_input_fixture(KF_VERSION + 1, KF_FILL_NORMAL, 0)), RAD_E_FORMAT);
}

/* ================================================================== the fallback shape table */

namespace {

int shape_wrong_rank(const RadParam*, int, int, RadOpdDesc* out) {
    *out = RadOpdDesc{};
    out->dtype = RAD_F32;
    return RAD_OK;                               /* rank 0, not absent */
}
int shape_zero_extent(const RadParam*, int, int, RadOpdDesc* out) {
    *out = RadOpdDesc{};
    out->dtype = RAD_F32;
    out->rank = 2;
    out->shape[0] = 4;
    out->shape[1] = 0;
    return RAD_OK;
}
int shape_absent(const RadParam*, int, int, RadOpdDesc* out) {
    *out = RadOpdDesc{};
    out->flags = RAD_OPD_F_ABSENT;
    return RAD_OK;
}

}  /* namespace */

/* A description that cannot be a buffer is a hook bug, not a decline: a zero-extent buffer passes
 * every comparison vacuously. Rank 0 is legal only for an absent operand. */
TEST(a_hook_describing_an_impossible_operand_is_refused) {
    RadKernelInfo ki{};
    std::vector<OpdShape> sh;
    ki.opd_shape = shape_wrong_rank;
    CHECK_EQ(rad_operand_shapes_hook(&ki, Geometry{}, 2, &sh), RAD_E_SHAPE);
    ki.opd_shape = shape_zero_extent;
    CHECK_EQ(rad_operand_shapes_hook(&ki, Geometry{}, 2, &sh), RAD_E_SHAPE);
    ki.opd_shape = shape_absent;
    CHECK_EQ(rad_operand_shapes_hook(&ki, Geometry{}, 2, &sh), RAD_OK);
    CHECK(sh.size() == 2 && sh[0].absent && sh[1].absent);
    ki.opd_shape = nullptr;
    CHECK_EQ(rad_operand_shapes_hook(&ki, Geometry{}, 2, &sh), RAD_E_UNSUPPORTED);
}

/* tools/opshapes.cpp's table is the fallback rad-kbench and rad-tune use when no hook describes an
 * op. A recipe that emits the wrong operand count is refused with only a debug line, so a stale
 * one silently drops the op (rad-tune then never tunes it). Each recipe must emit the
 * schema's count and describe what libref describes. */
TEST(the_fallback_shape_table_agrees_with_the_reference) {
    if (!libref_registry()) ARTIFACT_MISSING(libref_path() + " (does not load)");
    struct G { const char* op; std::vector<std::pair<const char*, const char*>> kv; int64_t M; };
    const std::vector<G> cases = {
        { "rmsnorm",      { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "add",          { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "mul",          { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "silu_mul",     { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "softmax",      { { "M", "4" }, { "n", "64" }, { "dtype", "f32" } }, 4 },
        { "gather_rows",  { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "cast",         { { "M", "4" }, { "n", "64" }, { "from", "bf16" }, { "to", "f32" } }, 4 },
        { "gemm_nt",      { { "M", "4" }, { "N", "96" }, { "K", "64" }, { "dtype", "bf16" } }, 4 },
        { "gemm_nt_bias", { { "M", "4" }, { "N", "96" }, { "K", "64" }, { "dtype", "bf16" } }, 4 },
        { "embed_lookup", { { "M", "4" }, { "n_vocab", "500" }, { "n_embd", "64" },
                            { "dtype", "bf16" } }, 4 },
        { "logits_gemm",  { { "M", "4" }, { "n_vocab", "500" }, { "n_embd", "64" },
                            { "dtype", "bf16" } }, 4 },
        { "quant_act_fp8", { { "M", "4" }, { "n", "256" }, { "group", "128" },
                             { "dtype", "bf16" } }, 4 },
        { "gemm_nt_q",    { { "M", "4" }, { "N", "256" }, { "K", "256" }, { "group", "128" },
                            { "dtype", "fp8a8" } }, 4 },
        { "kv_store",     { { "M", "4" }, { "head_dim", "64" }, { "n_head_kv", "2" },
                            { "block_size", "16" }, { "kv_dtype", "bf16" } }, 4 },
        { "attn_paged",   { { "q_len", "4" }, { "head_dim", "64" }, { "gqa", "2" },
                            { "n_head", "4" }, { "block_size", "16" }, { "causal", "1" },
                            { "window", "0" }, { "q_dtype", "bf16" }, { "kv_dtype", "bf16" } }, 4 },
        { "gdn_recurrent_update", { { "q_len", "2" }, { "head_k", "16" }, { "head_v", "16" },
                                    { "n_head_k", "2" }, { "n_head_v", "4" }, { "n_seq", "1" } },
          2 },
        { "gdn_gated_rmsnorm", { { "M", "8" }, { "channels", "16" } }, 8 },
        { "rope",         { { "M", "4" }, { "head_dim", "64" }, { "n_head", "4" },
                            { "n_head_kv", "2" }, { "rot", "64" }, { "mode", "neox" } }, 4 },
        { "rmsnorm_add",  { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "layernorm",    { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "gelu",         { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "silu",         { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "sigmoid",      { { "M", "4" }, { "n", "64" }, { "dtype", "bf16" } }, 4 },
        { "quant_act_i8", { { "M", "4" }, { "n", "64" }, { "group", "64" },
                            { "dtype", "bf16" } }, 4 },
        { "dequant",      { { "M", "4" }, { "n", "64" }, { "group", "32" },
                            { "dtype", "bf16" } }, 4 },
        { "all_reduce",   { { "numel", "64" }, { "dtype", "bf16" }, { "world_size", "1" } }, 1 },
        { "all_gather",   { { "numel", "64" }, { "dtype", "bf16" }, { "world_size", "2" } }, 1 },
        { "gdn_conv_prep", { { "M", "8" }, { "head_k", "16" }, { "head_v", "16" },
                             { "conv_width", "4" }, { "chunk", "64" }, { "n_head_k", "2" },
                             { "n_head_v", "4" }, { "n_seq", "1" } }, 8 },
        { "gdn_kkt_solve", { { "M", "8" }, { "head_k", "16" }, { "chunk", "64" },
                             { "n_head_k", "2" }, { "n_head_v", "4" }, { "n_seq", "1" } }, 8 },
        { "gdn_chunk_scan", { { "M", "8" }, { "head_k", "16" }, { "head_v", "16" },
                              { "chunk", "64" }, { "n_head_k", "2" }, { "n_head_v", "4" },
                              { "n_seq", "1" } }, 8 },
        { "gdn_conv_update", { { "q_len", "2" }, { "head_k", "16" }, { "head_v", "16" },
                               { "conv_width", "4" }, { "n_head_k", "2" }, { "n_head_v", "4" },
                               { "n_seq", "1" } }, 2 },
    };
    for (const G& c : cases) {
        Geometry g;
        for (const auto& [k, v] : c.kv) {
            char* end = nullptr;
            const long long iv = std::strtoll(v, &end, 10);
            if (end && *end == '\0') g.set_i(k, iv); else g.set_s(k, v);
        }
        std::vector<OpdShape> hook, tab;
        const RadOpSchema* schema = nullptr;
        if (!ref_describe(c.op, g, &hook, &schema)) {
            std::fprintf(stderr, "    %s: libref does not describe it here\n", c.op);
            CHECK(false);
            continue;
        }
        ShapeCtx sctx;
        long long v = 0;
        if (g.get_i("n_head_k", &v)) sctx.n_head_k = v;
        if (g.get_i("n_head_v", &v)) sctx.n_head_v = v;
        const int rc = rad_operand_shapes(c.op, g, schema->n_operands, c.M, &tab, &sctx);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "    %s: the table refuses it (%s)\n", c.op, rad_strerror(rc));
            CHECK(false);
            continue;
        }
        CHECK_EQ((int)tab.size(), schema->n_operands);
        for (size_t k = 0; k < tab.size() && k < hook.size(); ++k) {
            const bool same = tab[k].absent == hook[k].absent &&
                              (tab[k].absent || (tab[k].dtype == hook[k].dtype &&
                                                 tab[k].shape == hook[k].shape &&
                                                 tab[k].is_index == hook[k].is_index));
            if (!same)
                std::fprintf(stderr, "    %s operand %zu: the table says %s%s, libref %s%s\n",
                             c.op, k, rad_dtype_name(tab[k].dtype), tab[k].absent ? " absent" : "",
                             rad_dtype_name(hook[k].dtype), hook[k].absent ? " absent" : "");
            CHECK(same);
        }
    }
}

RAD_TEST_MAIN()
