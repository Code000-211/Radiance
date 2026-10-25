/* kbench_test.cpp -- rad-kbench's rules (tools/kbench_core.h), without plugins or a device.
 *
 * A replay that cannot fail proves nothing, and most ways it could not fail are rules: an id
 * judged by a relative norm, a tolerance that falls to a catch-all, a cap key that collapses the
 * attention depth sweep, a NaN dropped from the worst error, an exit code that ignores an overrun,
 * a cache that serves a stale answer. Each rule is pinned here against a value worked out by hand
 * or by an independent formulation. tests/kbench_mutant_test.cpp runs the tool end to end.
 */
#include "rad_test.h"
#include "kbench_core.h"
#include "kfixture.h"
#include "format/radfile.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

/* A new empty directory under the test's temporary directory, or "" when it cannot be made. */
std::string temp_dir() {
    std::string tmpl = radtest::tmp_path("dir.XXXXXX");
    return ::mkdtemp(tmpl.data()) ? tmpl : std::string();
}

Geometry geom(std::initializer_list<std::pair<const char*, const char*>> kv) {
    Geometry g;
    for (const auto& [k, v] : kv) {
        char* end = nullptr;
        const long long iv = std::strtoll(v, &end, 10);
        if (end && *end == '\0' && end != v) g.set_i(k, iv);
        else g.set_s(k, v);
    }
    return g;
}

/* A recorded output from the values `ref`, reduced as the recorder reduces it. */
KfOperand recorded(const std::vector<float>& ref, uint32_t dtype, uint64_t seed, bool sketch) {
    KfOperand d;
    d.role = RAD_OPD_OUT;
    d.dtype = dtype;
    d.shape = { (int64_t)ref.size() };
    d.have_ref = true;
    d.ref_n = (int64_t)ref.size();
    kf_reduce(ref.data(), d.ref_n, seed, &d.ref_sumsq, &d.ref_nonfinite, &d.sample);
    if (sketch) kf_sketch(ref.data(), d.ref_n, seed, &d.sketch);
    return d;
}

kb::Digest digest(const std::vector<float>& v, uint64_t seed, bool sketch = true) {
    kb::Digest d;
    kb::digest_of(v.data(), (int64_t)v.size(), seed, sketch, &d);
    return d;
}

std::vector<float> normals(int64_t n, uint64_t seed) {
    std::vector<float> v((size_t)n);
    rad_fill_normal(v.data(), n, seed, 1.0f);
    return v;
}

}  /* namespace */

/* ================================================================== errors */

/* std::max(worst, NaN) returns worst, so a NaN case would leave the summary's worst error at the
 * last finite value while the case itself failed. The worst is NaN-sticky. */
TEST(a_nan_error_stays_the_worst) {
    CHECK(std::isnan(kb::worst_of(1.0, std::nan(""))));
    CHECK(std::isnan(kb::worst_of(std::nan(""), 0.5)));
    CHECK_EQ(kb::worst_of(-1.0, 0.25), 0.25);
    CHECK_EQ(kb::worst_of(0.5, 0.25), 0.5);
}

/* A failing operand outranks a passing one with a larger error, and a NaN once worst stays. */
TEST(the_worst_operand_is_a_failing_one_then_the_largest) {
    CHECK(kb::takes_worst(0.1, false, -1.0, false));
    CHECK(kb::takes_worst(1e-9, true, 0.5, false));
    CHECK(!kb::takes_worst(0.9, false, 1e-3, true));
    CHECK(kb::takes_worst(std::nan(""), true, 0.5, true));
    CHECK(!kb::takes_worst(10.0, true, std::nan(""), true));
}

/* A NaN against a number is a disagreement whatever the norm says, so it is counted apart from
 * rel_l2 (which skips it). */
TEST(an_elementwise_comparison_counts_finiteness_disagreements_apart) {
    const float ref[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float got[] = { 1.0f, std::nanf(""), 3.0f, 4.0f };
    const kb::Err e = kb::compare(got, ref, 4);
    CHECK_EQ(e.n_nonfinite_mismatch, (int64_t)1);
    CHECK_EQ(e.rel_l2, 0.0);
    const float zero[] = { 0.0f, 0.0f };
    const float one[] = { 0.0f, 1.0f };
    CHECK_EQ(kb::compare(one, zero, 2).rel_l2, 1.0);
    CHECK_EQ(kb::compare(zero, zero, 2).rel_l2, 0.0);
}

/* ================================================================== tolerances */

/* The attention and cache ops carry no `dtype`, so a lookup keyed on it alone sent attn_paged to
 * the catch-all (3e-2) and kv_store to 2e-2, which hides a dropped rounding step. */
TEST(the_tolerance_reads_the_attention_and_cache_dtypes) {
    const kb::TolPick attn = kb::tolerance_for("attn_paged",
                                               geom({ { "q_dtype", "bf16" },
                                                      { "kv_dtype", "fp8_e4m3" } }), "out");
    CHECK_EQ(attn.rel_l2, 2e-2);
    CHECK(!attn.catch_all);
    CHECK_EQ(kb::tolerance_for("kv_store", geom({ { "kv_dtype", "fp8_e4m3" } }), "kv_cache")
                 .rel_l2, 0.0);
    CHECK_EQ(kb::tolerance_for("gemm_nt", geom({ { "dtype", "bf16" } }), "y").rel_l2, 8e-3);
    CHECK_EQ(kb::tolerance_for("gemm_nt_q", geom({ { "dtype", "fp8a8" } }), "y").rel_l2, 5e-3);
    CHECK_EQ(kb::tolerance_for("gemm_nt_q", geom({ { "dtype", "w4a8" } }), "y").rel_l2, 3e-2);
    CHECK_EQ(kb::tolerance_for("gemm_nt_q_gated", geom({ { "dtype", "fp8a8" } }), "q").rel_l2,
             3e-2);
    CHECK_EQ(kb::tolerance_for("rope_table", geom({ { "dtype", "f32" } }), "cos").rel_l2, 1e-5);
    CHECK_EQ(kb::tolerance_for("rope", Geometry{}, "qkv").rel_l2, 2e-3);
    CHECK_EQ(kb::tolerance_for("gdn_kkt_solve", Geometry{}, "A").rel_l2, 2e-3);
    CHECK_EQ(kb::tolerance_for("gdn_chunk_scan", Geometry{}, "o").rel_l2, 2e-2);
    CHECK_EQ(kb::tolerance_for("sample_temp", Geometry{}, "logits").rel_l2, 1e-6);
    /* A per-operand override wins over the op's row. */
    CHECK_EQ(kb::tolerance_for("hc_read", geom({ { "dtype", "bf16" } }), "q").rel_l2, 3e-2);
    /* An op nobody wrote a row for lands on the catch-all, and says so. */
    const kb::TolPick none = kb::tolerance_for("zz_new_op", Geometry{}, "y");
    CHECK(none.catch_all);
}

/* Copies, gathers and scatters move bits; anything but equality is a bug. */
TEST(a_copy_is_judged_exactly) {
    for (const char* op : { "gather_rows", "scatter_rows", "embed_lookup", "kv_store",
                            "row_topk", "pair_argmax", "all_gather" })
        CHECK_EQ(kb::tolerance_for(op, geom({ { "dtype", "bf16" } }), "y").rel_l2, 0.0);
}

/* Every float output of every op the checked-in fixtures record resolves to a row chosen for it,
 * not the catch-all. A new op gets a tolerance deliberately, not by default. */
TEST(every_op_in_the_checked_in_fixtures_has_a_tolerance_row_of_its_own) {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return;
    for (const char* leaf : { "kernels.rkb", "kernels_moe.rkb" }) {
        Kfixture fx;
        if (fx.load(std::string(home) + "/" + leaf) != RAD_OK) continue;
        std::set<std::string> bad;
        for (const KfCase& c : fx.cases) {
            Geometry g;
            for (const KfParam& p : c.params)
                if (p.kind == RAD_P_STR) g.set_s(p.key, p.s);
            for (const KfOperand& d : c.opd) {
                if (d.absent || !d.have_ref || kb::is_index_dtype(d.dtype)) continue;
                if (kb::tolerance_for(c.op, g, nullptr).catch_all) bad.insert(c.op);
            }
        }
        for (const std::string& op : bad)
            std::fprintf(stderr, "    %s: %s has no tolerance row of its own\n", leaf, op.c_str());
        CHECK(bad.empty());
    }
}

/* ================================================================== ids */

/* router_topk at M = 1 has 8 expert ids with rms ~150: one id off by one is rel_l2 2.4e-3, inside
 * a 6e-3 tolerance. An id is right or wrong. */
TEST(an_id_off_by_one_fails_at_any_tolerance) {
    std::vector<float> ref = { 140, 7, 250, 31, 99, 180, 3, 201 };
    std::vector<float> got = ref;
    got[4] = 100;
    const KfOperand rec = recorded(ref, RAD_I32, 11, true);
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(got, 11), 11, 1.0, false, nullptr);
    CHECK(v.index);
    CHECK(v.bad);
    CHECK_EQ(v.index_mismatch, (int64_t)1);
    CHECK(v.sample_rel < 6e-3);       /* what a norm would have said */
    CHECK(!kb::judge_operand(rec, digest(ref, 11), 11, 0.0, false, nullptr).bad);
}

/* sample_argmax's contract does not fix which of two equal logits wins, so an id that indexes a
 * value equal to the reference id's is a tie, not a failure. A different value is still wrong, and
 * an id pair plane's value lanes must be equal. */
TEST(a_tie_is_judged_by_the_value_the_id_indexes) {
    const float logits[] = { 0.5f, 2.0f, 2.0f, 1.0f,     /* row 0: ids 1 and 2 tie */
                             3.0f, 0.0f, 1.0f, 2.0f };   /* row 1 */
    kb::TieView tv;
    tv.in = logits;
    tv.in_dtype = RAD_F32;
    tv.in_rows = 2;
    tv.in_width = 4;
    tv.out_width = 1;
    CHECK(kb::index_matches(&tv, 0, 2, 1));
    CHECK(!kb::index_matches(&tv, 0, 3, 1));
    CHECK(!kb::index_matches(&tv, 1, 1, 0));         /* row 1: 0.0 against 3.0 */
    CHECK(!kb::index_matches(nullptr, 0, 2, 1));     /* no tie rule: exact */
    CHECK(!kb::index_matches(&tv, 0, 9, 1));         /* outside the row */
    tv.lane_stride = 2;
    tv.out_width = 2;
    CHECK(kb::index_matches(&tv, 0, 2, 1));          /* an id lane */
    CHECK(!kb::index_matches(&tv, 1, 2, 1));         /* a value lane */
    CHECK(kb::tie_rule("sample_argmax", "token") != nullptr);
    CHECK(kb::tie_rule("router_topk", "expert_ids") != nullptr);
    CHECK(kb::tie_rule("sample_topk", "pairs") != nullptr);
    CHECK_EQ(kb::tie_rule("sample_topk", "pairs")->lane_stride, 2);
    /* These break ties on the lower id by contract (docs/OPS.md). */
    CHECK(kb::tie_rule("row_topk", "idx") == nullptr);
    CHECK(kb::tie_rule("pair_argmax", "idx") == nullptr);
}

/* Past KF_SAMPLES ids only a sample is compared; the norm then covers the rest. Without a tie rule
 * it must be equal, with one it may move by a tie's difference. */
TEST(an_index_norm_is_exact_unless_the_op_allows_ties) {
    std::vector<float> ref((size_t)KF_SAMPLES * 4);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = (float)(i % 977);
    const uint64_t seed = 5;
    std::vector<int64_t> pos;
    kf_sample_positions(seed, (int64_t)ref.size(), &pos);
    std::set<int64_t> sampled(pos.begin(), pos.end());
    int64_t hidden = 0;
    while (sampled.count(hidden)) ++hidden;
    std::vector<float> got = ref;
    got[(size_t)hidden] += 1.0f;                     /* an id no sample sees */
    const KfOperand rec = recorded(ref, RAD_I32, seed, false);
    const kb::OpdVerdict exact = kb::judge_operand(rec, digest(got, seed, false), seed, 1.0,
                                                   false, nullptr);
    CHECK_EQ(exact.index_mismatch, (int64_t)0);
    CHECK(exact.bad);
    std::vector<float> in(ref.size(), 0.0f);
    kb::TieView tv;
    tv.in = in.data();
    tv.in_dtype = RAD_F32;
    tv.in_rows = 1;
    tv.in_width = 1000;
    tv.out_width = (int64_t)ref.size();
    CHECK(!kb::judge_operand(rec, digest(got, seed, false), seed, 1.0, false, &tv).bad);
}

/* ================================================================== the replay's verdict */

TEST(a_correct_output_passes_and_one_past_its_tolerance_fails) {
    const std::vector<float> ref = normals(5000, 3);
    const KfOperand rec = recorded(ref, RAD_F32, 9, true);
    CHECK(!kb::judge_operand(rec, digest(ref, 9), 9, 6e-3, false, nullptr).bad);
    std::vector<float> scaled = ref;
    for (float& x : scaled) x *= 1.018f;
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(scaled, 9), 9, 6e-3, false, nullptr);
    CHECK(v.bad);
    CHECK_NEAR(v.sample_rel, 0.018, 1e-4);
    CHECK_NEAR(v.norm_rel, 0.018, 1e-4);
    /* The verdict's error is the largest of the three; the sketch's is an estimate (+-25%). */
    CHECK(v.rel >= v.sample_rel && v.rel >= v.norm_rel);
    CHECK_NEAR(v.sketch_rel, 0.018, 0.25 * 0.018);
}

/* The samples cannot see an output whose first part is right and the rest unwritten if no sample
 * lands there; the full-tensor norm does. Built so every sampled value agrees. */
TEST(an_unwritten_tail_fails_by_the_norm) {
    const int64_t n = 1 << 16;
    const std::vector<float> ref = normals(n, 4);
    const uint64_t seed = 17;
    std::vector<int64_t> pos;
    kf_sample_positions(seed, n, &pos);
    std::set<int64_t> sampled(pos.begin(), pos.end());
    std::vector<float> got = ref;
    for (int64_t i = n / 2; i < n; ++i) if (!sampled.count(i)) got[(size_t)i] = 0.0f;
    const KfOperand rec = recorded(ref, RAD_F32, seed, false);
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(got, seed, false), seed, 6e-3, false,
                                               nullptr);
    CHECK_EQ(v.sample_rel, 0.0);
    CHECK(v.norm_rel > 0.2);
    CHECK(v.bad);
}

/* A sign-flipped tile keeps the norm and, if no sample lands in it, every sample: only the
 * projection sketch sees it. A version-4 recording, which has no sketch, passes it. */
TEST(a_flipped_tile_no_sample_sees_fails_by_the_sketch) {
    const int64_t n = 1 << 20;
    const std::vector<float> ref = normals(n, 6);
    const uint64_t seed = 23;
    std::vector<int64_t> pos;
    kf_sample_positions(seed, n, &pos);
    std::set<int64_t> sampled(pos.begin(), pos.end());
    /* The first 4096-element tile that holds no sampled position. */
    int64_t lo = 0;
    for (; lo + 4096 <= n; lo += 4096) {
        auto it = sampled.lower_bound(lo);
        if (it == sampled.end() || *it >= lo + 4096) break;
    }
    REQUIRE(lo + 4096 <= n);
    std::vector<float> got = ref;
    for (int64_t i = lo; i < lo + 4096; ++i) got[(size_t)i] = -got[(size_t)i];

    const KfOperand v5 = recorded(ref, RAD_F32, seed, true);
    const kb::OpdVerdict v = kb::judge_operand(v5, digest(got, seed), seed, 6e-3, false, nullptr);
    CHECK_EQ(v.sample_rel, 0.0);
    CHECK(v.norm_rel < 1e-6);
    /* The true rel_l2 is 2 * sqrt(4096 / n) = 0.125; the sketch estimates it to ~25%. */
    CHECK(v.sketch_rel > 0.06 && v.sketch_rel < 0.2);
    CHECK(v.bad);
    const KfOperand v4 = recorded(ref, RAD_F32, seed, false);
    CHECK(!kb::judge_operand(v4, digest(got, seed, false), seed, 6e-3, false, nullptr).bad);
}

/* Spread noise of known size: the sketch's estimate lands within its stated spread of the truth,
 * so the existing tolerances apply to it as they are. */
TEST(the_sketch_estimates_the_full_tensor_error) {
    const int64_t n = 200000;
    const std::vector<float> ref = normals(n, 8);
    const std::vector<float> noise = normals(n, 9);
    std::vector<float> got((size_t)n);
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        got[(size_t)i] = ref[(size_t)i] + 1e-3f * noise[(size_t)i];
        const double d = (double)got[(size_t)i] - (double)ref[(size_t)i];
        num += d * d;
        den += (double)ref[(size_t)i] * (double)ref[(size_t)i];
    }
    const double truth = std::sqrt(num / den);
    const KfOperand rec = recorded(ref, RAD_F32, 2, true);
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(got, 2), 2, 1.0, false, nullptr);
    CHECK(v.sketch_rel > 0.4 * truth && v.sketch_rel < 1.6 * truth);
}

/* Both sides all zero means a kernel that wrote nothing passed: vacuous, not a pass. A zero
 * reference against a written output is a failure. */
TEST(a_zero_reference_against_a_zero_output_is_vacuous) {
    const std::vector<float> zeros(64, 0.0f);
    const KfOperand rec = recorded(zeros, RAD_BF16, 1, true);
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(zeros, 1), 1, 6e-3, false, nullptr);
    CHECK(v.vacuous);
    CHECK(!v.bad);
    std::vector<float> some = zeros;
    some[3] = 1.0f;
    const kb::OpdVerdict w = kb::judge_operand(rec, digest(some, 1), 1, 6e-3, false, nullptr);
    CHECK(!w.vacuous);
    CHECK(w.bad);
}

/* The reference replays its own recording bit for bit: one ulp off fails it, while another
 * library is still within its tolerance. */
TEST(the_reference_is_judged_exactly_against_its_own_recording) {
    std::vector<float> ref = normals(300, 12);
    const KfOperand rec = recorded(ref, RAD_F32, 4, true);
    std::vector<float> ulp = ref;
    ulp[7] = std::nextafter(ulp[7], 100.0f);
    CHECK(!kb::judge_operand(rec, digest(ref, 4), 4, 1e-5, true, nullptr).bad);
    CHECK(kb::judge_operand(rec, digest(ulp, 4), 4, 1e-5, true, nullptr).bad);
    CHECK(!kb::judge_operand(rec, digest(ulp, 4), 4, 1e-5, false, nullptr).bad);
}

TEST(a_nan_against_a_number_fails_at_any_tolerance) {
    std::vector<float> ref = normals(100, 13);
    const KfOperand rec = recorded(ref, RAD_F32, 6, false);
    std::vector<float> got = ref;
    got[50] = std::nanf("");
    const kb::OpdVerdict v = kb::judge_operand(rec, digest(got, 6, false), 6, 1e9, false, nullptr);
    CHECK(v.nonfinite_gap > 0);
    CHECK(v.bad);
}

/* ================================================================== keys */

/* The attention depth sweep records the same parameters with a 128-block and a 1024-block cache;
 * a cap key without operand shapes keeps only the first. */
TEST(the_cap_key_keeps_cases_whose_operand_shapes_differ) {
    KfCase a;
    a.op = "attn_paged";
    a.params.push_back(KfParam{ "q_len", RAD_P_INT, 64, 0, "" });
    KfOperand kv;
    kv.role = RAD_OPD_IN;
    kv.dtype = RAD_F8E4M3;
    kv.shape = { 128, 4, 16, 512 };
    a.opd.push_back(kv);
    KfCase b = a;
    b.opd[0].shape[0] = 1024;
    const Geometry g = geom({ { "q_len", "64" } });
    CHECK(kb::cap_key(a, "k", "lib", 0, g) != kb::cap_key(b, "k", "lib", 0, g));
    KfCase a2 = a;
    a2.seed = 99;                                    /* a second seed of the same case */
    CHECK(kb::cap_key(a, "k", "lib", 0, g) == kb::cap_key(a2, "k", "lib", 0, g));
    CHECK(kb::cap_key(a, "k", "lib", 0, g) != kb::cap_key(a, "k", "lib", 1, g));
}

/* Two weights with one identity seed can differ in encoding or plane selection; sharing a cached
 * relayout between them hands a kernel the wrong planes. Both keys hash the encoding and the
 * selection, not only the seed and size. */
TEST(weight_cache_keys_separate_encodings_and_selections) {
    KfOperand w;
    w.role = RAD_OPD_WEIGHT;
    w.dtype = RAD_F32;
    w.shape = { 256, 128 };
    w.has_enc = true;
    w.enc = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    w.sel = { 0, 1 };
    w.sel_rows = { 256, 2 };
    w.sel_cols = { 128, 1 };
    KfOperand other_sel = w;
    other_sel.sel = { 0 };
    other_sel.sel_rows = { 256 };
    other_sel.sel_cols = { 128 };
    KfOperand other_enc = w;
    other_enc.enc = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    CHECK_EQ(kf_weight_seed(w, 0), kf_weight_seed(other_enc, 0));
    CHECK(kb::host_weight_key(w, 0) != kb::host_weight_key(other_sel, 0));
    CHECK(kb::host_weight_key(w, 0) != kb::host_weight_key(other_enc, 0));
    CHECK(kb::device_weight_key("l", "k", w, 0, 64) != kb::device_weight_key("l", "k", other_enc, 0, 64));
    CHECK(kb::device_weight_key("l", "k", w, 0, 64) != kb::device_weight_key("l", "k2", w, 0, 64));
    CHECK(kb::device_weight_key("l", "k", w, 0, 64) == kb::device_weight_key("l", "k", w, 0, 64));
}

/* ================================================================== memory */

/* A machine with 3 GB free and 11 GB available has 11 GB to spend: page cache is reclaimable. */
TEST(the_default_budget_is_half_of_the_available_memory) {
    const std::string meminfo =
        "MemTotal:       30000000 kB\nMemFree:         3000000 kB\nMemAvailable:   11000000 kB\n";
    int64_t avail = 0;
    CHECK_EQ(kb::default_case_budget(meminfo, &avail), (int64_t)11000000 * 1024 / 2);
    CHECK_EQ(avail, (int64_t)11000000 * 1024);
    CHECK_EQ(kb::default_case_budget("MemAvailable: 100000000 kB\n", nullptr), (int64_t)8 << 30);
    CHECK_EQ(kb::default_case_budget("MemAvailable: 1000 kB\n", nullptr), (int64_t)512 << 20);
    CHECK_EQ(kb::default_case_budget("MemFree: 5000000 kB\n", &avail), (int64_t)8 << 30);
    CHECK_EQ(avail, (int64_t)0);
}

/* The budget estimate multiplies extents; one that overflows or is not positive must not come out
 * as a small number that passes the budget. */
TEST(a_case_size_that_cannot_be_represented_is_refused) {
    KfOperand d;
    d.dtype = RAD_BF16;
    d.shape = { 8, 64 };
    CHECK_EQ(kb::case_bytes({ d }), (int64_t)3 * 8 * 64 * 2);
    KfOperand absent = d;
    absent.absent = true;
    absent.shape = { 1 << 30, 1 << 30 };
    CHECK_EQ(kb::case_bytes({ d, absent }), (int64_t)3 * 8 * 64 * 2);
    KfOperand big = d;
    big.shape = { (int64_t)1 << 40, (int64_t)1 << 40 };
    CHECK_EQ(kb::case_bytes({ big }), (int64_t)-1);
    KfOperand neg = d;
    neg.shape = { 8, -1 };
    CHECK_EQ(kb::case_bytes({ neg }), (int64_t)-1);
}

/* ================================================================== red zones */

TEST(a_margin_overrun_is_measured_from_the_buffer) {
    std::vector<uint8_t> lo(256, 0xA5), hi(256, 0xA5);
    kb::Overrun o = kb::scan_margins(lo.data(), hi.data(), 256, 0xA5);
    CHECK_EQ(o.before, (int64_t)0);
    CHECK_EQ(o.after, (int64_t)0);
    lo[250] = 0;          /* 6 bytes before the start */
    hi[0] = 0;            /* the first byte past the end */
    hi[9] = 0;            /* and the tenth */
    o = kb::scan_margins(lo.data(), hi.data(), 256, 0xA5);
    CHECK_EQ(o.before, (int64_t)6);
    CHECK_EQ(o.after, (int64_t)10);
}

/* ================================================================== the exit code */

/* The red zones stand in for device ASan and the fusion check for substitutability; neither means
 * anything if the run still exits 0. Stale fixtures fail only when asked to (the ctests). */
TEST(an_overrun_a_fusion_mismatch_or_a_written_input_fails_the_run) {
    kb::Stats st;
    st.checked = st.passed = 3;
    CHECK_EQ(kb::exit_code(st, true), 0);
    kb::Stats s1 = st; s1.overrun = 1;
    CHECK_EQ(kb::exit_code(s1, false), 1);
    kb::Stats s2 = st; s2.fuse_failed = 1;
    CHECK_EQ(kb::exit_code(s2, false), 1);
    kb::Stats s3 = st; s3.input_written = 1;
    CHECK_EQ(kb::exit_code(s3, false), 1);
    kb::Stats s4 = st; s4.stale = 2;
    CHECK_EQ(kb::exit_code(s4, false), 0);
    CHECK_EQ(kb::exit_code(s4, true), 1);
    kb::Stats s5 = st; s5.vacuous = 4; s5.unserved = 2; s5.skipped = 9; s5.diverged = 1;
    CHECK_EQ(kb::exit_code(s5, true), 0);
    kb::Stats s6 = st; s6.failed = 1;
    CHECK_EQ(kb::exit_code(s6, false), 1);
}

/* The measurement gate (kb::bench_refusal): a time printed for an op whose cases were skipped is a
 * time for something nobody checked, which is how a run that checked nothing came to be read as a
 * speed result. An op is named only when the run has a time for it -- a skip of an op that was
 * never timed costs nothing -- and a run with no timed op at all refuses too, since a table of
 * nothing is not a table of zeroes. */
TEST(a_time_for_an_unchecked_op_refuses_the_measurement_gate) {
    kb::Stats st;
    st.checked = 12;
    const std::set<std::string> measured = { "add", "qsa_table" };
    const std::map<std::string, int> none;
    CHECK(kb::bench_refusal(st, measured, none, none).empty());
    /* Both ops were timed and checked. */
    std::map<std::string, int> skipped = { { "mul  (no libref kernel: no oracle exists for it)", 40 },
                                           { "qsa_table  (no libref kernel: no oracle exists for "
                                             "it)", 520 } };
    const std::string why = kb::bench_refusal(st, measured, skipped, none);
    CHECK(!why.empty());
    CHECK(why.find("qsa_table") != std::string::npos);
    CHECK(why.find("520") != std::string::npos);
    CHECK(why.find("mul") == std::string::npos);   /* skipped, but no time was reported for it */
    /* A stale case under a timed op is the same gap as a skipped one. */
    std::map<std::string, int> stale = { { "add  (libref now describes an output differently from "
                                           "its recording)", 1 } };
    const std::string bad = kb::bench_refusal(st, measured, none, stale);
    CHECK(bad.find("add") != std::string::npos);
    CHECK(kb::bench_refusal(st, {}, none, none).find("no op of this run") != std::string::npos);
    kb::Stats unchecked;
    CHECK(kb::bench_refusal(unchecked, measured, none, none).find("no case") != std::string::npos);
}

/* ================================================================== where the time went */

TEST(the_phase_line_names_each_phase_and_library) {
    kb::Phases ph;
    ph.add(kb::PH_DRAW, 1.25);
    ph.add(kb::PH_LAUNCH, 3.0);
    ph.launch_by_lib["libavx"] = 2.0;
    ph.launch_by_lib["libref"] = 1.0;
    {
        kb::PhaseTimer t(&ph, kb::PH_COMPARE);
        CHECK(t.stop() >= 0.0);
        CHECK_EQ(t.stop(), 0.0);                     /* stopped once */
    }
    const std::string s = kb::phase_line(ph, 5.0);
    CHECK(has(s, "draw 1.25 s"));
    CHECK(has(s, "launch 3.00 s (libavx 2.00 s, libref 1.00 s)"));
    CHECK(has(s, "5.00 s wall"));
    CHECK(!has(s, "upload"));
}

/* ================================================================== hashing */

/* The reference vectors from xxHash's own documentation (XXH64, seed 0). */
TEST(the_hash_is_xxh64) {
    CHECK_EQ(kb::hash64("", 0, 0), 0xEF46DB3751D8E999ull);
    CHECK_EQ(kb::hash64("a", 1, 0), 0xD24EC4F1A98C6E5Bull);
    CHECK_EQ(kb::hash64("abc", 3, 0), 0x44BC2CF5AD770999ull);
    const char* s = "Nobody inspects the spammish repetition";
    CHECK_EQ(kb::hash64(s, std::strlen(s), 0), 0xFBCEA83C8A378BF1ull);
}

/* A fingerprint is blocks hashed on every core and combined in order, so it must equal the same
 * blocks hashed on one thread. */
TEST(a_fingerprint_does_not_depend_on_the_thread_count) {
    const size_t block = (size_t)4 << 20;
    std::vector<uint8_t> buf(block * 3 + 1234);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(i * 2654435761u >> 13);
    uint64_t want = (uint64_t)buf.size();
    for (size_t b = 0; b * block < buf.size(); ++b)
        want = kb::hash_combine(want, kb::hash64(buf.data() + b * block,
                                                 std::min(block, buf.size() - b * block), b));
    CHECK_EQ(kb::fingerprint(buf.data(), buf.size()), want);
    buf[block * 2 + 7] ^= 1;
    CHECK(kb::fingerprint(buf.data(), buf.size()) != want);
}

/* Any change to what the reference computes from misses the cache: its code, the row, the case's
 * parameters to the last bit, its seed, an operand's shape, or one input byte. */
TEST(the_reference_cache_key_changes_with_everything_the_answer_depends_on) {
    KfCase c;
    c.op = "rmsnorm";
    c.seed = 77;
    KfOperand x;
    x.role = RAD_OPD_IN;
    x.dtype = RAD_BF16;
    x.shape = { 4, 64 };
    c.opd.push_back(x);
    Geometry g = geom({ { "M", "4" }, { "n", "64" } });
    g.set_f("eps", 1e-6);
    const uint64_t k = kb::ref_cache_key(1, "ref_rmsnorm", c, g, 1234);
    CHECK_EQ(k, kb::ref_cache_key(1, "ref_rmsnorm", c, g, 1234));
    CHECK(k != kb::ref_cache_key(2, "ref_rmsnorm", c, g, 1234));
    CHECK(k != kb::ref_cache_key(1, "ref_rmsnorm_p", c, g, 1234));
    CHECK(k != kb::ref_cache_key(1, "ref_rmsnorm", c, g, 1235));
    Geometry g2 = g;
    g2.set_f("eps", std::nextafter(1e-6, 1.0));
    CHECK(k != kb::ref_cache_key(1, "ref_rmsnorm", c, g2, 1234));
    KfCase c2 = c;
    c2.seed = 78;
    CHECK(k != kb::ref_cache_key(1, "ref_rmsnorm", c2, g, 1234));
    KfCase c3 = c;
    c3.opd[0].shape = { 4, 65 };
    CHECK(k != kb::ref_cache_key(1, "ref_rmsnorm", c3, g, 1234));
}

/* A hit returns exactly what was stored; a damaged, truncated or foreign entry is a miss, never a
 * pass. */
TEST(a_cached_reference_round_trips_and_a_damaged_entry_is_a_miss) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    kb::RefCache cache;
    cache.open(dir, 0x1234);
    REQUIRE(cache.enabled());
    std::vector<kb::Digest> outs(2);
    outs[0] = digest(normals(3000, 1), 5);
    outs[1] = digest(normals(10, 2), 6, false);
    cache.put(42, outs);
    CHECK_EQ(cache.writes(), 1);
    std::vector<kb::Digest> back;
    REQUIRE(cache.get(42, &back));
    REQUIRE_EQ(back.size(), (size_t)2);
    for (size_t i = 0; i < 2; ++i) {
        CHECK_EQ(back[i].n, outs[i].n);
        CHECK_EQ(back[i].sumsq, outs[i].sumsq);
        CHECK(back[i].sample == outs[i].sample);
        CHECK(back[i].sketch == outs[i].sketch);
    }
    CHECK(!cache.get(43, &back));

    const std::string path = cache.dir() + "/000000000000002a.rkc";
    FILE* f = std::fopen(path.c_str(), "r+b");
    REQUIRE(f != nullptr);
    std::fseek(f, 40, SEEK_SET);
    const int c = std::fgetc(f);
    std::fseek(f, 40, SEEK_SET);
    std::fputc(c ^ 0x10, f);
    std::fclose(f);
    CHECK(!cache.get(42, &back));
    cache.put(42, outs);
    CHECK(cache.get(42, &back));
    CHECK_EQ(::truncate(path.c_str(), 30), 0);
    CHECK(!cache.get(42, &back));

    /* Another identity is another directory: nothing carries over. */
    cache.put(42, outs);
    kb::RefCache other;
    other.open(dir, 0x5678);
    CHECK(!other.get(42, &back));
    kb::RefCache off;
    off.open(dir, 0);
    CHECK(!off.enabled());
}

/* A rebuilt reference leaves its old answers behind under the old identity; only the current one
 * and the one before it are kept. */
TEST(the_cache_keeps_only_recent_identities) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    std::vector<kb::Digest> outs(1);
    outs[0] = digest(normals(16, 1), 1);
    for (uint64_t id : { 0xA1ull, 0xA2ull, 0xA3ull }) {
        kb::RefCache c;
        c.open(dir, id);
        c.put(1, outs);
    }
    struct stat st{};
    CHECK(::stat((dir + "/00000000000000a3").c_str(), &st) == 0);
    int kept = 0;
    for (const char* d : { "/00000000000000a1", "/00000000000000a2" })
        if (::stat((dir + d).c_str(), &st) == 0) ++kept;
    CHECK(kept <= 1);
}

/* The identity is the library's build-id and those of what it links; a library with none is not
 * cached at all rather than cached under a name. */
TEST(the_code_identity_follows_build_ids) {
    const std::string ref = home() + "/kernels/libref.so";
    void* h = ::dlopen(ref.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) ARTIFACT_MISSING(ref + " (does not load: " + dlerror() + ")");
    CHECK(kb::build_id_of(ref).size() >= 16);
    const uint64_t id = kb::code_identity(ref, {});
    CHECK(id != 0);
    CHECK_EQ(id, kb::code_identity(ref, {}));
    CHECK_EQ(kb::code_identity(home() + "/kernels/no_such.so", {}), (uint64_t)0);
    ::dlclose(h);
}

/* ================================================================== the coverage ratchet */

TEST(a_lost_row_fails_the_ratchet_unless_a_memory_skip_kept_it_from_running) {
    const kb::Ratchet r = kb::ratchet({ "a", "b", "c" }, { "a", "d" }, { "b" });
    CHECK(r.lost == std::vector<std::string>{ "c" });
    CHECK(r.gained == std::vector<std::string>{ "d" });
    CHECK(kb::ratchet({ "a" }, { "a" }, {}).lost.empty());
}

TEST(a_rows_file_round_trips_and_ignores_comments) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/rows.txt";
    CHECK_OK(kb::write_rows_file(path, { "r2", "r1" }));
    bool exists = false;
    CHECK((kb::read_rows_file(path, &exists) == std::set<std::string>{ "r1", "r2" }));
    CHECK(exists);
    kb::read_rows_file(dir + "/none.txt", &exists);
    CHECK(!exists);
    CHECK_EQ(kb::rows_file_name("data/kernels_moe.rkb", "libavx"),
             std::string("kbench_rows_kernels_moe_libavx.txt"));
}

/* ================================================================== synthetic geometry */

TEST(a_geometry_is_moved_into_a_rows_constraints) {
    const RadConstraint cs[] = {
        RAD_CEQ("block_size", 16), RAD_CLE("M", 4), RAD_CGE("n", 128), RAD_CDIV("K", 64),
        RAD_CIN("dtype", "fp8a8 w4a8"), RAD_CIN("causal", "1"),
    };
    RadKernelInfo ki{};
    ki.constraints = cs;
    ki.n_constraints = (int)(sizeof cs / sizeof cs[0]);
    Geometry g = geom({ { "M", "64" }, { "n", "32" }, { "K", "100" }, { "dtype", "bf16" },
                        { "causal", "0" } });
    REQUIRE(kb::satisfy_constraints(&ki, &g));
    long long v = 0;
    CHECK(g.get_i("block_size", &v) && v == 16);
    CHECK(g.get_i("M", &v) && v == 4);
    CHECK(g.get_i("n", &v) && v == 128);
    CHECK(g.get_i("K", &v) && v == 128);
    CHECK(g.get_s("dtype") && std::string(g.get_s("dtype")) == "fp8a8");
    CHECK(g.get_i("causal", &v) && v == 1);
    /* A fact about the machine is never invented. */
    const RadConstraint fact[] = { RAD_CEQ("device.p2p", 1) };
    RadKernelInfo kf{};
    kf.constraints = fact;
    kf.n_constraints = 1;
    Geometry g2;
    CHECK(!kb::satisfy_constraints(&kf, &g2));
}

/* ================================================================== skeleton containers */

/* A skeleton opens as the same container (metadata, directory, encodings) while its data region is
 * a hole: what a recording reads, at a fraction of the space. */
TEST(a_skeleton_container_keeps_every_table_and_drops_the_data) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string src = dir + "/model.rad", dst = dir + "/skeleton.rad";
    {
        RadWriter w;
        REQUIRE_EQ(w.begin(src.c_str()), RAD_OK);
        w.set_arch("radtest");
        w.set_model_name("skeleton-test");
        w.meta_i("n_embd", 1024);
        w.meta_s("note", "kept");
        RadWriter::Weight wt;
        wt.name = "big.weight";
        wt.enc = rad_enc_plain(RAD_BF16);
        wt.rank = 2;
        wt.shape[0] = 2048;
        wt.shape[1] = 1024;
        std::vector<uint16_t> plane((size_t)2048 * 1024, 0x3F80);   /* bf16 1.0 */
        const void* planes[] = { plane.data() };
        REQUIRE(w.add_weight(wt, planes) >= 0);
        REQUIRE_EQ(w.finish(), RAD_OK);
    }
    std::string err;
    REQUIRE_EQ(kb::write_skeleton(src, dst, &err), RAD_OK);
    struct stat a{}, b{};
    REQUIRE(::stat(src.c_str(), &a) == 0 && ::stat(dst.c_str(), &b) == 0);
    CHECK_EQ(a.st_size, b.st_size);
    CHECK((int64_t)b.st_blocks * 512 < (int64_t)b.st_size / 4);
    RadFile f;
    REQUIRE_EQ(f.open(dst.c_str()), RAD_OK);
    int64_t n = 0;
    CHECK(f.get_i("n_embd", &n) && n == 1024);
    CHECK_EQ(std::string(f.get_s("note", "")), std::string("kept"));
    const RadFileEntry* e = f.find("big.weight");
    REQUIRE(e != nullptr);
    CHECK_EQ(e->shape[0], (int64_t)2048);
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    CHECK(rad_enc_equal(&f.encoding(*e), &plain));
    const RadFilePlane& p = f.plane(*e, 0);
    const uint8_t* data = f.data(p);
    bool zero = true;
    for (uint64_t i = 0; i < p.bytes; i += 4096) zero = zero && data[i] == 0;
    CHECK(zero);
    CHECK(kb::write_skeleton(dir + "/none.rad", dir + "/x.rad", &err) != RAD_OK);
}

/* ================================================================== timing */

namespace {

/* Little-endian struct packing of a Python struct format ("<HBBQ..."): H 2 bytes, B 1, I 4, Q 8,
 * digits repeat the next code. Builds a gpu_metrics table field by field, independently of the
 * parser's offsets. */
std::vector<uint8_t> pack(const char* f, const std::vector<uint64_t>& vals) {
    std::vector<uint8_t> out;
    size_t vi = 0;
    for (const char* c = f; *c; ++c) {
        if (*c == '<') continue;
        int rep = 0;
        while (*c >= '0' && *c <= '9') rep = rep * 10 + (*c++ - '0');
        if (rep == 0) rep = 1;
        const int w = *c == 'B' ? 1 : *c == 'H' ? 2 : *c == 'I' ? 4 : 8;
        for (int r = 0; r < rep; ++r) {
            const uint64_t v = vi < vals.size() ? vals[vi++] : 0;
            for (int b = 0; b < w; ++b) out.push_back((uint8_t)(v >> (8 * b)));
        }
    }
    return out;
}

std::vector<uint8_t> from_hex(const char* h) {
    std::vector<uint8_t> out;
    for (size_t i = 0; h[i] && h[i + 1]; i += 2)
        out.push_back((uint8_t)std::strtoul(std::string(h + i, 2).c_str(), nullptr, 16));
    return out;
}

/* One drawn index operand. */
KfOperand index_opd(std::vector<int64_t> shape, int64_t idx_max) {
    KfOperand d;
    d.role = RAD_OPD_IN;
    d.dtype = RAD_I32;
    d.shape = std::move(shape);
    d.is_index = true;
    d.idx_max = idx_max;
    return d;
}

std::vector<int32_t> as_i32(const std::vector<uint8_t>& b) {
    std::vector<int32_t> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

}  /* namespace */

/* The reported median comes with an interval from order statistics alone, so it holds whatever the
 * distribution of batch times (a clock step, a stray interrupt). At n = 20 the ranks are
 * floor(10 - 0.98 sqrt(20)) = 5 and ceil(11 + 0.98 sqrt(20)) = 16, worked out by hand. */
TEST(the_median_and_its_interval_are_order_statistics) {
    std::vector<double> v;
    for (int i = 0; i < 20; ++i) v.push_back((double)((7 * i) % 20 + 1));   /* 1..20, shuffled */
    const kb::TimeStats s = kb::time_stats(v);
    CHECK_EQ(s.n, 20);
    CHECK_EQ(s.median, 10.5);
    CHECK_EQ(s.min, 1.0);
    CHECK_EQ(s.max, 20.0);
    CHECK_EQ(s.p90, 18.0);
    CHECK_EQ(s.ci_lo, 5.0);
    CHECK_EQ(s.ci_hi, 16.0);
    CHECK_NEAR(s.ci_pct, 100.0 * 11.0 / 2.0 / 10.5, 1e-9);
}

/* With few samples no order statistic inside the range gives 95% coverage, so the interval is the
 * range itself up to ten, and from eleven it excludes the extremes: one wild batch cannot move it.
 * One sample has no interval. */
TEST(the_interval_is_the_range_up_to_ten_samples_and_ignores_an_outlier_from_eleven) {
    const kb::TimeStats one = kb::time_stats({ 7 });
    CHECK_EQ(one.median, 7.0);
    CHECK_EQ(one.ci_pct, -1.0);
    const kb::TimeStats five = kb::time_stats({ 3, 1, 2, 5, 4 });
    CHECK_EQ(five.median, 3.0);
    CHECK_EQ(five.ci_lo, 1.0);
    CHECK_EQ(five.ci_hi, 5.0);
    CHECK_EQ(five.p90, 5.0);
    std::vector<double> ten;
    for (int i = 1; i <= 10; ++i) ten.push_back(i);
    const kb::TimeStats t = kb::time_stats(ten);
    CHECK_EQ(t.ci_lo, 1.0);
    CHECK_EQ(t.ci_hi, 10.0);
    std::vector<double> eleven;
    for (int i = 0; i < 10; ++i) eleven.push_back(100.0 + 0.1 * i);
    eleven.push_back(1e6);
    const kb::TimeStats e = kb::time_stats(eleven);
    CHECK_NEAR(e.ci_lo, 100.1, 1e-9);
    CHECK_NEAR(e.ci_hi, 100.9, 1e-9);
    CHECK(e.ci_pct < 1.0);
    CHECK_EQ(e.median, 100.5);
}

/* A case is timed until its median is known to 1% or the cap is reached, never on fewer than the
 * minimum. A median at or below zero (a restore-dominated difference) has no relative interval and
 * runs to the cap. Non-finite samples are dropped. */
TEST(timing_stops_at_a_tight_interval_or_at_the_cap) {
    const std::vector<double> tight = { 100, 100.4, 99.6, 100.2, 99.8 };
    CHECK(kb::timing_done(tight, 5, 20, 1.0));
    CHECK(!kb::timing_done({ 100, 100.4, 99.6, 100.2 }, 5, 20, 1.0));
    const std::vector<double> wide = { 100, 110, 90, 105, 95 };
    CHECK(!kb::timing_done(wide, 5, 20, 1.0));
    CHECK(kb::timing_done(wide, 5, 5, 1.0));
    CHECK(kb::timing_done({ 1, 2 }, 5, 2, 1.0));
    CHECK(!kb::timing_done({ -1, -1.1, -0.9, -1, -1 }, 5, 20, 1.0));
    CHECK_EQ(kb::time_stats({ std::nan(""), 1, 2, 3 }).n, 3);
}

/* A batch lasts a millisecond and fifty launch floors, so the two barriers around it are a small
 * share; it never exceeds what fits behind the gate, and --bench-iters fixes it. */
TEST(a_batch_lasts_a_millisecond_and_fifty_floors_within_its_room) {
    CHECK_EQ(kb::batch_target_us(2.0), 1000.0);
    CHECK_EQ(kb::batch_target_us(30.0), 1500.0);
    CHECK_EQ(kb::batch_target_us(-1.0), 1000.0);
    CHECK_EQ(kb::batch_launches(4.0, 1000.0, 2000, 0), (int64_t)250);
    CHECK_EQ(kb::batch_launches(4.0, 1000.0, 100, 0), (int64_t)100);
    CHECK_EQ(kb::batch_launches(5000.0, 1000.0, 2000, 0), (int64_t)1);
    CHECK_EQ(kb::batch_launches(4.0, 1000.0, 2000, 7), (int64_t)7);
    CHECK_EQ(kb::batch_launches(4.0, 1000.0, 3, 7), (int64_t)3);
    CHECK_EQ(kb::batch_launches(1e-4, 1000.0, 0, 0), (int64_t)4096);
    CHECK_EQ(kb::batch_launches(-1.0, 1000.0, 2000, 0), (int64_t)1);
}

/* A restore-dominated batch or a time under the floor is marked; a measured time is not. */
TEST(a_restore_dominated_or_sub_floor_time_is_an_order_of_magnitude) {
    CHECK(kb::timing_caveat(10.0, 2.0, 40.0).empty());
    CHECK(kb::timing_caveat(10.0, 2.0, -1.0).empty());          /* no restores */
    CHECK(kb::timing_caveat(2.0, 2.07, 50.0).empty());          /* within 5% of the floor */
    CHECK(kb::timing_caveat(1.0, -1.0, 10.0).empty());          /* no floor measured */
    CHECK(has(kb::timing_caveat(10.0, 2.0, 80.0), "restore-dominated (80% of the batch)"));
    CHECK(has(kb::timing_caveat(0.85, 2.07, 98.0), "under the launch floor (2.07 us)"));
    CHECK(has(kb::timing_caveat(1.0, 2.07, -1.0), "under the launch floor"));
}

/* R copies leave R - 1 others between two uses of one, so (R - 1) launches' reads must reach the
 * footprint and R - 2 must not (no more memory than needed). An operand set as large as the
 * footprint evicts itself: one copy. */
TEST(rotated_copies_put_the_footprint_between_two_uses_of_one) {
    CHECK_EQ(kb::rotation_stride(1000), (int64_t)1024);
    CHECK_EQ(kb::rotation_stride(256), (int64_t)256);
    CHECK_EQ(kb::rotation_stride(0), (int64_t)0);
    const int64_t fp = (int64_t)256 << 20;
    for (int64_t pl : { (int64_t)10240, (int64_t)100 << 20, (int64_t)3 << 20, (int64_t)77777 }) {
        const int64_t r = kb::rotation_copies(pl, fp);
        CHECK((r - 1) * pl >= fp);
        CHECK((r - 2) * pl < fp);
    }
    CHECK_EQ(kb::rotation_copies(10240, fp), (int64_t)26216);
    CHECK_EQ(kb::rotation_copies(fp, fp), (int64_t)1);
    CHECK_EQ(kb::rotation_copies(fp + 1, fp), (int64_t)1);
    CHECK_EQ(kb::rotation_copies(4096, 0), (int64_t)1);
    CHECK_EQ(kb::default_footprint((int64_t)72 << 20), fp);
    CHECK_EQ(kb::default_footprint((int64_t)200 << 20), (int64_t)400 << 20);
    CHECK_EQ(kb::default_footprint(0), fp);
}

/* Weights stream once a step in the engine and big inputs are state read once a step (a KV cache);
 * small inputs are activations the previous op left in the cache, and written operands are
 * restored, not rotated. */
TEST(weights_and_big_inputs_rotate_and_written_operands_do_not) {
    const int64_t big = 8 << 20;
    CHECK(kb::rotated(RAD_OPD_WEIGHT, 1, big));
    CHECK(kb::rotated(RAD_OPD_WTAB, 1, big));
    CHECK(kb::rotated(RAD_OPD_IN, big, big));
    CHECK(!kb::rotated(RAD_OPD_IN, big - 1, big));
    CHECK(!kb::rotated(RAD_OPD_IN, (int64_t)1 << 40, 0));
    CHECK(!kb::rotated(RAD_OPD_OUT, (int64_t)1 << 40, big));
    CHECK(!kb::rotated(RAD_OPD_INOUT, (int64_t)1 << 40, big));
}

/* Each launch's pointer table is a permutation of the table's slots, a different one for the next
 * launch, and the same for the same arguments (a rerun reads the same bytes). */
TEST(a_weight_tables_slots_are_permuted_differently_for_each_launch) {
    std::vector<int64_t> a, b, again;
    for (int64_t l = 0; l < 8; ++l) {
        kb::table_permutation(64, 99, l, &a);
        REQUIRE_EQ(a.size(), (size_t)64);
        std::vector<int64_t> s = a;
        std::sort(s.begin(), s.end());
        for (int64_t i = 0; i < 64; ++i) CHECK_EQ(s[(size_t)i], i);
    }
    kb::table_permutation(64, 99, 0, &a);
    kb::table_permutation(64, 99, 1, &b);
    kb::table_permutation(64, 99, 0, &again);
    CHECK(a != b);
    CHECK(a == again);
}

/* Each pointer table names every slot of one copy exactly once: table l points into copy l mod R
 * through the permutation for l, and the table count is a multiple of R, so launch i's table and
 * its other rotated operands agree on the copy. */
TEST(a_weight_tables_pointers_name_every_slot_of_one_copy) {
    CHECK_EQ(kb::table_count(1), (int64_t)64);
    CHECK_EQ(kb::table_count(3), (int64_t)66);
    CHECK_EQ(kb::table_count(100), (int64_t)100);
    CHECK_EQ(kb::table_count(0), (int64_t)64);
    const std::vector<uint64_t> bases = { 0x10000, 0x90000, 0x110000 };
    std::vector<int64_t> perm;
    for (int64_t l = 0; l < 6; ++l) {
        std::vector<uint64_t> p = { 42 };
        kb::table_pointers(bases, 8, 0x100, 7, l, &p);
        REQUIRE_EQ(p.size(), (size_t)9);
        CHECK_EQ(p[0], (uint64_t)42);
        kb::table_permutation(8, 7, l, &perm);
        const uint64_t base = bases[(size_t)(l % 3)];
        std::set<uint64_t> slots;
        for (int64_t e = 0; e < 8; ++e) {
            const uint64_t x = p[(size_t)e + 1];
            CHECK_EQ(x, base + (uint64_t)perm[(size_t)e] * 0x100);
            CHECK(x >= base && x < base + 8 * 0x100 && (x - base) % 0x100 == 0);
            slots.insert(x);
        }
        CHECK_EQ(slots.size(), (size_t)8);
    }
}

/* Skewed routing places every row, respects the per-expert cap (a token visits an expert once), and
 * is skewed: with exponent 0.5 over 128 experts the hottest takes 1 / sum(r^-0.5) = 4.7% of the
 * rows, six times the mean. The counts by rank do not depend on the seed, only which expert gets
 * which. At decode (one token, eight slots, cap one) it picks eight distinct experts. */
TEST(skewed_routing_places_every_row_within_the_cap) {
    const std::vector<int64_t> c = kb::skewed_counts(128, 16384, 2048, 5);
    REQUIRE_EQ(c.size(), (size_t)128);
    int64_t sum = 0, mx = 0;
    for (int64_t x : c) { sum += x; mx = std::max(mx, x); CHECK(x >= 0 && x <= 2048); }
    CHECK_EQ(sum, (int64_t)16384);
    CHECK(mx >= 5 * 128 && mx <= 7 * 128);
    std::vector<int64_t> other = kb::skewed_counts(128, 16384, 2048, 6);
    CHECK(other != c);
    std::vector<int64_t> s1 = c, s2 = other;
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());
    CHECK(s1 == s2);
    CHECK(kb::skewed_counts(128, 16384, 2048, 5) == c);
    const std::vector<int64_t> decode = kb::skewed_counts(128, 8, 1, 3);
    int64_t ones = 0, zeros = 0;
    for (int64_t x : decode) { ones += x == 1; zeros += x == 0; }
    CHECK_EQ(ones, (int64_t)8);
    CHECK_EQ(zeros, (int64_t)120);
    /* A cap no partition can meet is dropped, not violated by losing rows. */
    int64_t all = 0;
    for (int64_t x : kb::skewed_counts(4, 100, 10, 1)) all += x;
    CHECK_EQ(all, (int64_t)100);
}

/* The timing pass's own inputs: an expert histogram skewed within its contract (offsets from 0 to
 * the row count, non-decreasing, each run at most M), routing ids distinct within a row and
 * concentrated on a few experts, a block table as a permutation. Anything else keeps the recorded
 * draw. */
TEST(timing_inputs_skew_routing_and_permute_block_tables_and_leave_the_rest) {
    Geometry g;
    g.set_i("M", 2048);
    g.set_i("n_expert", 128);
    g.set_i("top_k", 8);
    std::vector<uint8_t> out;

    KfOperand off = index_opd({ 129 }, 2048 * 8 + 1);
    off.idx_prefix = true;
    REQUIRE(kb::timing_input(off, g, 11, &out));
    const std::vector<int32_t> o = as_i32(out);
    REQUIRE_EQ(o.size(), (size_t)129);
    CHECK_EQ(o.front(), 0);
    CHECK_EQ(o.back(), 2048 * 8);
    int32_t widest = 0;
    for (size_t i = 1; i < o.size(); ++i) {
        CHECK(o[i] >= o[i - 1]);
        widest = std::max(widest, o[i] - o[i - 1]);
    }
    CHECK(widest <= 2048 && widest > 3 * 128);

    KfOperand ids = index_opd({ 64, 8 }, 128);
    REQUIRE(kb::timing_input(ids, g, 12, &out));
    const std::vector<int32_t> id = as_i32(out);
    REQUIRE_EQ(id.size(), (size_t)512);
    std::vector<int> hist(128, 0);
    for (int r = 0; r < 64; ++r) {
        std::set<int32_t> row;
        for (int j = 0; j < 8; ++j) {
            const int32_t e = id[(size_t)(r * 8 + j)];
            CHECK(e >= 0 && e < 128);
            row.insert(e);
            if (e >= 0 && e < 128) ++hist[(size_t)e];
        }
        CHECK_EQ(row.size(), (size_t)8);
    }
    CHECK(*std::max_element(hist.begin(), hist.end()) >= 12);   /* the mean is 4 */

    KfOperand table = index_opd({ 1, 1024 }, 1024);
    REQUIRE(kb::timing_input(table, g, 13, &out));
    std::vector<int32_t> t = as_i32(out);
    std::sort(t.begin(), t.end());
    for (int32_t i = 0; i < 1024; ++i) CHECK_EQ(t[(size_t)i], i);

    std::vector<uint8_t> untouched = { 7 };
    CHECK(!kb::timing_input(index_opd({ 64 }, 64), g, 1, &untouched));     /* positions */
    KfOperand uniq = index_opd({ 1, 64 }, 64);
    uniq.idx_unique = true;
    CHECK(!kb::timing_input(uniq, g, 1, &untouched));
    KfOperand cu = index_opd({ 2, 2 }, 4);
    cu.idx_cu = true;
    CHECK(!kb::timing_input(cu, g, 1, &untouched));
    KfOperand written = table;
    written.role = RAD_OPD_INOUT;
    CHECK(!kb::timing_input(written, g, 1, &untouched));
    KfOperand act = table;
    act.is_index = false;
    CHECK(!kb::timing_input(act, g, 1, &untouched));
    CHECK(untouched == std::vector<uint8_t>{ 7 });
}

/* A v1.3 table (RDNA3/RDNA4 discrete cards) built field by field from the layout cardclock.py
 * unpacks and checks against hwmon on the R9700: the averaged and current graphics clock, the
 * hotspot, power, activity and the ASIC-independent throttler bits come from their own fields.
 * 0xFFFF is "not reported". */
TEST(a_v1_3_metrics_table_reads_clock_temperature_and_throttlers) {
    const char* v13 = "<HBB6H4HQQ7H7HIHHHHII4HQHHHHQ";
    std::vector<uint64_t> f = { 120, 1, 3,
                                45, 62, 70, 50, 48, 52,                 /* temperatures */
                                97, 40, 0, 290,                         /* activity, power */
                                0, 0,                                   /* energy, clock counter */
                                2310, 1200, 1250, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                2350, 1210, 1250, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                0x10,                                   /* throttle_status */
                                1500, 16, 160, 0, 0, 0,
                                0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0,
                                1100, 750, 1250, 0,
                                ((uint64_t)1 << 36) | 1u };
    const std::vector<uint8_t> blob = pack(v13, f);
    REQUIRE_EQ(blob.size(), (size_t)120);
    kb::GpuMetrics m;
    REQUIRE(kb::parse_gpu_metrics(blob.data(), blob.size(), &m));
    CHECK_EQ(m.gfxclk_mhz, 2310.0);
    CHECK_EQ(m.gfxclk_now_mhz, 2350.0);
    CHECK_EQ(m.hotspot_c, 62.0);
    CHECK_EQ(m.power_w, 290.0);
    CHECK_EQ(m.gfx_busy_pct, 97.0);
    CHECK_EQ(m.throttle, ((uint64_t)1 << 36) | 1u);
    CHECK_EQ(kb::throttle_names(m.throttle), std::string("PPT0+TEMP_HOTSPOT"));
    CHECK_EQ(kb::throttle_names((uint64_t)1 << 60), std::string("BIT60"));
    CHECK_EQ(kb::throttle_names(0), std::string());
    f[15] = 0xFFFF;
    const std::vector<uint8_t> unreported = pack(v13, f);
    REQUIRE(kb::parse_gpu_metrics(unreported.data(), unreported.size(), &m));
    CHECK_EQ(m.gfxclk_mhz, -1.0);
}

/* Another layout is refused, never read at v1.3's offsets: a v2.1 table captured from a Ryzen 7000
 * integrated GPU (1002:13c0), a v1.4 header, and a v1.3 table cut short. */
TEST(another_metrics_layout_is_refused_not_misread) {
    const std::vector<uint8_t> apu = from_hex(
        "7800020130113d13ffffffffffffffffffffffffffffffffffffffff0000000051ce28796e7004005d00ffff"
        "1b3e65f3ffffffffffffffffffffffffffffffff5802b004b80bb80b19001900ffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffff00080000ffffffffffffffff");
    REQUIRE_EQ(apu.size(), (size_t)120);
    kb::GpuMetrics m;
    CHECK(!kb::parse_gpu_metrics(apu.data(), apu.size(), &m));
    CHECK_EQ(m.format, 2);
    CHECK_EQ(m.content, 1);
    std::vector<uint8_t> v14 = pack("<HBB", { 128, 1, 4 });
    v14.resize(128, 0);
    CHECK(!kb::parse_gpu_metrics(v14.data(), v14.size(), &m));
    std::vector<uint8_t> cut = pack("<HBB", { 120, 1, 3 });
    cut.resize(100, 0);
    CHECK(!kb::parse_gpu_metrics(cut.data(), cut.size(), &m));
    CHECK(!kb::parse_gpu_metrics(nullptr, 0, &m));
}

/* Cold copies run at memory speed at every size; a smaller cold probe much faster than the largest
 * was a cache hit, which would make every cold timing one too. The largest probe is past every
 * cache warm or cold, so the two must agree. Each refusal names its reason. */
TEST(the_calibration_refuses_a_cold_probe_served_from_a_cache) {
    const int64_t MiB = 1 << 20;
    std::vector<kb::BwProbe> p = { { 4 * MiB, 2400, 590 }, { 16 * MiB, 1800, 600 },
                                   { 64 * MiB, 900, 595 }, { 256 * MiB, 610, 600 } };
    CHECK_EQ(kb::bandwidth_refusal(p), std::string());
    std::vector<kb::BwProbe> hit = p;
    hit[0].cold_gbs = 1500;
    const std::string why = kb::bandwidth_refusal(hit);
    CHECK(has(why, "4.00 MiB") && has(why, "served from a cache"));
    std::vector<kb::BwProbe> clock = p;
    clock[3].warm_gbs = 1200;
    CHECK(has(kb::bandwidth_refusal(clock), "clock moved"));
    std::vector<kb::BwProbe> none = p;
    none[1].warm_gbs = 0;
    CHECK(has(kb::bandwidth_refusal(none), "no rate"));
    CHECK(has(kb::bandwidth_refusal({}), "no bandwidth probe"));
}

RAD_TEST_MAIN()
