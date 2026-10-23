/* diagnose_test.cpp -- the arithmetic `radiance --diagnose DIR` reports, against numbers computed
 * by hand, and the step record it is computed from.
 *
 * These are the cases the report is built on. Each one says which quantity it is about, so a
 * failure names the number rather than the file: an order statistic, a mean, an acceptance at a
 * draft position, the pool over a category, the parity comparison's first mismatch, the refusal of
 * a baseline that is not the same workload. The numbers below are written out where they come from
 * -- a percentile at 0.9 of four samples is 3.7 because the convention says so, and the convention
 * is in core/diagnose.h.
 *
 * The rendering is checked with the arithmetic rather than separately: every expected number is
 * looked up in the JSON the report writes, so a value that is right in the C++ and wrong on disk
 * fails here too. A quantity the run could not measure is checked to be absent with a reason, which
 * is the property the report exists to have: nothing in it reads as a measurement of zero.
 *
 * The startup report's own tests are in tests/diag_test.cpp; this file is the self-measurement
 * (core/diagnose.h), which is a different thing that used to want the same name.
 */
#include "diagnose.h"
#include "rad_test.h"
#include "sched/steplog.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using nlohmann::json;
using namespace rad;
using diagnose::Acceptance;
using diagnose::Dist;
using diagnose::Measure;

namespace {

/* Look up the JSON the report writes, so the number and its rendering are one check. */
json rendered(const diagnose::Report& r) { return json::parse(diagnose::report_json(r)); }

/* A report with one point and one request, whose step-time series is {1,2,3,4} microseconds:
 * mean 2.5, p50 2.5, p90 3.7, p99 3.97, max 4. Every case that reads a number out of the JSON
 * reads one of these. */
diagnose::Report sample_report(std::string name = "category:prose",
                               std::string category = "prose") {
    diagnose::Report r;
    r.workload = 1234;
    r.salt = 17;
    r.engine = { { "backend", "host" }, { "drafter", "none" } };

    diagnose::PointReport p;
    p.name = std::move(name);
    p.category = std::move(category);
    p.concurrency = 1;
    p.steps = 4;
    p.requests = 1;
    p.prompt_tokens = 5;
    p.out_tokens = 4;
    std::vector<double> s = { 1, 2, 3, 4 };
    p.step_us_decode = diagnose::summarize(s, "no decode step ran in this point");

    diagnose::RequestRecord q;
    q.id = 1;
    q.point = p.name;
    q.category = p.category;
    q.prompt = "prose";
    q.prompt_tokens = 5;
    q.tokens = { 5, 6, 7, 8 };
    q.finish = "length";
    p.reqs.push_back(std::move(q));

    r.points.push_back(std::move(p));
    return r;
}

}  /* namespace */

/* ------------------------------------------------------------------ order statistics
 * The convention (core/diagnose.h): at p/100 * (n-1) of the sorted samples, between the two
 * neighbours when that is not a whole index. Written out for each case below. */
TEST(diagnose_percentile_interpolates_between_four_samples) {
    const std::vector<double> s = { 1, 2, 3, 4 };
    /* p50: position 0.5 * 3 = 1.5 -> 2 + 0.5 * (3 - 2) = 2.5 */
    CHECK_NEAR(diagnose::percentile(s, 50.0).value, 2.5, 1e-12);
    /* p90: position 0.9 * 3 = 2.7 -> 3 + 0.7 * (4 - 3) = 3.7 */
    CHECK_NEAR(diagnose::percentile(s, 90.0).value, 3.7, 1e-12);
    /* p99: position 0.99 * 3 = 2.97 -> 3 + 0.97 * 1 = 3.97 */
    CHECK_NEAR(diagnose::percentile(s, 99.0).value, 3.97, 1e-12);
    /* p100 lands past the last index, so it is the last sample itself. */
    CHECK_NEAR(diagnose::percentile(s, 100.0).value, 4.0, 1e-12);
    CHECK_NEAR(diagnose::percentile(s, 0.0).value, 1.0, 1e-12);
}

TEST(diagnose_percentile_of_one_sample_is_that_sample_at_every_level) {
    const std::vector<double> s = { 7 };
    CHECK_NEAR(diagnose::percentile(s, 0.0).value, 7.0, 1e-12);
    CHECK_NEAR(diagnose::percentile(s, 50.0).value, 7.0, 1e-12);
    CHECK_NEAR(diagnose::percentile(s, 99.0).value, 7.0, 1e-12);
}

TEST(diagnose_percentile_of_nothing_is_absent_with_a_reason) {
    const std::vector<double> s;
    const Measure m = diagnose::percentile(s, 50.0);
    CHECK(!m.present);
    CHECK_EQ(m.value, 0.0);              /* never read as a measurement of zero */
    CHECK(!m.absent.empty());
}

TEST(diagnose_mean_of_a_step_series) {
    const std::vector<double> s = { 2, 4, 6 };
    /* (2 + 4 + 6) / 3 = 4 */
    CHECK_NEAR(diagnose::mean_of(s).value, 4.0, 1e-12);
    CHECK(!diagnose::mean_of({}).present);
}

TEST(diagnose_summarize_sorts_what_arrives_unsorted_and_counts_it) {
    std::vector<double> s = { 9, 1, 5 };
    const Dist d = diagnose::summarize(s, "nothing");
    /* Sorted: {1, 5, 9}. mean = 15 / 3 = 5; p50 at position 1.0 = 5;
     * p90 at position 1.8 = 5 + 0.8 * (9 - 5) = 8.2; p99 at 1.98 = 5 + 0.98 * 4 = 8.92. */
    CHECK_EQ(d.count, 3);
    CHECK_NEAR(d.mean.value, 5.0, 1e-12);
    CHECK_NEAR(d.p50.value, 5.0, 1e-12);
    CHECK_NEAR(d.p90.value, 8.2, 1e-12);
    CHECK_NEAR(d.p99.value, 8.92, 1e-12);
    CHECK_NEAR(d.max.value, 9.0, 1e-12);
    /* Ordered the way a distribution has to be: this is the invariant review() re-checks on the
     * way out of a run. */
    CHECK(d.p50.value <= d.p90.value);
    CHECK(d.p90.value <= d.p99.value);
    CHECK(d.p99.value <= d.max.value);
    CHECK(d.mean.value >= 1.0 && d.mean.value <= 9.0);
}

TEST(diagnose_an_empty_distribution_is_absent_with_the_reason_the_caller_gave) {
    std::vector<double> s;
    const Dist d = diagnose::summarize(s, "no decode step ran in this point");
    CHECK_EQ(d.count, 0);
    CHECK_EQ(d.absent, std::string("no decode step ran in this point"));
    CHECK(!d.mean.present);
    CHECK(!d.p50.present);
    CHECK(!d.max.present);
    /* And on disk it is the reason, not an empty object a reader guesses about. */
    diagnose::Report r = sample_report();
    r.points[0].step_us_decode = d;
    const json j = rendered(r);
    const json& dec = j["points"][0]["step_us"]["decode"];
    CHECK_EQ(dec.value("absent", std::string()), std::string("no decode step ran in this point"));
    CHECK(!dec.contains("p50"));
    CHECK(!dec.contains("mean"));
}

TEST(diagnose_the_p50_of_four_samples_reaches_the_json_unchanged) {
    const json j = rendered(sample_report());
    const json& dec = j["points"][0]["step_us"]["decode"];
    CHECK_EQ(dec.value("count", (int64_t)0), 4);
    CHECK_NEAR(dec["mean"].get<double>(), 2.5, 1e-12);
    CHECK_NEAR(dec["p50"].get<double>(), 2.5, 1e-12);
    CHECK_NEAR(dec["p90"].get<double>(), 3.7, 1e-12);
    CHECK_NEAR(dec["p99"].get<double>(), 3.97, 1e-12);
    CHECK_NEAR(dec["max"].get<double>(), 4.0, 1e-12);
}

/* ------------------------------------------------------------------ ratios */
TEST(diagnose_a_ratio_over_a_zero_denominator_is_absent_not_zero) {
    const Measure m = Measure::ratio(0, 0, "no draft token was verified");
    CHECK(!m.present);
    CHECK_EQ(m.absent, std::string("no draft token was verified"));
    /* 3 / 4 = 0.75, and a zero numerator over a real denominator IS a measurement: nothing of
     * that kind happened out of something that did. */
    CHECK_NEAR(Measure::ratio(3, 4, "unused").value, 0.75, 1e-12);
    CHECK(Measure::ratio(0, 8, "unused").present);
    CHECK_NEAR(Measure::ratio(0, 8, "unused").value, 0.0, 1e-12);
}

TEST(diagnose_acceptance_by_draft_position_is_conditional_on_reaching_it) {
    Acceptance a;
    a.drafted = 20;
    a.accepted = 10;
    a.reached = { 10, 5, 0 };
    a.hits = { 8, 2, 0 };
    /* Pooled: 10 / 20 = 0.5. Conditional: position 0 is 8 / 10 = 0.8, position 1 is 2 / 5 = 0.4. */
    CHECK_NEAR(a.pooled().value, 0.5, 1e-12);
    CHECK_NEAR(a.at(0).value, 0.8, 1e-12);
    CHECK_NEAR(a.at(1).value, 0.4, 1e-12);
    /* Nothing reached position 2, so there is no acceptance there to report -- a drafter that
     * never got that far is not a drafter that failed there. */
    CHECK(!a.at(2).present);
    CHECK(has(a.at(2).absent, "position 2"));
    CHECK(!a.at(9).present);
    CHECK_EQ(a.deepest(), 1);
}

TEST(diagnose_acceptance_with_no_drafts_is_absent_everywhere) {
    const Acceptance a;
    CHECK(a.empty());
    CHECK(!a.pooled().present);
    CHECK(!a.at(0).present);
    CHECK_EQ(a.deepest(), -1);
    /* And the report says so in words a reader of the JSON can act on. */
    diagnose::Report r = sample_report();
    r.points[0].acceptance = a;
    const json j = rendered(r);
    const json& acc = j["points"][0]["acceptance"];
    CHECK_EQ(acc.value("drafted", (int64_t)-1), 0);
    CHECK(acc["pooled"].contains("absent"));
    CHECK(acc["positions"].empty());
}

/* ------------------------------------------------------------------ categories */
TEST(diagnose_acceptance_pools_by_category_over_the_points_that_share_one) {
    diagnose::Report r = sample_report("category:prose", "prose");
    diagnose::PointReport second = sample_report().points[0];
    second.name = "category:code";
    second.category = "code";
    diagnose::PointReport third = sample_report().points[0];
    third.name = "category:structured";
    third.category = "prose";           /* a second prose point */
    r.points.push_back(second);
    r.points.push_back(third);

    Acceptance a;  a.drafted = 10; a.accepted = 5;  a.reached = { 10 };    a.hits = { 5 };
    Acceptance b;  b.drafted = 4;  b.accepted = 2;  b.reached = { 4 };     b.hits = { 2 };
    Acceptance c;  c.drafted = 10; c.accepted = 1;  c.reached = { 10, 5 }; c.hits = { 1, 1 };
    r.points[0].acceptance = a;
    r.points[1].acceptance = b;
    r.points[2].acceptance = c;

    const json j = rendered(r);
    REQUIRE_EQ((int)j["by_category"].size(), 2);
    /* prose: 10 + 10 drafted, 5 + 1 accepted -> 6 / 20 = 0.3, position 0: (5 + 1) / 20 = 0.3,
     * position 1: nothing reached it in the first point and 5 in the third -> 1 / 5 = 0.2. */
    const json& prose = j["by_category"][0];
    CHECK_EQ(prose.value("category", std::string()), std::string("prose"));
    const json& pa = prose["acceptance"];
    CHECK_EQ(pa.value("drafted", (int64_t)0), 20);
    CHECK_EQ(pa.value("accepted", (int64_t)0), 6);
    CHECK_NEAR(pa["pooled"]["value"].get<double>(), 0.3, 1e-12);
    CHECK_EQ(pa["positions"].size(), 2u);
    CHECK_EQ(pa["positions"][0]["reached"].get<int64_t>(), 20);
    CHECK_EQ(pa["positions"][0]["hits"].get<int64_t>(), 6);
    CHECK_NEAR(pa["positions"][0]["conditional"]["value"].get<double>(), 0.3, 1e-12);
    CHECK_NEAR(pa["positions"][1]["conditional"]["value"].get<double>(), 0.2, 1e-12);
    /* code stands alone: 2 / 4 = 0.5. */
    const json& code = j["by_category"][1];
    CHECK_EQ(code.value("category", std::string()), std::string("code"));
    CHECK_NEAR(code["acceptance"]["pooled"]["value"].get<double>(), 0.5, 1e-12);
}

/* ------------------------------------------------------------------ what the report holds */
TEST(diagnose_every_generated_token_is_in_the_report_and_counted_where_it_is_written) {
    diagnose::Report r = sample_report();
    diagnose::RequestRecord q2;
    q2.id = 2; q2.prompt = "code"; q2.category = "code"; q2.finish = "length";
    q2.tokens = { 9 };
    r.points[0].requests = 2;
    r.points[0].out_tokens = 5;
    r.points[0].reqs.push_back(std::move(q2));

    const json j = rendered(r);
    const json& reqs = j["points"][0]["requests"];
    REQUIRE_EQ((int)reqs.size(), 2);
    /* The ordinal is the report's own, in submission order: `--diagnose-baseline` keys parity on
     * it, so it has to start at 1 and go up. */
    CHECK_EQ(reqs[0].value("n", (int64_t)0), 1);
    CHECK_EQ(reqs[1].value("n", (int64_t)0), 2);
    CHECK_EQ(reqs[0]["tokens"].size(), 4u);
    CHECK_EQ(reqs[1]["tokens"][0].get<int64_t>(), 9);
    CHECK_EQ(j["points"][0]["tokens"]["out"].get<int64_t>(), 5);
    /* The markdown agrees with the JSON about how many there are. */
    CHECK(has(diagnose::report_markdown(r), "2 requests, 5 generated tokens"));
}

TEST(diagnose_a_point_that_did_not_measure_says_so_and_the_markdown_says_it_too) {
    diagnose::Report r = sample_report();
    r.points[0].measured = false;
    r.points[0].not_measured = "the prefix cache served 5 of this point's 5 prompt tokens";
    r.points[0].absent.push_back("decode step time: the point did not run");
    const json j = rendered(r);
    CHECK_EQ(j["points"][0].value("measured", true), false);
    CHECK(has(j["points"][0].value("not_measured", std::string()), "prefix cache"));
    CHECK_EQ(j["points"][0]["absent"].size(), 1u);
    const std::string md = diagnose::report_markdown(r);
    CHECK(has(md, "not a measurement"));
    CHECK(has(md, "decode step time: the point did not run"));
    /* A point that is not a measurement is a finding as well: the log says it, not only the file. */
    std::vector<diagnose::Finding> f;
    diagnose::review(diagnose::report_json(r), &f);
    bool found = false;
    for (const auto& x : f) found = found || x.id == "not-a-measurement";
    CHECK(found);
}

TEST(diagnose_the_markdown_carries_the_same_numbers_as_the_json) {
    const diagnose::Report r = sample_report();
    const std::string md = diagnose::report_markdown(r);
    /* The point, the run's identity, and the p50 of the series, rendered to one decimal: 2.5. */
    CHECK(has(md, "# radiance --diagnose"));
    CHECK(has(md, "category:prose"));
    CHECK(has(md, "p50 2.5"));
    CHECK(has(md, "p90 3.7"));
    /* The salt and the workload number are in both, because a run without them stated is not
     * comparable and the report has to say which one ran. */
    CHECK(has(md, "17"));
    CHECK_EQ(rendered(r)["salt"].get<int64_t>(), 17);
}

/* ------------------------------------------------------------------ against an earlier report */
TEST(diagnose_a_baseline_with_the_same_tokens_passes_and_says_what_it_checked) {
    const std::string a = diagnose::report_json(sample_report());
    std::vector<diagnose::Finding> f;
    diagnose::compare_tokens(a, a, &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK_EQ((int)f[0].level, (int)diagnose::Finding::Note);
    CHECK_EQ(f[0].id, std::string("token-parity"));
    CHECK(has(f[0].what, "1 request(s)"));
}

TEST(diagnose_the_first_token_mismatch_names_its_point_its_request_and_its_position) {
    diagnose::Report base = sample_report();
    diagnose::Report now = sample_report();
    /* The third token of the one request of category:prose moves: 5 6 7 8 -> 5 6 9 8, so the two
     * agree for 2 tokens and part at position 2. */
    now.points[0].reqs[0].tokens[2] = 9;
    std::vector<diagnose::Finding> f;
    diagnose::compare_tokens(diagnose::report_json(base), diagnose::report_json(now), &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK_EQ((int)f[0].level, (int)diagnose::Finding::Fail);
    CHECK(has(f[0].what, "category:prose"));
    CHECK(has(f[0].what, "request 1"));
    CHECK(has(f[0].what, "the first 2 token(s) match"));
    CHECK(has(f[0].what, "7"));        /* what the baseline had there */
    CHECK(has(f[0].what, "9"));        /* what this build wrote */
}

TEST(diagnose_a_baseline_whose_tokens_differ_does_not_stop_the_points_that_still_match) {
    diagnose::Report base = sample_report("category:prose", "prose");
    diagnose::PointReport other = sample_report("category:code", "code").points[0];
    base.points.push_back(other);
    diagnose::Report now = base;
    now.points[0].reqs[0].tokens[0] = 11;           /* only prose moves */

    std::vector<diagnose::Finding> f;
    diagnose::compare_tokens(diagnose::report_json(base), diagnose::report_json(now), &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK(has(f[0].what, "category:prose"));
    CHECK(!has(f[0].what, "category:code"));
    /* The report keeps the point that still passes, with its timings: a regression in one family of
     * text is read beside the family that did not move, not instead of it. */
    const json j = rendered(now);
    REQUIRE_EQ((int)j["points"].size(), 2);
    CHECK(j["points"][1]["step_us"]["decode"].contains("p50"));
    CHECK_EQ(j["points"][1]["tokens"]["out"].get<int64_t>(), 4);
}

TEST(diagnose_a_shorter_generation_is_a_mismatch_at_the_first_token_the_baseline_has) {
    diagnose::Report base = sample_report();
    diagnose::Report now = sample_report();
    now.points[0].reqs[0].tokens.pop_back();        /* 5 6 7 8 against 5 6 7 */
    std::vector<diagnose::Finding> f;
    diagnose::compare_tokens(diagnose::report_json(base), diagnose::report_json(now), &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK_EQ((int)f[0].level, (int)diagnose::Finding::Fail);
    /* Three tokens match; the baseline has an 8 there and this run has nothing at all. */
    CHECK(has(f[0].what, "the first 3 token(s) match"));
    CHECK(has(f[0].what, "nothing"));
    CHECK(has(f[0].what, "4 tokens against 3"));
}

TEST(diagnose_a_request_missing_from_the_baseline_is_counted_not_skipped) {
    diagnose::Report base = sample_report();
    diagnose::Report now = sample_report();
    diagnose::RequestRecord q = now.points[0].reqs[0];
    q.id = 2;
    now.points[0].reqs.push_back(q);
    std::vector<diagnose::Finding> f;
    diagnose::compare_tokens(diagnose::report_json(base), diagnose::report_json(now), &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK_EQ((int)f[0].level, (int)diagnose::Finding::Fail);
    CHECK(has(f[0].what, "1 are in this run and not the baseline"));
}

TEST(diagnose_an_acceptance_drop_past_the_threshold_is_a_finding_and_one_under_it_is_not) {
    diagnose::Report base = sample_report();
    diagnose::Report now = sample_report();
    Acceptance a;  a.drafted = 20; a.accepted = 10;  a.reached = { 10, 5 };  a.hits = { 5, 2 };
    Acceptance b;  b.drafted = 20; b.accepted = 8;   b.reached = { 10, 5 };  b.hits = { 2, 2 };
    base.points[0].acceptance = a;
    now.points[0].acceptance = b;
    const std::string bjson = diagnose::report_json(base), njson = diagnose::report_json(now);

    /* Pooled 10/20 = 0.5 against 8/20 = 0.4: a fall to 80% of what it was. At a 10% threshold that
     * is a finding; position 0 is 0.5 against 0.2, a fall to 40%; position 1 is 0.4 both times, so
     * nothing says a word about it. Category prose pools the one point, so it moves with it. */
    std::vector<diagnose::Finding> f;
    diagnose::compare_acceptance(bjson, njson, 0.10, &f);
    int pooled = 0, position = 0, category = 0;
    for (const auto& x : f) {
        CHECK_EQ(x.id, std::string("acceptance-drop"));
        if (has(x.what, "position 0")) ++position;
        else if (has(x.what, "category prose")) ++category;
        else ++pooled;
        /* Position 1 held, and a finding about it would be wrong either way. */
        CHECK(!has(x.what, "position 1"));
    }
    CHECK_EQ(pooled, 1);
    CHECK_EQ(position, 1);
    CHECK_EQ(category, 1);

    /* The threshold itself, with the positions held: 0.5 against 0.4 is a fifth of the way gone,
     * which is a finding at 10% and the workload breathing at 25%. */
    diagnose::Report hold = sample_report();
    diagnose::Report move = sample_report();
    Acceptance deep;  deep.drafted = 10; deep.accepted = 5;  deep.reached = { 10 }; deep.hits = { 5 };
    Acceptance less;  less.drafted = 10; less.accepted = 4;  less.reached = { 10 }; less.hits = { 4 };
    hold.points[0].acceptance = deep;
    move.points[0].acceptance = less;
    const std::string hj = diagnose::report_json(hold), mj = diagnose::report_json(move);
    std::vector<diagnose::Finding> tight;
    diagnose::compare_acceptance(hj, mj, 0.10, &tight);
    int warns = 0;
    for (const auto& x : tight) warns += x.level == diagnose::Finding::Warn ? 1 : 0;
    CHECK_EQ(warns, 3);                    /* the point, its category, and the one position */

    std::vector<diagnose::Finding> loose;
    diagnose::compare_acceptance(hj, mj, 0.25, &loose);
    warns = 0;
    for (const auto& x : loose) warns += x.level == diagnose::Finding::Warn ? 1 : 0;
    CHECK_EQ(warns, 0);
    /* And the report says it checked: an absence of findings is not the same as no comparison. */
    bool noted = false;
    for (const auto& x : loose) noted = noted || x.level == diagnose::Finding::Note;
    CHECK(noted);
}

TEST(diagnose_nothing_drafted_in_either_run_is_not_a_drop) {
    const std::string a = diagnose::report_json(sample_report());
    std::vector<diagnose::Finding> f;
    diagnose::compare_acceptance(a, a, 0.10, &f);
    REQUIRE_EQ((int)f.size(), 1);
    CHECK_EQ((int)f[0].level, (int)diagnose::Finding::Note);
    CHECK(has(f[0].what, "no point's pooled acceptance"));
}

TEST(diagnose_a_baseline_of_another_workload_is_refused_by_name) {
    diagnose::Report base = sample_report();
    diagnose::Report now = sample_report();
    now.workload = 4321;                       /* the points changed */
    std::string why;
    CHECK(!diagnose::same_workload(diagnose::report_json(base), diagnose::report_json(now), &why));
    CHECK(has(why, "1234"));
    CHECK(has(why, "4321"));
    /* Same points, different salt: the prompts differ by construction, so the tokens are not
     * comparable and comparing them would produce a finding per request instead of this one. */
    diagnose::Report salted = sample_report();
    salted.salt = 18;
    CHECK(!diagnose::same_workload(diagnose::report_json(base), diagnose::report_json(salted), &why));
    CHECK(has(why, "salt"));
    why.clear();
    CHECK(diagnose::same_workload(diagnose::report_json(base), diagnose::report_json(base), &why));
    CHECK(why.empty());
}

TEST(diagnose_a_baseline_that_is_not_one_of_these_reports_is_refused_before_it_is_read) {
    std::string err;
    CHECK(!diagnose::parse_ok("not json at all", &err));
    CHECK(!diagnose::parse_ok("[1,2,3]", &err));
    CHECK(!diagnose::parse_ok(R"({"schema":3})", &err));          /* no points */
    CHECK(!diagnose::parse_ok(R"({"schema":2,"points":[]})", &err));
    CHECK(has(err, "2"));                                         /* says which schema it wanted */
    CHECK(diagnose::parse_ok(diagnose::report_json(sample_report()), &err));
}

/* ------------------------------------------------------------------ the report's own arithmetic */
TEST(diagnose_review_refuses_a_series_that_is_not_ordered_the_way_a_distribution_is) {
    /* Hand-written rather than rendered, because the point of the check is a report the renderer
     * should never have produced: p50 above p90 cannot happen, and if it does, the file is wrong. */
    const char* bad = R"({
      "schema": 3, "workload": 7, "salt": 1,
      "points": [{ "name": "a", "category": "prose", "measured": true,
                   "step_us": { "decode": { "count": 2, "mean": 5, "p50": 9, "p90": 4,
                                            "p99": 10, "max": 12 } },
                   "requests": [{ "n": 1, "tokens": [1, 2] }] }]
    })";
    std::vector<diagnose::Finding> f;
    diagnose::review(bad, &f);
    bool arithmetic = false;
    for (const auto& x : f) arithmetic = arithmetic || x.id == "report-arithmetic";
    CHECK(arithmetic);
    /* The level is a failure: a number that cannot be true is not a warning to read past. */
    bool failed = false;
    for (const auto& x : f) failed = failed || x.level == diagnose::Finding::Fail;
    CHECK(failed);
}

TEST(diagnose_review_reports_a_cache_hit_dropped_step_records_and_an_empty_workload) {
    const char* hit = R"({
      "schema": 3, "workload": 7, "salt": 1,
      "points": [{ "name": "a", "category": "prose", "measured": true,
                   "step_records_dropped": 3,
                   "tokens": { "cached": 9 },
                   "requests": [{ "n": 1, "tokens": [1, 2] }] }]
    })";
    std::vector<diagnose::Finding> f;
    diagnose::review(hit, &f);
    bool cache = false, steplog = false;
    for (const auto& x : f) {
        cache = cache || x.id == "cache";
        steplog = steplog || x.id == "step-log";
    }
    CHECK(cache);
    CHECK(steplog);

    /* Nothing generated: there is no run there, whatever the timings say. */
    std::vector<diagnose::Finding> g;
    diagnose::review(R"({"schema":3,"workload":7,"salt":1,"points":[{"name":"a","category":"prose",
                     "measured":true,"requests":[]}]})", &g);
    bool empty_workload = false, no_speculation = false;
    for (const auto& x : g) {
        empty_workload = empty_workload || (x.id == "workload" && x.level == diagnose::Finding::Fail);
        no_speculation = no_speculation || (x.id == "acceptance" && x.level == diagnose::Finding::Note);
    }
    CHECK(empty_workload);
    /* And a run with no drafter says that its acceptance is absent rather than zero, so a reader
     * does not take the absence for a drafter that stopped working. */
    CHECK(no_speculation);
}

/* ------------------------------------------------------------------ the workload itself */
TEST(diagnose_the_workload_table_holds_together) {
    const auto& P = diagnose::prompts();
    const auto& X = diagnose::points();
    CHECK(!P.empty());
    CHECK(!X.empty());
    std::vector<std::string> names;
    for (const auto& p : X) {
        CHECK(p.name && *p.name);
        CHECK(p.category && *p.category);
        CHECK(std::find(names.begin(), names.end(), std::string(p.name)) == names.end());
        names.push_back(p.name);
        CHECK(p.concurrency >= 1);
        CHECK(p.n_prompts >= 1);
        CHECK(p.repeat >= 1);
        CHECK(p.max_tokens >= 1);
        CHECK(p.first_prompt >= 0);
        CHECK(p.first_prompt < (int)P.size());
        CHECK(p.interfere_prompt >= 0);
        CHECK(p.interfere_prompt < (int)P.size());
        /* An interference point holds one prompt back, so it needs more prompts than it keeps in
         * flight -- otherwise nothing arrives while anything is decoding, and the point measures
         * the arrival of the first request instead. */
        if (p.interfere_after_tokens > 0) {
            CHECK(p.repeat * p.n_prompts > p.concurrency);
        }
    }
    /* Every family the report claims to cover is in the table. */
    bool decode = false, prefill = false, interference = false, concurrency = false;
    for (const auto& p : X) {
        const std::string c = p.category;
        decode = decode || c == "decode";
        prefill = prefill || c == "prefill";
        interference = interference || c == "interference";
        concurrency = concurrency || c == "concurrency";
    }
    CHECK(decode);
    CHECK(prefill);
    CHECK(interference);
    CHECK(concurrency);
    /* A concurrency point above one sequence, and a prefill point longer than the shortest. */
    int widest = 1, longest = 0;
    for (const auto& p : X) widest = std::max(widest, p.concurrency);
    for (const auto& p : P) longest = std::max(longest, p.tokens);
    CHECK(widest >= 8);
    CHECK(longest >= 1024);
}

TEST(diagnose_the_workload_id_belongs_to_the_workload_and_is_stable_within_a_build) {
    const uint64_t a = diagnose::workload_id();
    const uint64_t b = diagnose::workload_id();
    CHECK_EQ(a, b);
    CHECK(a != 0);
}

/* ------------------------------------------------------------------ the step record */
TEST(steplog_keeps_the_records_in_order_and_counts_what_it_overwrote) {
    rad::StepLog log;
    log.enable(4);
    CHECK(log.enabled());
    CHECK_EQ(log.capacity(), 4u);
    for (int i = 0; i < 6; ++i) {
        rad::StepRecord r;
        r.step = i;
        r.tok_out = 1;
        log.push(r);
    }
    std::vector<rad::StepRecord> out;
    const uint64_t dropped = log.drain(&out);
    /* Six pushed into four slots: the first two went, and the report has to know that. */
    REQUIRE_EQ(out.size(), 4u);
    CHECK_EQ(dropped, 2u);
    /* Oldest first, so a reader's gaps are in the order they happened. */
    CHECK_EQ(out[0].step, 2);
    CHECK_EQ(out[1].step, 3);
    CHECK_EQ(out[2].step, 4);
    CHECK_EQ(out[3].step, 5);
    /* Drained: empty, and the drop count starts again. */
    CHECK_EQ(log.size(), 0u);
    std::vector<rad::StepRecord> again;
    CHECK_EQ(log.drain(&again), 0u);
    CHECK(again.empty());
}

TEST(steplog_that_was_never_enabled_holds_nothing_and_allocates_nothing) {
    rad::StepLog log;
    CHECK(!log.enabled());
    rad::StepRecord r;
    log.push(r);                              /* the step path, with the mode off */
    std::vector<rad::StepRecord> out;
    CHECK_EQ(log.drain(&out), 0u);
    CHECK(out.empty());
    log.enable(8);
    CHECK_EQ(log.capacity(), 8u);
    log.disable();
    CHECK(!log.enabled());
    log.push(r);
    CHECK_EQ(log.drain(&out), 0u);
}

RAD_TEST_MAIN()
