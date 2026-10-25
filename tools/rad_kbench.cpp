/* rad-kbench -- check every resolved kernel against libref, per op, on the real container or a
 * recorded fixture (spec §17), and optionally time it.
 *
 * Design rules:
 *
 *   The artifact, not a synthetic: weight operands come from the .rad, in the layout the resolved
 *   kernel asked for. Quantising in-process would share the quantiser, layout and geometry with
 *   the code under test.
 *
 *   Geometry comes from the container's metadata through the engine's declare phase.
 *
 *   A kernel that legitimately does not serve a case is skipped, with the reason, not failed.
 *
 *   Tolerances are per (op, dtype): no single threshold suits 5-bit codebook experts and bf16 dense
 *   alike. Ids are compared exactly.
 *
 *   Non-finite values are compared first: `rel > tol` is false for NaN, so the finiteness pattern
 *   of both sides is checked before any norm.
 *
 *   Per-case seeds, hashed from (op, band, domain, operand), so a case is reproducible regardless
 *   of which other cases or flags ran.
 *
 * The rules themselves (comparison, tolerances, keys, budget, exit code, the reference cache) are
 * in kbench_core.h; this file wires them to plugins, devices and fixtures.
 */
#include "kbench_core.h"
#include "kbench_timer.h"
#include "kfixture.h"
#include "opshapes.h"
#include "format/encoding.h"

#include "rad_device.h"
#include "sample/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <set>
#include <unistd.h>

using namespace rad;
using kb::Err;
using kb::Stats;
using kb::Phases;
using kb::PhaseTimer;
using kb::takes_worst;

namespace {

/* ================================================================== the run */
struct Row {
    std::string op, kernel, plugin, geom, dtype, choice, verdict;
    /* The band this point came from, e.g. "M in (64, 512]". */
    std::string band;
    const char* domain = "";
    double      rel_l2 = -1;
    double      tol = 0;
    /* --bench (kbench_timer.h): microseconds a launch, < 0 when not timed, and how sure. */
    double      us = -1;
    double      us_min = -1, us_p90 = -1;
    double      us_ci = -1;     /* the median's 95% interval half-width, percent */
    int         us_n = 0;       /* timed batches */
    double      clock_mhz = -1;
    double      restore_pct = -1;
    int         threads = 0;    /* a host row's OpenMP threads */
    bool        gated = false;  /* a device row timed behind the gate */
    std::string tnote;          /* why a time is missing or worth less */
    int64_t     timing_id = -1; /* the device timer's id until its pass has run */
    int64_t     bytes = 0;      /* operand bytes touched once, for a GB/s that names its own basis */
    bool        synthesised = false;
};

/* A timing result into its report row. */
void apply_timing(Row& r, const kb::Timing& t) {
    r.us = t.st.n > 0 ? t.st.median : -1;
    r.us_min = t.st.min;
    r.us_p90 = t.st.p90;
    r.us_ci = t.st.ci_pct;
    r.us_n = t.st.n;
    r.clock_mhz = t.clock_mhz;
    r.restore_pct = t.restore_pct;
    r.threads = t.threads;
    r.gated = t.gated;
    r.tnote = t.note;
    if (t.retried) r.tnote += (r.tnote.empty() ? "" : "; ") +
                              fmt("%d batch(es) repeated for a throttler", t.retried);
}

/* What the report says about how the times were taken: the device timer's header, and why times
 * were withheld if they were. Null in write_report when nothing was timed. */
struct BenchInfo {
    std::vector<std::string> header;
    std::string withheld;
    int64_t     host_footprint = 0;
    bool        footprint_flag = false;   /* --bench-flush set it */
};

/* One library's summary. Completeness comes from its row tables in the registry, not from the run
 * (which only covers what this container declares), measured against every op any loaded plugin
 * declares a schema for (§2.3). Correctness and speed are folded in from the run. */
struct LibRow {
    std::string name, version, file, build_target;
    int  order = 0;
    bool oracle = false;
    /* op -> the domains this library has a row for it in; the report names the domain, since a
     * host-only row is a fallback, not the fast path. */
    std::map<std::string, std::set<int>> ops;
    std::set<int>         domains;
    int  n_rows = 0;
    int  n_tunable_rows = 0;            /* rows declaring at least one axis */
    int  n_fusion_rows  = 0;            /* rows declaring `replaces` */
    int  n_layout_rows  = 0;            /* rows that re-lay a weight out */
    /* Every row by name, so the report can list rows no case reached (a band no fixture lands
     * in, an unused dtype, a variant behind an unmet constraint); counts alone cannot show
     * them. */
    std::map<std::string, std::string> rows;   /* kernel name -> the op it serves */
    std::set<std::string> ran;                 /* kernel names a case actually reached */
    /* Filled from the run. */
    int    cases = 0, passed = 0, failed = 0;
    double worst = -1;
    std::vector<double> us;
};

/* An operand description such as `i8[56,5120]`, for reporting what two libraries disagree on. */
std::string shape_str(const OpdShape& d) {
    std::string s = rad_dtype_name(d.dtype);
    s += "[";
    for (size_t i = 0; i < d.shape.size(); ++i) {
        if (i) s += ",";
        s += fmt("%lld", (long long)d.shape[i]);
    }
    return s + "]";
}

std::string md_escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '|') o += "\\|"; else o += c; }
    return o;
}

const char* domain_short(int d) {
    switch (d) {
        case RAD_DOMAIN_HOST:   return "host";
        case RAD_DOMAIN_DEVICE: return "device";
        default:                return "?";
    }
}

std::string domains_str(const std::set<int>& d) {
    std::string s;
    for (int x : d) { if (!s.empty()) s += "+"; s += domain_short(x); }
    return s.empty() ? "-" : s;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* One stream for the whole run. The backend never reuses a queue id (event waiters name queues by
 * id), so a stream per case would exhaust the queues after about a thousand cases; per-stream
 * kernel state (a finisher's arrival counter) is keyed by it too. A case drains the stream before
 * freeing anything it could still touch. */
RadStream run_stream() {
    static RadStream s = nullptr;
    if (!s && rad_stream_create(&s, 0) < 0) s = nullptr;
    return s;
}

/* Runs the timing pass over what the timer holds, puts each result into its row, prints how the
 * times were taken, and returns that for the report. */
BenchInfo finish_timing(kb::DeviceTimer* timer, std::vector<Row>& rows, int64_t host_footprint,
                        bool footprint_flag) {
    BenchInfo b;
    b.host_footprint = host_footprint;
    b.footprint_flag = footprint_flag;
    if (timer) {
        timer->flush();
        for (Row& r : rows) {
            if (r.timing_id < 0) continue;
            if (const kb::Timing* t = timer->result(r.timing_id)) apply_timing(r, *t);
            r.timing_id = -1;
        }
        b.header = timer->header();
        b.withheld = timer->withheld();
    }
    int timed = 0, untimed = 0, device = 0, gated = 0;
    for (const Row& r : rows) {
        if (r.us >= 0) ++timed;
        else if (!r.tnote.empty()) ++untimed;
        if (r.us >= 0 && std::string(r.domain) == domain_name(RAD_DOMAIN_DEVICE)) {
            ++device;
            gated += r.gated;
        }
    }
    std::printf("\ntiming: %d case(s) timed", timed);
    if (device) std::printf(", %d of the %d on the device behind a gate", gated, device);
    if (untimed) std::printf(", %d not (the report's speed notes say why)", untimed);
    std::printf("\n");
    for (const std::string& h : b.header) std::printf("  %s\n", h.c_str());
    return b;
}

/* The census of each kernel library, read from the registry rather than the run (see LibRow), so
 * it also covers a library that serves nothing in this container. */
std::vector<LibRow> census(const Registry& reg, const std::string& ref_name) {
    std::map<std::string, LibRow> by_name;
    for (const Plugin& p : reg.plugins()) {
        if (p.kind != RAD_PLUGIN_KERNEL) continue;
        LibRow l;
        l.name = p.name; l.version = p.version; l.file = p.file;
        l.build_target = p.build_target;
        l.order = p.order;
        l.oracle = p.name == ref_name;
        by_name[p.name] = std::move(l);
    }
    for (const KernelRow& r : reg.rows()) {
        auto it = by_name.find(r.plugin);
        if (it == by_name.end() || !r.info || !r.info->op) continue;
        LibRow& l = it->second;
        ++l.n_rows;
        l.ops[r.info->op].insert(r.info->domain);
        if (r.info->name) l.rows[r.info->name] = r.info->op;
        l.domains.insert(r.info->domain);
        if (r.info->n_tunables > 0) ++l.n_tunable_rows;
        if (r.info->replaces)       ++l.n_fusion_rows;
        if (r.info->layout)         ++l.n_layout_rows;
    }
    /* Hierarchy (selection) order, with the oracle last. */
    std::vector<LibRow> out;
    for (auto& [n, l] : by_name) out.push_back(std::move(l));
    std::sort(out.begin(), out.end(), [](const LibRow& a, const LibRow& b) {
        if (a.oracle != b.oracle) return b.oracle;
        if (a.order != b.order)   return a.order < b.order;
        return a.name < b.name;
    });
    return out;
}

/* Fold the run's rows into the library summaries; shared by the report and the coverage line. */
void fold_run(std::vector<LibRow>& libs, const std::vector<Row>& rows) {
    std::map<std::string, LibRow*> by_name;
    for (LibRow& l : libs) by_name[l.name] = &l;
    for (const Row& r : rows) {
        auto it = by_name.find(r.plugin);
        if (it == by_name.end()) continue;
        LibRow& l = *it->second;
        ++l.cases;
        if (!r.kernel.empty()) l.ran.insert(r.kernel);
        if (r.verdict == "ok") ++l.passed;
        else if (r.verdict != "timed" && r.verdict != "vacuous") ++l.failed;
        l.worst = kb::worst_of(l.worst, r.rel_l2);
        if (r.us >= 0) l.us.push_back(r.us);
    }
}

/* The row-coverage line printed after every pass: declared rows that ran, and per library how
 * many did not (`--report` names them). */
void print_row_coverage(std::vector<LibRow> libs, const std::vector<Row>& rows) {
    fold_run(libs, rows);
    size_t declared = 0, ran = 0;
    for (const LibRow& l : libs) { declared += l.rows.size(); ran += l.ran.size(); }
    if (declared == 0) return;
    std::printf("%zu of %zu declared kernel row(s) ran", ran, declared);
    std::string cold;
    for (const LibRow& l : libs) {
        const size_t missed = l.rows.size() - l.ran.size();
        if (!missed) continue;
        cold += (cold.empty() ? "" : ", ") + l.name + " " + std::to_string(missed);
    }
    if (cold.empty()) std::printf(" -- every row every loaded library declares\n");
    else std::printf("; not reached: %s (--report names them)\n", cold.c_str());
}

/* Named lines a pass collects, by what they mean. */
struct Named {
    std::map<std::string, int> skipped;    /* declined cases: a gap in coverage, not a failure */
    std::map<std::string, int> stale;      /* the fixture and the libraries no longer agree */
    std::map<std::string, int> unserved;   /* no library under test has a row */
    std::map<std::string, int> vacuous;    /* both sides identically zero */
    std::map<std::string, int> overruns;
    std::map<std::string, int> diverged;
    std::map<std::string, int> written;    /* a host kernel changed an input */
    std::map<std::string, int> fuse_skipped;
};

/* The end-of-pass summary both passes print: every skip, overrun and divergence by name, then
 * the totals. */
void print_summary(const Stats& st, bool timing_only, const Named& nm) {
    auto list = [](const char* head, const std::map<std::string, int>& m) {
        if (m.empty()) return;
        std::printf("\n%s\n", head);
        for (const auto& [what, n] : m) std::printf("  %-70s x%d\n", what.c_str(), n);
    };
    list("skipped", nm.skipped);
    list("no library under test serves these (named, not a failure)", nm.unserved);
    list("stale: the fixture and the loaded libraries disagree -- record it again "
         "(a failure under --strict)", nm.stale);
    list("vacuous: the reference and the output are both identically zero, so nothing was "
         "checked", nm.vacuous);
    list("out of bounds: a kernel wrote outside what it was given", nm.overruns);
    list("a kernel wrote into an operand it was handed to read", nm.written);
    list("described differently by the two libraries (not a correctness result)", nm.diverged);
    std::printf("\n%d checked, %d passed, %d failed, %d skipped, %d diverged, %d capped",
                st.checked, st.passed, st.failed, st.skipped, st.diverged, st.capped);
    if (st.vacuous) std::printf(", %d vacuous", st.vacuous);
    if (st.stale) std::printf(", %d stale", st.stale);
    if (st.unserved) std::printf(", %d unserved", st.unserved);
    std::printf(". worst rel_l2 = %.3e\n", st.worst);
    if (st.cached)
        std::printf("%d reference result(s) came from the cache (the comparison ran; --no-cache "
                    "runs the reference)\n", st.cached);
    if (timing_only)
        std::printf("%d timed. Timing only: nothing was checked against a reference.\n",
                    st.timed);
    std::printf("memory: %d case(s) ran inside guarded allocations, %d wrote outside one\n",
                st.guarded, st.overrun);
}

/* ================================================================== the report
 *
 * One Markdown document covering every library: correctness, completeness and speed from one run,
 * since they are not separable (a library fast on the ops it has but falling back to the reference
 * on the rest is slower end to end). Correctness and timing share one setup, so they measure the
 * same case. How the times were taken (kbench_timer.h) is printed above the speed table, with the
 * calibration it was checked against.
 */
int write_report(const std::string& path, const std::vector<Row>& rows,
                 const std::map<std::string, int>& skipped,
                 const std::map<std::string, int>& fuse_skipped,
                 const std::map<std::string, int>& diverged,
                 const std::map<std::string, int>& overruns,
                 const std::string& analysis_path,
                 /* One line saying where the numbers came from: the container, or the fixture and
                  * its recording date. */
                 const std::string& provenance,
                 int fuse_passed, int fuse_failed,
                 const Stats& st, const std::string& ref_name,
                 const std::string& libraries, const std::string& arch_plugin,
                 const std::vector<std::string>& graph_ops,
                 std::vector<LibRow> libs, const BenchInfo* bench) {
    const bool benched = bench != nullptr;
    /* ---------------------------------------------------------------- fold the run into libs */
    fold_run(libs, rows);

    /* The vocabulary completeness is measured against: every op any loaded plugin declares. */
    std::set<std::string> vocab;
    for (const LibRow& l : libs) for (const auto& [o, _d] : l.ops) vocab.insert(o);
    std::set<std::string> needed(graph_ops.begin(), graph_ops.end());

    std::string s;
    s += "# radiance kernel libraries\n\n";
    s += "Generated by `rad-kbench`. Every kernel of every loaded library ran on the same "
         "inputs" + (benched ? std::string(" and each\ncase that passed was timed in the same "
                                           "run, so correctness, completeness and speed come from "
                                           "one run\n(the Speed section says how the times were "
                                           "taken).\n\n")
                             : std::string(";\nnothing was timed (`--bench` was not given).\n\n"));
    s += "- " + provenance + "\n";
    if (!arch_plugin.empty()) s += "- architecture plugin: `" + arch_plugin + "`\n";
    s += "- libraries: `" + libraries + "`\n";
    s += "- reference: `" + ref_name + "`; every correctness number is relative to it, so it "
         "has none of its own\n";
    s += fmt("- %d case(s) checked, %d passed, %d failed, %d skipped, %d capped; worst rel_l2 "
             "%.3e\n", st.checked, st.passed, st.failed, st.skipped, st.capped, st.worst);
    if (st.timed)
        s += fmt("- %d case(s) timed and not checked (`--timing-only`): this report ranks speed "
                 "only\n", st.timed);
    if (st.vacuous)
        s += fmt("- %d case(s) vacuous: the reference and the output both identically zero, so "
                 "nothing was\n  checked (listed under Not covered)\n", st.vacuous);
    if (st.stale)
        s += fmt("- %d case(s) stale: the fixture and the loaded libraries no longer agree "
                 "(listed under Not covered)\n", st.stale);
    if (st.cached)
        s += fmt("- %d reference result(s) taken from the cache; every comparison ran\n",
                 st.cached);
    if (st.diverged)
        s += fmt("- %d case(s) where two libraries describe the same output differently (not a\n"
                 "  correctness result; see Divergences below)\n", st.diverged);
    /* Only when the fusion check ran (container passes only; a replay does not run it). */
    if (fuse_passed || fuse_failed)
        s += fmt("- %d fusion(s) byte-identical to the chain they declare, %d differ\n",
                 fuse_passed, fuse_failed);
    s += "\n";

    /* ================================================================ 1. the summary */
    s += "## Libraries\n\n";
    s += "`ops` is how many of the " + fmt("%zu", vocab.size()) +
         "-op vocabulary this library has a row for, and `graph` how many of\nthe " +
         fmt("%zu", needed.size()) + " ops this container's declared graph asks for.\n\n";
    s += "`rows` is how many kernel rows the library declares and `run` how many of them a case "
         "in this\npass reached; the Rows nothing ran section lists the rest.\n\n";
    s += "| library | version | domain | rows | run | ops | graph | cases | pass | fail | "
         "worst rel_l2 |";
    if (benched) s += " median us |";
    s += "\n|---|---|---|---|---|---|---|---|---|---|---|";
    if (benched) s += "---|";
    s += "\n";
    for (const LibRow& l : libs) {
        int in_graph = 0;
        for (const std::string& o : needed) if (l.ops.count(o)) ++in_graph;
        s += "| `" + md_escape(l.name) + "`" + (l.oracle ? " (oracle)" : "") + " | " +
             md_escape(l.version) + " | " + domains_str(l.domains) + " | " +
             fmt("%d", l.n_rows) + " | " +
             fmt("%zu of %zu", l.ran.size(), l.rows.size()) + " | " +
             fmt("%zu of %zu", l.ops.size(), vocab.size()) + " | " +
             fmt("%d of %zu", in_graph, needed.size()) + " | " +
             fmt("%d", l.cases) + " | " + fmt("%d", l.passed) + " | " +
             (l.failed ? fmt("**%d**", l.failed) : std::string("0")) + " | " +
             /* The oracle has a number in a replay, none in a container pass. */
             (l.worst >= 0 ? fmt("%.2e", l.worst)
                           : (l.oracle ? std::string("- (it is the oracle)")
                                       : std::string("-"))) + " |";
        if (benched) s += (l.us.empty() ? std::string(" - |") : fmt(" %.2f |",
                                                                   median_of(l.us)));
        s += "\n";
    }
    s += "\nRows declaring an optional hook (tuning, fusion, weight layout):\n\n";
    s += "| library | tunable rows | fusion claims | weight layouts |\n|---|---|---|---|\n";
    for (const LibRow& l : libs)
        s += "| `" + md_escape(l.name) + "` | " + fmt("%d", l.n_tunable_rows) + " | " +
             fmt("%d", l.n_fusion_rows) + " | " + fmt("%d", l.n_layout_rows) + " |\n";

    /* ================================================================ 1b. rows nothing ran
     *
     * The completeness table is per op, and one op has several rows (bands, dtypes, tail
     * variants). Unreached rows are listed by name; covering them needs a fixture recorded from a
     * model that reaches them.
     */
    s += "\n## Rows nothing ran\n\n";
    {
        size_t declared = 0, ran = 0;
        for (const LibRow& l : libs) { declared += l.rows.size(); ran += l.ran.size(); }
        s += fmt("%zu of %zu declared row(s) were reached by a case in this pass.\n\n", ran,
                 declared);
        bool any = false;
        for (const LibRow& l : libs) {
            std::vector<std::pair<std::string, std::string>> missed;
            for (const auto& [kernel, op] : l.rows)
                if (!l.ran.count(kernel)) missed.push_back({ kernel, op });
            if (missed.empty()) continue;
            any = true;
            s += "`" + md_escape(l.name) + "`: " + fmt("%zu", missed.size()) +
                 " row(s) not reached:\n\n| kernel | op |\n|---|---|\n";
            for (const auto& [kernel, op] : missed)
                s += "| `" + md_escape(kernel) + "` | `" + md_escape(op) + "` |\n";
            s += "\n";
        }
        if (!any) s += "Every row every loaded library declares ran at least once.\n";
    }
    /* ================================================================ 2. completeness */
    s += "\n## Completeness\n\n";
    s += "One row per op in the vocabulary. A cell names the domain the library serves that op in, "
         "or `-`\nwhere it has no row. `graph` marks the ops this container's declared program "
         "asks for; a `-`\non a marked row is a hole this model falls into.\n\n";
    s += "| op | graph |";
    for (const LibRow& l : libs) s += " " + md_escape(l.name) + " |";
    s += "\n|---|---|";
    for (size_t i = 0; i < libs.size(); ++i) s += "---|";
    s += "\n";
    for (const std::string& o : vocab) {
        s += "| `" + md_escape(o) + "` | " + (needed.count(o) ? "yes" : " ") + " |";
        for (const LibRow& l : libs) {
            const auto it = l.ops.find(o);
            if (it == l.ops.end()) { s += " - |"; continue; }
            /* The domains of this library's rows for this op. */
            s += " " + domains_str(it->second) + " |";
        }
        s += "\n";
    }

    /* ================================================================ 3. correctness */
    /* One row per distinct case (op, kernel, plugin, domain, geometry, tuned point, operand
     * bytes), not per layer, carrying the run count, the worst error (one bad layer is the
     * finding) and the median time. The operand bytes keep apart cases whose parameters agree
     * and whose operands do not: attention's depth sweep lives in the cache's extent, and a
     * 16384-deep case takes a hundred times a 64-deep one. */
    struct Agg {
        Row     first;
        int     n = 0;
        double  worst_rel = -1;
        bool    any_fail = false;
        bool    any_timed = false;    /* --timing-only: ran and was timed, checked by nothing */
        bool    any_vacuous = false;  /* both sides identically zero */
        bool    any_synth = false;
        /* The timed runs of the group: medians, minima, p90s and clocks, and the widest
         * interval and the notes, as for the error. */
        std::vector<double> us, us_min, us_p90, clock, restore;
        int     us_n = 0, threads = 0;
        double  worst_ci = -1;
        std::string tnote;
    };
    std::map<std::string, Agg> agg;
    std::vector<std::string> order;
    for (const Row& r : rows) {
        const std::string key = r.op + "\x1f" + r.kernel + "\x1f" + r.plugin + "\x1f" + r.domain +
                                "\x1f" + r.geom + "\x1f" + r.choice + "\x1f" +
                                std::to_string(r.bytes);
        auto it = agg.find(key);
        if (it == agg.end()) { it = agg.emplace(key, Agg{}).first; it->second.first = r;
                               order.push_back(key); }
        Agg& a = it->second;
        ++a.n;
        a.worst_rel = kb::worst_of(a.worst_rel, r.rel_l2);
        a.any_fail  = a.any_fail  || (r.verdict != "ok" && r.verdict != "timed" &&
                                      r.verdict != "vacuous");
        a.any_timed = a.any_timed || r.verdict == "timed";
        a.any_vacuous = a.any_vacuous || r.verdict == "vacuous";
        a.any_synth = a.any_synth || r.synthesised;
        if (r.us >= 0) {
            a.us.push_back(r.us);
            a.us_min.push_back(r.us_min);
            a.us_p90.push_back(r.us_p90);
            a.us_n += r.us_n;
            a.worst_ci = std::max(a.worst_ci, r.us_ci);
            if (r.clock_mhz > 0) a.clock.push_back(r.clock_mhz);
            if (r.restore_pct >= 0) a.restore.push_back(r.restore_pct);
            a.threads = std::max(a.threads, r.threads);
        }
        if (!r.tnote.empty() && a.tnote.find(r.tnote) == std::string::npos)
            a.tnote += (a.tnote.empty() ? "" : "; ") + r.tnote;
    }

    s += "\n## Correctness\n\n";
    s += fmt("%zu distinct case(s) over %zu check(s). `n` is how many times the case ran "
             "(typically one per\nlayer) and `rel_l2` is the worst of them. `(synth w)` means "
             "this op's weights are not in the\ncontainer and were drawn instead, so the case "
             "checks the arithmetic, not the artifact.\n\n", order.size(), rows.size());
    s += "| op | kernel | library | domain | geometry | tuned | n | verdict | rel_l2 | tol |\n";
    s += "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const std::string& key : order) {
        Agg& a = agg[key];
        const Row& r = a.first;
        s += "| `" + md_escape(r.op) + "` | `" + md_escape(r.kernel) + "` | " +
             md_escape(r.plugin) + " | " + r.domain + " | `" + md_escape(r.geom) + "` | " +
             md_escape(r.choice.empty() ? "-" : r.choice) + " | " + fmt("%d", a.n) + " | " +
             (a.any_fail ? "**FAIL**" : a.any_timed ? "timed" : a.any_vacuous ? "vacuous" : "ok") +
             (a.any_synth ? " (synth w)" : "") + " | " +
             (a.worst_rel < 0 ? "-" : fmt("%.2e", a.worst_rel)) + " | " +
             fmt("%.0e", r.tol) + " |\n";
    }

    /* ================================================================ 4. speed */
    if (benched) {
        s += "\n## Speed\n\n";
        s += "How the times were taken (`--bench`, tools/kbench_timer.h):\n\n";
        s += "- **Device rows** are device timestamps around a batch of back-to-back launches, "
             "divided by the\n  launch count. The batch is queued behind a gate the host opens "
             "once all of it is queued,\n  so neither the host's submission rate nor its "
             "synchronisation is in the interval, and a\n  keep-alive copy runs ahead of the "
             "gate so the card does not idle (and drop its clock) while\n  the host submits. "
             "Correctness ran first; the passing cases were then timed back to back after\n  "
             "one sustained warm-up, two untimed launches ahead of every batch. A batch lasts "
             "at least a\n  millisecond and fifty launch floors.\n";
        s += "- A time **includes the launch floor** below, the gap between two dependent "
             "dispatches, which\n  the engine pays too; a kernel near the floor is "
             "launch-bound, not slow.\n";
        s += "- In-place operands are restored from a pristine copy before every launch, so "
             "mutation does not\n  drift. The restores are timed in a batch of their own, each "
             "followed by a floor launch in the\n  kernel's place, and subtracted, the floor "
             "added back (`restore` in the notes is their share\n  of the batch). A large share "
             "leaves the kernel a small difference of two large\n  times, which the interval "
             "shows as noise but not as bias (the restores leave the operand in\n  the cache): "
             "a share of 75% or more, or a time under the launch floor, is noted as an order\n"
             "  of magnitude.\n";
        s += "- The timing pass draws its own **routing and block tables**: expert offsets and "
             "ids skewed as\n  real routing is (Zipf, exponent 0.5) and block tables as "
             "permutations. The checked launch\n  keeps the recorded draw.\n";
        s += "- `±%` is the half-width of a distribution-free 95% interval for the median, as a "
             "percent of\n  it (below six batches the range, which covers less); a case is "
             "timed until it is 1% or\n  the rep cap (`--bench-reps`) is reached, so a wide one "
             "marks a noisy case. `n` counts the\n  timed batches. "
             "`MHz` is the firmware's averaged graphics\n  clock read after each batch; a batch "
             "during which a throttler came on that was off at idle and\n  after the warm-up is "
             "discarded and repeated.\n";
        s += "- **Host rows** (their own column) are steady_clock times on the calling thread, "
             "after an untimed\n  launch, with in-place operands restored outside the timed "
             "region; " +
             (bench->host_footprint > 0
                  ? "weights, weight tables and\n  inputs of " +
                        humanb(bench->host_footprint / 32) + " or more are rotated over copies "
                        "spanning " + humanb(bench->host_footprint) + " (" +
                        (bench->footprint_flag ? std::string("`--bench-flush`")
                                               : std::string("twice the CPU's last-level\n  "
                                                             "cache, at least 256 MiB")) +
                        "), so read from memory"
                  : std::string("every operand is reused\n  (`--bench-flush 0`, warm)")) +
             ". The notes name the OpenMP threads.\n";
        s += "- `operands` is the total the case allocated, not what the kernel read (a gather "
             "touches M rows\n  of a table, not the table), so dividing the two gives no "
             "bandwidth figure.\n\n";
        if (!bench->header.empty()) {
            s += "Device timing and its calibration:\n\n";
            for (const std::string& h : bench->header) s += "> " + md_escape(h) + "\n>\n";
            s.resize(s.size() - 2);
            s += "\n";
        }
        if (!bench->withheld.empty())
            s += "**Device timings withheld:** " + md_escape(bench->withheld) + "\n\n";
        s += "| op | kernel | library | geometry | device us | host us | min | p90 | n | ±% | MHz "
             "| operands | notes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
        /* Sorted by cost, descending; then the rows that were to be timed and were not. */
        std::vector<std::string> by_cost, untimed;
        for (const std::string& k : order) {
            if (!agg[k].us.empty()) by_cost.push_back(k);
            else if (!agg[k].tnote.empty()) untimed.push_back(k);
        }
        std::sort(by_cost.begin(), by_cost.end(), [&](const std::string& a, const std::string& b) {
            return median_of(agg[a].us) > median_of(agg[b].us);
        });
        for (const std::string& key : untimed) by_cost.push_back(key);
        for (const std::string& key : by_cost) {
            Agg& a = agg[key];
            const Row& r = a.first;
            const bool host = std::string(r.domain) == domain_name(RAD_DOMAIN_HOST);
            const double us = median_of(a.us);
            const std::string t = us < 0 ? std::string("-") : fmt("%.2f", us);
            std::string notes = a.tnote;
            if (!a.restore.empty())
                notes = fmt("restore %.0f%%", median_of(a.restore)) +
                        (notes.empty() ? "" : "; ") + notes;
            if (host && a.threads > 0)
                notes = fmt("%d thread(s)", a.threads) + (notes.empty() ? "" : "; ") + notes;
            s += "| `" + md_escape(r.op) + "` | `" + md_escape(r.kernel) + "` | " +
                 md_escape(r.plugin) + " | `" + md_escape(r.geom) + "` | " +
                 (host ? std::string("-") : t) + " | " + (host ? t : std::string("-")) + " | " +
                 (a.us_min.empty() ? std::string("-")
                                   : fmt("%.2f", *std::min_element(a.us_min.begin(),
                                                                   a.us_min.end()))) + " | " +
                 (a.us_p90.empty() ? std::string("-") : fmt("%.2f", median_of(a.us_p90))) +
                 " | " + (a.us_n ? fmt("%d", a.us_n) : std::string("-")) + " | " +
                 (a.worst_ci < 0 ? std::string("-") : fmt("%.1f", a.worst_ci)) + " | " +
                 (a.clock.empty() ? std::string("-") : fmt("%.0f", median_of(a.clock))) + " | " +
                 (r.bytes > 0 ? humanb(r.bytes) : std::string("-")) + " | " +
                 md_escape(notes) + " |\n";
        }
    }

    /* ================================================================ 4b. memory safety */
    s += "\n## Memory safety\n\n";
    s += "Every device allocation in this run (operands and kernel scratch) had a poisoned "
         "256-byte margin\non both sides, verified after the checked launch. This catches "
         "out-of-bounds writes, which\ncorrupt a neighbouring tensor; out-of-bounds reads go "
         "unseen.\n\n";
    s += "AMD's amdgcn AddressSanitizer (`/opt/rocm/amdgcn/bitcode/asanrtl.bc`) needs `xnack+` "
         "for its\npage-fault-based shadow, which RDNA does not implement (`rocminfo` reports "
         "`XNACK enabled: NO`),\nso the margins stand in for it.\n\n";
    s += fmt("- %d case(s) ran inside guarded allocations\n", st.guarded);
    s += fmt("- %d wrote outside one\n", st.overrun);
    if (!overruns.empty()) {
        s += "\n| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : overruns)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    /* ================================================================ 5. the gaps */
    s += "\n## Not covered\n\n";
    s += "A skip means the tool declined a case rather than invent an extent, a dtype or an "
         "oracle it\ndoes not have. Each line below is a gap in coverage, not a failing "
         "kernel.\n\n";
    /* Collectives are named explicitly: they need a rendezvous with real peer ranks, so a single
     * process cannot check them (a kernel waiting for a missing peer hangs). */
    s += "**Collective ops are not checked.** `all_reduce`, `all_gather` and the fused all-reduce "
         "forms\nrendezvous with peer ranks over peer memory; in a single process a kernel waiting "
         "for a peer\nhangs. They appear in the completeness matrix but carry no correctness or "
         "speed number.\n\n";
    if (!skipped.empty()) {
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : skipped)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    if (!diverged.empty()) {
        s += "\n## Divergences\n\n";
        s += "Two libraries implement the same op name but describe an output differently. These "
             "cases are\ncounted apart from passes and failures, since comparing them would compare "
             "two different\ntensors. One of the two libraries needs a different op name.\n\n";
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : diverged)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    if (!fuse_skipped.empty()) {
        s += "\n## Fusions not checked\n\n";
        s += "A kernel that declares `replaces` is run against that chain from its own library "
             "and the outputs\ncompared byte for byte. These are the cases where the chain could "
             "not be assembled.\n\n";
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : fuse_skipped)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }

    /* ================================================================ 6. what the analysers say
     *
     * Static-analysis output produced elsewhere (e.g. CI), spliced in verbatim. A missing file
     * omits the section with a warning. */
    if (!analysis_path.empty()) {
        FILE* af = std::fopen(analysis_path.c_str(), "r");
        if (af) {
            s += "\n";
            char buf[4096];
            size_t got;
            while ((got = std::fread(buf, 1, sizeof buf, af)) > 0) s.append(buf, got);
            std::fclose(af);
        } else {
            RAD_WARN("--analysis %s: %s (the section is omitted)", analysis_path.c_str(),
                     std::strerror(errno));
        }
    }

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { RAD_ERR("--report %s: %s", path.c_str(), std::strerror(errno)); return RAD_E_IO; }
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
    return RAD_OK;
}
/* A guarded device allocation: the real base, the pointer handed to the kernel (past the red
 * zone), and whether this case owns it or borrowed it from the weight cache, which outlives
 * cases. */
struct DevBuf {
    void*   base = nullptr;
    void*   p = nullptr;
    int64_t bytes = 0;
    bool    on = false;      /* guarded, so its margins are worth reading back */
    bool    owned = true;    /* false: the cache frees it, not the case that used it */
    /* The description the relayout produced, cached with the bytes. */
    RadTensor desc{};
};

/* ================================================================== red zones
 *
 * An out-of-bounds write (last-tile off-by-one, stored tile padding, a wrong split-K stride) lands
 * in a separate allocation here and would pass, while in the engine it corrupts another tensor.
 * Device ASan cannot run on RDNA (amdgcn ASan needs xnack+, which RDNA lacks), so every device
 * allocation gets a poisoned margin on both sides, checked after the launch. This catches
 * out-of-bounds writes, not reads.
 *
 * 256 bytes: wide enough for a tile's worth of overrun, and a multiple of every operand alignment
 * the kernels require.
 */
constexpr int64_t RED_BYTES = 256;
constexpr int     RED_FILL  = 0xA5;

/* Mixed into an operand's seed for the timing pass's own draw (kb::timing_input), so it never
 * repeats the recorded one. */
constexpr uint64_t kTimingSalt = 0x74696d696e67ull;   /* "timing" */

/* `bytes` between two poisoned margins; the kernel is handed d.p. */
bool guarded_alloc(DevBuf& d, int64_t bytes, RadStream stream) {
    d.bytes = bytes;
    d.base  = rad_dev_alloc(bytes + 2 * RED_BYTES, RAD_MEM_DEVICE);
    if (!d.base) return false;
    d.p = (uint8_t*)d.base + RED_BYTES;
    d.on = true;
    return rad_memset_async(d.base, RED_FILL, RED_BYTES, stream) >= 0 &&
           rad_memset_async((uint8_t*)d.p + bytes, RED_FILL, RED_BYTES, stream) >= 0;
}

/* Reads back the margins of every guarded operand and of the scratch after the checked launch,
 * and records each overrun as "<op>  (<who> wrote N byte(s) before the start ... of <what>)".
 * Distances are from the buffer, not counts of changed bytes. */
void check_margins(const std::vector<DevBuf>& dev, const DevBuf& scratch,
                   const RadOpSchema* schema, const std::string& op, const std::string& who,
                   const Geometry& g, RadStream stream, std::map<std::string, int>* overruns,
                   Stats* st) {
    auto check = [&](const DevBuf& d, const std::string& what) {
        if (!d.on) return;
        uint8_t lo[RED_BYTES], hi[RED_BYTES];
        if (rad_memcpy_async(lo, d.base, RED_BYTES, stream) < 0) return;
        if (rad_memcpy_async(hi, (const uint8_t*)d.p + d.bytes, RED_BYTES, stream) < 0) return;
        if (rad_stream_sync(stream) < 0) return;
        const kb::Overrun o = kb::scan_margins(lo, hi, RED_BYTES, (uint8_t)RED_FILL);
        if (!o.before && !o.after) return;
        std::string m = op + "  (" + who + " wrote ";
        if (o.before) m += fmt("%lld byte(s) before the start", (long long)o.before);
        if (o.before && o.after) m += " and ";
        if (o.after)  m += fmt("%lld byte(s) past the end", (long long)o.after);
        m += " of " + what + ")";
        (*overruns)[m]++;
        ++st->overrun;
        std::printf("  %-22s %-44s out of bounds: %s\n", op.c_str(), g.str().c_str(), m.c_str());
    };
    for (size_t k = 0; k < dev.size(); ++k)
        check(dev[k], std::string("operand '") +
                      (schema->operands[k].name ? schema->operands[k].name : "?") + "'");
    check(scratch, "its own scratch");
    ++st->guarded;
}

/* The launch arguments of a single-rank case. */
RadArgs case_args(std::vector<RadTensor>& t, const Geometry& g) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = g.params(); a.n_p = g.n_params();
    a.rank = 0; a.world_size = 1;
    return a;
}

/* A host tensor description of operand `d` over `data`. */
RadTensor host_tensor(const KfOperand& d, const void* data) {
    RadTensor x{};
    x.data  = const_cast<void*>(data);
    x.dtype = d.dtype;
    x.rank  = (uint32_t)d.shape.size();
    for (size_t i = 0; i < d.shape.size() && i < RAD_MAX_RANK; ++i) x.shape[i] = d.shape[i];
    rad_tensor_pack(&x);
    return x;
}

/* The library the others are checked against: the one --ref names (it must be loaded, see
 * --kernels), else the first kernel library that declares itself a reference implementation
 * (RadPluginReferenceFn). Found by declaration, never by name. */
std::string resolve_reference(const Registry& reg, const std::string& named) {
    if (!named.empty()) {
        const Plugin* p = reg.plugin(named);
        if (p && p->kind == RAD_PLUGIN_KERNEL) return p->name;
        RAD_ERR("--ref names '%s', which is not a loaded kernel library: name it in --kernels "
                "too when that lists the libraries to load", named.c_str());
        return std::string();
    }
    for (const Plugin& p : reg.plugins())
        if (p.kind == RAD_PLUGIN_KERNEL && p.reference) return p.name;
    RAD_ERR("no kernel library loaded declares itself a reference implementation, so there is "
            "nothing to check against: name one with --ref, and in --kernels when that lists the "
            "libraries to load (libref is the one radiance ships)");
    return std::string();
}

/* ================================================================== which row runs
 *
 * Each library is asked as the engine would ask it with that library alone (Registry::select_in):
 * the same constraint match, derived keys and encoding filter. The filter is the builder's own
 * (weight_refusal), applied to the encodings the case recorded, so a row the engine would skip
 * for a weight's encoding is skipped here too and the next row runs instead. */
Registry::RowFilter weight_filter(const KfCase& c, const RadOpSchema* schema) {
    return [&c, schema](const KernelRow& r, const Geometry& g) -> std::string {
        for (size_t k = 0; k < c.opd.size(); ++k) {
            const KfOperand& d = c.opd[k];
            if (d.absent || !d.has_enc) continue;
            if (d.role != RAD_OPD_WEIGHT && d.role != RAD_OPD_WTAB) continue;
            RadTensor pt[RAD_ENC_MAX_PLANES];
            const int n = (int)std::min<size_t>(d.sel.size(), RAD_ENC_MAX_PLANES);
            for (int j = 0; j < n; ++j) {
                pt[j] = RadTensor{};
                pt[j].dtype = d.enc.plane[d.sel[(size_t)j]].dtype;
                pt[j].rank = 2;
                pt[j].shape[0] = d.sel_rows[(size_t)j];
                pt[j].shape[1] = d.sel_cols[(size_t)j];
                pt[j].stride[0] = d.sel_cols[(size_t)j];
                pt[j].stride[1] = 1;
            }
            const char* nm = schema && (int)k < schema->n_operands && schema->operands[k].name
                                 ? schema->operands[k].name : "?";
            std::string why = weight_refusal(r.info, g, (int)k, d.enc, d.sel.data(), pt, n, nm);
            if (!why.empty()) return why;
        }
        return std::string();
    };
}

/* One library's run of one case: the row, and the geometry it launches with (the recording plus
 * the row's derived values the recording does not name, as the engine's resolve() writes them). */
struct Run {
    std::string      lib;
    int              dom = 0;
    const KernelRow* row = nullptr;
    std::string      cap;            /* its cap key, when --max-cases caps */
    Geometry         g;
};

/* The rows of `lib` that run a case: the one select_in() picks, or with `all` every row whose
 * constraints and filter hold, best first. */
std::vector<Run> rows_for_case(const Registry& reg, const std::string& lib, const std::string& op,
                               const Geometry& g, int dom, const Registry::RowFilter& accept,
                               bool all) {
    std::vector<Run> out;
    std::set<const KernelRow*> taken;
    const Registry::RowFilter f = [&](const KernelRow& r, const Geometry& gg) -> std::string {
        if (taken.count(&r)) return "already taken";
        return accept(r, gg);
    };
    for (;;) {
        SelectionTrace t;
        const KernelRow* row = reg.select_in(lib, op, g, dom, &f, &t);
        if (!row || !row->info->launch) break;
        Run r;
        r.lib = lib;
        r.dom = dom;
        r.row = row;
        r.g = g;
        for (const SelectionTrace::DerivedParam& d : t.derived)
            if (d.known && !r.g.has(d.key)) r.g.set_i(d.key, d.value);
        out.push_back(std::move(r));
        taken.insert(row);
        if (!all) break;
    }
    return out;
}

/* ================================================================== a case's inputs
 *
 * Drawn once per case and shared, read-only, by every library: each operand's bytes as the
 * reference reads them, a weight's planes for the layout hooks, and a fingerprint. Weights are
 * also shared across cases through the weight cache (an M sweep asks for the same weight at every
 * point). Only what a kernel writes gets a private copy. */
using Bytes = std::vector<uint8_t>;
struct Drawn {
    std::shared_ptr<const Bytes>              bytes;
    std::shared_ptr<const std::vector<Bytes>> planes;
    uint64_t                                  fp = 0;     /* kb::fingerprint, 0 until computed */
};

/* ================================================================== the replay
 *
 * Every library on every recorded case, with no model and no live reference run: the fixture holds
 * the reference's answers. No declare phase (the geometries are in the file); each library is asked
 * for its own row (rows_for_case), so libraries can be compared.
 */
struct ReplayOpts {
    std::string home;
    std::vector<std::string> hierarchy;
    std::string ref_arg;
    std::vector<std::string> op_filters;
    double  tol_override = -1;
    int     max_cases = 4;
    int64_t max_case_bytes = 0;
    bool    bench = false, timing_only = false, verbose = false;
    /* The timing pass's settings (kb::TimerOpts), from the --bench-* flags. */
    kb::TimerOpts timer;
    std::string report_path, analysis_path;
    std::vector<std::string> fixtures;
    /* Stale fixtures fail the run (the ctests). */
    bool    strict = false;
    /* Leave the reference library out of the libraries under test: its answers are the fixture. */
    bool    no_ref = false;
    /* The reference cache (kb::RefCache). */
    bool    no_cache = false;
    std::string cache_dir;
    /* Every row whose constraints hold, not only the one selection picks. */
    bool    all_rows = false;
    /* The row coverage ratchet (kb::ratchet). */
    std::string coverage_dir;
    bool    coverage_write = false;
    /* Write the fixtures' cases again with the reference's answers (--rerecord), or a row-driven
     * synthetic fixture (--record-rows). */
    std::string rerecord_path, record_rows_path;
};

class Replay {
public:
    explicit Replay(const ReplayOpts& o) : o_(o) {}
    int run();

private:
    struct Coverage {
        std::set<std::string> reached, unknown;
    };

    int  load();
    void replay_fixture(size_t fi);
    void replay_case(size_t fi, const KfCase& c);
    bool draw_case(const KfCase& c, std::vector<Drawn>* in, std::string* why);
    void run_one(size_t fi, const KfCase& c, const RadOpSchema* schema,
                 const std::vector<Drawn>& in, uint64_t input_hash, const Run& run);
    bool lay_out(const KfCase& c, size_t k, const RadOpSchema* schema, const Run& run,
                 const Drawn& d, Bytes* laid, RadTensor* desc);
    int  record_case(const KfCase& base, KfCase* out, std::string* why);
    int  rerecord();
    int  record_rows();
    int  finish();

    const ReplayOpts& o_;
    LoadedPlugins plugins_;
    Registry* reg_ = nullptr;
    std::string ref_name_;
    std::vector<std::string> libs_;            /* under test, hierarchy order */
    std::vector<Kfixture> fx_;
    Stats st_;
    Named nm_;
    Phases ph_;
    std::vector<Row> rows_;
    std::map<std::string, int> group_seen_;
    std::set<std::string> ran_keys_;           /* cap keys whose case reached its row */
    /* Per fixture, per library: the rows reached and the rows a skipped case would have run. */
    std::vector<std::map<std::string, Coverage>> cov_;
    kb::RefCache cache_;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

    /* Drawn weights by kb::host_weight_key, bounded by half the case budget, and the one most
     * recent weight past that budget (an M sweep of a vocabulary-sized table), kept only while
     * consecutive cases read it. */
    std::map<uint64_t, Drawn> wcache_;
    int64_t wcache_bytes_ = 0;
    Drawn    recent_;
    uint64_t recent_key_ = 0;
    /* Laid-out device weights by kb::device_weight_key, bounded by the case budget. */
    std::map<std::string, DevBuf> dcache_;
    int64_t dcache_bytes_ = 0;
    /* --bench: the device cases waiting for the timing pass (made at the first one), and the
     * rotation footprint host rows are timed with. */
    std::unique_ptr<kb::DeviceTimer> timer_;
    int64_t host_footprint_ = 0;
};

int Replay::load() {
    PhaseTimer pt(&ph_, kb::PH_LOAD);
    for (const std::string& path : o_.fixtures) {
        Kfixture fx;
        const int rc = fx.load(path);
        if (rc < 0) {
            RAD_ERR("no fixture at %s: %s", path.c_str(), rad_strerror(rc));
            RAD_ERR("record one with: rad-kbench -m <model.rad> --record %s", path.c_str());
            return 1;
        }
        fx_.push_back(std::move(fx));
    }
    /* Kernel plugins only, no architecture: there is no container, and a kernel library must not
     * need one. */
    if (rad_tools_load_plugins(o_.home, o_.hierarchy, "", "", &plugins_) < 0) return 1;
    reg_ = &rad_tools_registry();
    ref_name_ = resolve_reference(*reg_, o_.ref_arg);
    if (ref_name_.empty()) return 1;

    for (const Plugin& p : reg_->plugins())
        if (p.kind == RAD_PLUGIN_KERNEL && !(o_.no_ref && p.name == ref_name_))
            libs_.push_back(p.name);
    if (libs_.empty() && o_.rerecord_path.empty() && o_.record_rows_path.empty()) {
        RAD_ERR("no kernel library under test loaded from %s/kernels%s", o_.home.c_str(),
                o_.no_ref ? " (--no-ref leaves the reference out)" : "");
        return 1;
    }
    cov_.resize(fx_.size());

    /* The reference cache: keyed by the reference's code and everything it links, except the
     * other kernel libraries, which cannot change its answers. */
    const bool ref_runs = std::find(libs_.begin(), libs_.end(), ref_name_) != libs_.end();
    if (!o_.no_cache && ref_runs && !o_.timing_only) {
        const Plugin* rp = reg_->plugin(ref_name_);
        std::vector<std::string> others;
        for (const Plugin& p : reg_->plugins())
            if (p.name != ref_name_ && !p.path.empty()) others.push_back(p.path);
        const uint64_t id = rp ? kb::code_identity(rp->path, others) : 0;
        if (id == 0)
            RAD_WARN("rad-kbench: %s has no build-id, so its results are not cached",
                     ref_name_.c_str());
        cache_.open(o_.cache_dir.empty() ? o_.home + "/cache/kbench" : o_.cache_dir, id);
    }
    return 0;
}

int Replay::run() {
    if (load() != 0) return 1;
    if (!o_.rerecord_path.empty()) return rerecord();
    if (!o_.record_rows_path.empty()) return record_rows();

    std::printf("rad-kbench %s\n", o_.fixtures.size() == 1 ? o_.fixtures[0].c_str() : "");
    for (size_t i = 0; i < fx_.size(); ++i)
        std::printf("  fixture      %s: %zu case(s), version %u, recorded %s from %s\n",
                    o_.fixtures[i].c_str(), fx_[i].cases.size(), fx_[i].version,
                    fx_[i].recorded.empty() ? "?" : fx_[i].recorded.c_str(),
                    fx_[i].source.empty() ? "?" : fx_[i].source.c_str());
    std::printf("  reference    %s, %d sampled position(s) an operand plus the full norm\n",
                ref_name_.c_str(), KF_SAMPLES);
    std::string under;
    for (const std::string& l : libs_) under += (under.empty() ? "" : " ") + l;
    std::printf("  libraries    %s (under test: %s)\n", plugins_.libraries().c_str(),
                under.c_str());
    if (cache_.enabled()) std::printf("  cache        %s\n", cache_.dir().c_str());
    if (o_.bench) {
        host_footprint_ = o_.timer.footprint >= 0 ? o_.timer.footprint
                                                  : kb::default_footprint(kb::host_llc_bytes());
        std::printf("  timing       after the checks: device rows %s, host rows %s\n",
                    o_.timer.footprint == 0 ? "warm" : "with cold weights",
                    host_footprint_ > 0 ? ("rotated over " + humanb(host_footprint_)).c_str()
                                        : "warm");
    }
    std::printf("\n");

    for (size_t fi = 0; fi < fx_.size(); ++fi) replay_fixture(fi);
    return finish();
}

void Replay::replay_fixture(size_t fi) {
    for (const KfCase& c : fx_[fi].cases) {
        if (!o_.op_filters.empty()) {
            bool hit = false;
            for (const std::string& f : o_.op_filters)
                if (c.op.rfind(f, 0) == 0) { hit = true; break; }
            if (!hit) continue;
        }
        replay_case(fi, c);
    }
}

/* The fixture's reference library, without any "@version" suffix. */
std::string oracle_name(const Kfixture& fx) {
    return fx.oracle.substr(0, fx.oracle.find('@'));
}

/* The recorded parameters as a geometry. */
Geometry case_geometry(const KfCase& c) {
    Geometry g;
    for (const KfParam& p : c.params) {
        if (p.kind == RAD_P_INT)      g.set_i(p.key, p.i);
        else if (p.kind == RAD_P_F64) g.set_f(p.key, p.d);
        else if (p.kind == RAD_P_STR) g.set_s(p.key, p.s);
    }
    return g;
}

/* Where the recording and a row's own description of an output disagree, or "". Inputs are not
 * compared: a differently read input shows up as a wrong output. */
std::string describes_differently(const KfCase& c, const RadOpSchema* schema, const KernelRow* row,
                                  const Geometry& g) {
    if (!row->info->opd_shape) return std::string();
    std::vector<OpdShape> ksh;
    if (rad_operand_shapes_hook(row->info, g, schema->n_operands, &ksh) < 0 ||
        ksh.size() != c.opd.size())
        return std::string();
    for (size_t k = 0; k < ksh.size(); ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        if (c.opd[k].absent || ksh[k].absent) continue;
        if (c.opd[k].dtype == ksh[k].dtype && c.opd[k].shape == ksh[k].shape) continue;
        OpdShape rec;
        rec.dtype = c.opd[k].dtype;
        rec.shape = c.opd[k].shape;
        const char* nm = schema->operands[k].name ? schema->operands[k].name : "?";
        return std::string("'") + nm + "': the recording says " + shape_str(rec) + ", " +
               row->info->name + " says " + shape_str(ksh[k]);
    }
    return std::string();
}

void Replay::replay_case(size_t fi, const KfCase& c) {
    const Kfixture& fx = fx_[fi];
    const RadOpSchema* schema = reg_->schema(c.op);
    if (!schema) {
        nm_.stale[c.op + "  (no plugin declares a schema for it)"]++;
        ++st_.stale;
        ++st_.skipped;
        return;
    }
    /* Operands are positional, so a fixture whose count differs from the schema's is stale. */
    if ((int)c.opd.size() != schema->n_operands) {
        nm_.stale[c.op + "  (fixture recorded " + std::to_string(c.opd.size()) + " operand(s), "
                  "the schema declares " + std::to_string(schema->n_operands) +
                  " -- re-record this fixture)"]++;
        ++st_.stale;
        ++st_.skipped;
        return;
    }
    const Geometry g = case_geometry(c);
    const Registry::RowFilter accept = weight_filter(c, schema);

    /* The reference must still serve and describe what it recorded; otherwise the case is stale
     * on every machine, whichever libraries are under test. */
    if (oracle_name(fx) == ref_name_) {
        std::vector<Run> rr = rows_for_case(*reg_, ref_name_, c.op, g, RAD_DOMAIN_HOST, accept,
                                            false);
        if (rr.empty()) {
            nm_.stale[c.op + "  (" + ref_name_ + " recorded this case and has no row for it "
                      "now)"]++;
            ++st_.stale;
        } else {
            const std::string why = describes_differently(c, schema, rr[0].row, rr[0].g);
            if (!why.empty()) {
                nm_.stale[c.op + "  (" + ref_name_ + " now describes an output differently from "
                          "its recording -- " + why + ")"]++;
                ++st_.stale;
                return;
            }
        }
    }

    /* Which libraries run this case, decided before anything is drawn, so a case every library
     * caps or skips costs nothing. */
    std::vector<Run> runs;
    bool served = false;
    for (const std::string& lib : libs_) {
        for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
            /* A timing-only run times device kernels; a host row has no clock here. */
            if (o_.timing_only && dom != RAD_DOMAIN_DEVICE) continue;
            for (Run& r : rows_for_case(*reg_, lib, c.op, g, dom, accept, o_.all_rows)) {
                served = true;
                if (o_.max_cases > 0) {
                    r.cap = kb::cap_key(c, r.row->info->name, lib, dom, g);
                    if (++group_seen_[r.cap] > o_.max_cases) {
                        /* An identical case ran, perhaps from another fixture, so this fixture
                         * reaches the row too; if that one was skipped, the row is unknown. */
                        Coverage& cv = cov_[fi][lib];
                        (ran_keys_.count(r.cap) ? cv.reached : cv.unknown)
                            .insert(r.row->info->name);
                        ++st_.capped;
                        continue;
                    }
                }
                /* Operands are recorded from the reference's description. A kernel whose own
                 * hook describes an output differently (another scale grid, say) is a
                 * divergence, not a comparison. */
                const std::string why = describes_differently(c, schema, r.row, r.g);
                if (!why.empty()) {
                    nm_.diverged[c.op + "  (" + lib + "/" + r.row->info->name + " describes an "
                                 "output differently -- " + why + ")"]++;
                    ++st_.diverged;
                    continue;
                }
                runs.push_back(std::move(r));
            }
        }
    }
    if (!served) {
        nm_.unserved[c.op + "  (no library under test has a row for it)"]++;
        ++st_.unserved;
        return;
    }
    if (runs.empty()) return;

    /* Skip by name a case over --max-bytes rather than be OOM-killed (replays run in ctest on
     * arbitrary machines). The rows it would have run are unknown to the coverage ratchet, not
     * lost. */
    const int64_t want = kb::case_bytes(c.opd);
    if (want < 0 || want > o_.max_case_bytes) {
        nm_.skipped[c.op + "  (one case needs " + (want < 0 ? std::string("more than int64")
                                                            : humanb(want)) +
                    ", over the " + humanb(o_.max_case_bytes) +
                    " budget -- raise --max-bytes to check it)"]++;
        ++st_.skipped;
        for (const Run& r : runs) cov_[fi][r.lib].unknown.insert(r.row->info->name);
        return;
    }

    std::vector<Drawn> in;
    std::string why;
    if (!draw_case(c, &in, &why)) {
        nm_.skipped[c.op + "  (a weight's planes do not make the operand the reference reads, " +
                    why + ")"]++;
        ++st_.skipped;
        return;
    }

    /* Fingerprints: the reference cache's key, and the check that a host kernel left its inputs
     * alone. Only paid for when a host row or the cache needs them. */
    uint64_t input_hash = 0;
    bool need_fp = false;
    for (const Run& r : runs) need_fp = need_fp || r.dom == RAD_DOMAIN_HOST;
    if (need_fp) {
        PhaseTimer pt(&ph_, kb::PH_HASH);
        input_hash = 0x6b62696e70757473ull;
        for (size_t k = 0; k < in.size(); ++k) {
            Drawn& d = in[k];
            if (d.bytes && !d.fp) {
                d.fp = kb::fingerprint(d.bytes->data(), d.bytes->size());
                /* A cached weight keeps its fingerprint for the rest of the sweep. */
                if (c.opd[k].role == RAD_OPD_WEIGHT) {
                    auto it = wcache_.find(kb::host_weight_key(c.opd[k], k));
                    if (it != wcache_.end() && it->second.bytes == d.bytes) it->second.fp = d.fp;
                    if (recent_.bytes == d.bytes) recent_.fp = d.fp;
                }
            }
            input_hash = kb::hash_combine(input_hash, d.fp);
        }
    }
    for (const Run& r : runs) run_one(fi, c, schema, in, input_hash, r);
}

bool Replay::draw_case(const KfCase& c, std::vector<Drawn>* in, std::string* why) {
    PhaseTimer pt(&ph_, kb::PH_DRAW);
    in->assign(c.opd.size(), Drawn{});
    const int64_t vocab_rows = kf_vocab_rows(c.opd);
    /* The weight kept past the cache's budget is released before anything is drawn unless this
     * case reads it, so it never adds to a different case's memory. */
    if (recent_.bytes) {
        bool used = false;
        for (size_t k = 0; k < c.opd.size() && !used; ++k)
            used = c.opd[k].role == RAD_OPD_WEIGHT && !c.opd[k].absent &&
                   kb::host_weight_key(c.opd[k], k) == recent_key_;
        if (!used) recent_ = Drawn{};
    }
    for (size_t k = 0; k < c.opd.size(); ++k) {
        const KfOperand& d = c.opd[k];
        Drawn& out = (*in)[k];
        if (d.absent) continue;
        if (d.role == RAD_OPD_WEIGHT) {
            /* Cached: an M sweep requests the same weight at every point, and kf_weight_seed makes
             * those requests identical. */
            const uint64_t ck = kb::host_weight_key(d, k);
            auto it = wcache_.find(ck);
            if (it != wcache_.end()) { out = it->second; continue; }
            if (recent_.bytes && recent_key_ == ck) { out = recent_; continue; }
            auto bytes = std::make_shared<Bytes>();
            auto planes = std::make_shared<std::vector<Bytes>>();
            if (d.has_enc) {
                kf_draw_planes(d, kf_weight_seed(d, k), planes.get());
                if (kf_planes_logical(d, *planes, bytes.get(), why) != RAD_OK) {
                    *why = enc_name(d.enc) + ": " + *why;
                    return false;
                }
            } else {
                kf_draw(c.opd, k, c.m, vocab_rows, 0, bytes.get());
            }
            out.planes = planes;
            int64_t held = 0;
            for (const Bytes& pl : *planes) held += (int64_t)pl.size();
            /* A single plane read as stored decodes to itself (a plain bf16 vocabulary table):
             * one buffer, not two. */
            if (planes->size() == 1 && (*planes)[0] == *bytes) {
                out.bytes = std::shared_ptr<const Bytes>(planes, &(*planes)[0]);
            } else {
                out.bytes = bytes;
                held += (int64_t)bytes->size();
            }
            if (wcache_bytes_ + held <= o_.max_case_bytes / 2) {
                wcache_bytes_ += held;
                wcache_.emplace(ck, out);
            } else {
                recent_ = out;
                recent_key_ = ck;
            }
            continue;
        }
        if (d.role == RAD_OPD_WTAB && d.has_enc && !d.undrawn) {
            auto bytes = std::make_shared<Bytes>();
            if (kf_draw_table(d, k, bytes.get(), why) != RAD_OK) {
                *why = enc_name(d.enc) + ": " + *why;
                return false;
            }
            out.bytes = bytes;
            continue;
        }
        auto bytes = std::make_shared<Bytes>();
        kf_draw(c.opd, k, c.m, vocab_rows, kf_operand_seed(c.seed, (int)k), bytes.get());
        out.bytes = bytes;
    }
    return true;
}

/* Whether `row` lays out any weight operand of `c` from its planes rather than reading the tensor
 * they decode to: its layout hook answers RAD_OK for the operand's encoding and selection. Such a
 * row reads the planes' exact values (a joined fp8 lm head, abi/rad_planes.h), which the decoded
 * tensor the recording's reference read may round, so its answer can differ from that recording
 * by the rounding even when it is the recording's own library. */
static bool lays_out_weight(const KernelRow* row, const Geometry& g, const KfCase& c) {
    const RadKernelInfo* info = row ? row->info : nullptr;
    if (!info || !info->layout) return false;
    for (size_t k = 0; k < c.opd.size(); ++k) {
        const KfOperand& d = c.opd[k];
        if (d.absent || d.role != RAD_OPD_WEIGHT || !d.has_enc) continue;
        RadTensor pt[RAD_ENC_MAX_PLANES];
        const int n = (int)d.sel.size();
        if (n < 1 || n > RAD_ENC_MAX_PLANES) continue;
        for (int j = 0; j < n; ++j) {
            pt[j] = RadTensor{};
            pt[j].dtype = d.enc.plane[d.sel[(size_t)j]].dtype;
            pt[j].rank = 2;
            pt[j].shape[0] = d.sel_rows[(size_t)j];
            pt[j].shape[1] = d.sel_cols[(size_t)j];
            rad_tensor_pack(&pt[j]);
        }
        RadLayout lay{};
        if (info->layout(g.params(), g.n_params(), (int)k, &d.enc, d.sel.data(), pt, n, &lay) ==
            RAD_OK)
            return true;
    }
    return false;
}

/* Each kernel lays the weight out from the shared planes: `d.bytes` is what the reference read
 * (the planes decoded to the operand's tensor), `d.planes` the canonical planes its layout hooks
 * arrange, which is why the fixture never stores a laid-out plane. False with a named skip when the
 * row cannot take them; `laid` empty means the row reads the operand as stored. */
bool Replay::lay_out(const KfCase& c, size_t k, const RadOpSchema* schema, const Run& run,
                     const Drawn& d, Bytes* laid, RadTensor* desc) {
    PhaseTimer pt(&ph_, kb::PH_LAYOUT);
    static const std::vector<Bytes> none;
    RadLayout lay{};
    const int lrc = kf_lay_weight(run.row->info, run.g.params(), run.g.n_params(), (int)k,
                                  c.opd[k], d.planes ? *d.planes : none, &lay, laid);
    if (lrc == RAD_E_UNSUPPORTED) { laid->clear(); return true; }
    if (lrc < 0) {
        const char* opd_name = schema->operands[k].name ? schema->operands[k].name : "?";
        nm_.skipped[c.op + "  (" + run.lib + "/" + run.row->info->name + " cannot lay out '" +
                    opd_name + "' from " + enc_name(c.opd[k].enc) + ": " + rad_strerror(lrc) +
                    ")"]++;
        return false;
    }
    kf_describe_laid(c.opd[k], lay, laid->data(), desc);
    return true;
}

void Replay::run_one(size_t fi, const KfCase& c, const RadOpSchema* schema,
                     const std::vector<Drawn>& in, uint64_t input_hash, const Run& run) {
    const size_t n_opd = c.opd.size();
    const KernelRow* row = run.row;
    const std::string& lib = run.lib;
    const int dom = run.dom;
    const Geometry& g = run.g;
    /* The recording's own library is held to its answers exactly, unless this row reads a weight's
     * planes where the recording read their decoded tensor (lays_out_weight). */
    const bool exact = lib == oracle_name(fx_[fi]) && !lays_out_weight(row, g, c);
    auto role_of = [&](size_t k) { return schema->operands[k].role; };
    auto written = [&](size_t k) { return role_of(k) == RAD_OPD_OUT || role_of(k) == RAD_OPD_INOUT; };
    auto name_of = [&](size_t k) {
        return schema->operands[k].name ? schema->operands[k].name : "?";
    };

    /* The reference's reduced outputs may come from the cache (kb::RefCache). The verdict below
     * still runs. */
    const bool cacheable = cache_.enabled() && lib == ref_name_ && dom == RAD_DOMAIN_HOST &&
                           !o_.timing_only;
    uint64_t ckey = 0;
    std::vector<kb::Digest> cached;
    bool from_cache = false;
    size_t n_ref = 0;
    for (size_t k = 0; k < n_opd; ++k)
        if (!c.opd[k].absent && c.opd[k].have_ref && written(k)) ++n_ref;
    if (cacheable) {
        PhaseTimer pt(&ph_, kb::PH_CACHE);
        ckey = kb::ref_cache_key(cache_.identity(), row->info->name, c, g, input_hash);
        from_cache = cache_.get(ckey, &cached) && cached.size() == n_ref;
    }

    std::vector<Bytes> own(n_opd);           /* written operands; laid weights */
    std::vector<RadTensor> t(n_opd);
    std::vector<DevBuf> dev(n_opd);
    DevBuf scr;
    RadStream stream = nullptr;
    std::vector<uint8_t> host_scratch;
    RadArgs a{};
    bool launched = false;
    bool wrote_input = false;        /* a host kernel changed an input: its timing would too */
    double checked_s = 0;            /* the checked launch's wall time */

    auto release = [&]() {
        if (stream) (void)rad_stream_sync(stream);
        for (auto& d : dev) if (d.base && d.owned) rad_dev_free(d.base, RAD_MEM_DEVICE);
        if (scr.base) rad_dev_free(scr.base, RAD_MEM_DEVICE);
    };

    if (!from_cache) {
        for (size_t k = 0; k < n_opd; ++k) {
            if (c.opd[k].absent || !in[k].bytes) { t[k] = RadTensor{}; continue; }
            t[k] = host_tensor(c.opd[k], in[k].bytes->data());
        }

        if (dom == RAD_DOMAIN_DEVICE) {
            PhaseTimer pt(&ph_, kb::PH_UPLOAD);
            if (!(stream = run_stream())) { ++st_.skipped; return; }
            for (size_t k = 0; k < n_opd; ++k) {
                if (c.opd[k].absent || !in[k].bytes || in[k].bytes->empty()) continue;
                const Bytes& src = *in[k].bytes;
                /* A weight is laid out, uploaded and described once per (library, kernel, weight
                 * identity, encoding, size). Shared weights keep their red zones: they are
                 * read-only, and the margins are still checked after every launch. */
                if (c.opd[k].role == RAD_OPD_WEIGHT) {
                    const std::string dk = kb::device_weight_key(lib, row->info->name, c.opd[k],
                                                                 k, src.size());
                    auto it = dcache_.find(dk);
                    if (it != dcache_.end()) {
                        dev[k] = it->second;
                        dev[k].owned = false;            /* the cache frees it, not this case */
                        t[k] = it->second.desc;
                        t[k].data = dev[k].p;
                        continue;
                    }
                    Bytes laid;
                    if (!lay_out(c, k, schema, run, in[k], &laid, &t[k])) {
                        ++st_.skipped; release(); return;
                    }
                    const Bytes& up = laid.empty() ? src : laid;
                    if (!guarded_alloc(dev[k], (int64_t)up.size(), stream)) {
                        nm_.skipped[c.op + "  (no device memory for this case)"]++;
                        ++st_.skipped; release(); return;
                    }
                    if (rad_memcpy_async(dev[k].p, up.data(), dev[k].bytes, stream) < 0 ||
                        rad_stream_sync(stream) < 0) {
                        ++st_.skipped; release(); return;
                    }
                    t[k].data = dev[k].p;
                    if (dcache_bytes_ + dev[k].bytes <= o_.max_case_bytes) {
                        dcache_bytes_ += dev[k].bytes;
                        dev[k].owned = false;
                        dev[k].desc  = t[k];
                        dcache_.emplace(dk, dev[k]);
                    }
                    continue;
                }
                if (!guarded_alloc(dev[k], (int64_t)src.size(), stream)) {
                    nm_.skipped[c.op + "  (no device memory for this case)"]++;
                    ++st_.skipped; release(); return;
                }
                if (rad_memcpy_async(dev[k].p, src.data(), dev[k].bytes, stream) < 0) {
                    ++st_.skipped; release(); return;
                }
                t[k].data = dev[k].p;
            }
            if (rad_stream_sync(stream) < 0) { ++st_.skipped; release(); return; }
        } else {
            /* Host: weights laid out in place, written operands copied, inputs shared. */
            for (size_t k = 0; k < n_opd; ++k) {
                if (c.opd[k].absent || !in[k].bytes) continue;
                if (c.opd[k].role == RAD_OPD_WEIGHT) {
                    Bytes laid;
                    if (!lay_out(c, k, schema, run, in[k], &laid, &t[k])) {
                        ++st_.skipped; return;
                    }
                    if (!laid.empty()) { own[k].swap(laid); t[k].data = own[k].data(); }
                    continue;
                }
                if (written(k)) {
                    own[k] = *in[k].bytes;
                    t[k].data = own[k].data();
                }
            }
        }

        a = case_args(t, g);
        if (row->info->scratch) {
            const int64_t need = row->info->scratch(&a);
            if (need > 0) {
                if (dom == RAD_DOMAIN_DEVICE) {
                    if (!guarded_alloc(scr, need, stream)) { ++st_.skipped; release(); return; }
                    rad_memset_async(scr.p, 0, need, stream);
                    a.scratch = scr.p;
                } else {
                    host_scratch.assign((size_t)need, 0);
                    a.scratch = host_scratch.data();
                }
                a.scratch_bytes = need;
            }
        }

        int s, sync = RAD_OK;
        {
            const auto l0 = std::chrono::steady_clock::now();
            s = row->info->launch(&a, dom == RAD_DOMAIN_DEVICE ? stream : nullptr);
            sync = (s >= 0 && dom == RAD_DOMAIN_DEVICE) ? rad_stream_sync(stream) : RAD_OK;
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                             l0).count();
            ph_.add(kb::PH_LAUNCH, sec);
            ph_.launch_by_lib[lib] += sec;
            checked_s = sec;
        }
        if (s < 0 || sync < 0) {
            /* A kernel refusing a geometry its own constraints accepted is a bug in the row
             * (spec §17). */
            std::printf("  %-22s %-44s FAIL  %s/%s refused a geometry its constraints accept: %s\n",
                        c.op.c_str(), g.str().c_str(), lib.c_str(), row->info->name,
                        rad_strerror(s < 0 ? s : sync));
            ++st_.failed;
            cov_[fi][lib].reached.insert(row->info->name);
            ran_keys_.insert(run.cap);
            release();
            return;
        }
        launched = true;

        /* Outputs are read back here; the timing pass runs later on the operands it is handed
         * and restores in-place ones itself. A timing-only run reads nothing back. */
        if (dom == RAD_DOMAIN_DEVICE && !o_.timing_only) {
            PhaseTimer pt(&ph_, kb::PH_READBACK);
            for (size_t k = 0; k < n_opd; ++k) {
                if (!dev[k].on || !written(k)) continue;
                own[k].assign((size_t)dev[k].bytes, 0);
                rad_memcpy_async(own[k].data(), dev[k].p, dev[k].bytes, stream);
            }
            rad_stream_sync(stream);
            check_margins(dev, scr, schema, c.op, lib + "/" + row->info->name, g, stream,
                          &nm_.overruns, &st_);
        }
        /* A host kernel was handed the shared inputs; one that changed any of them is wrong, and
         * would hand every later library a different case. */
        if (dom == RAD_DOMAIN_HOST && !o_.timing_only) {
            PhaseTimer pt(&ph_, kb::PH_HASH);
            for (size_t k = 0; k < n_opd; ++k) {
                if (c.opd[k].absent || !in[k].bytes || written(k) || !in[k].fp) continue;
                if (c.opd[k].role == RAD_OPD_WEIGHT && !own[k].empty()) continue;   /* laid copy */
                if (kb::fingerprint(in[k].bytes->data(), in[k].bytes->size()) == in[k].fp)
                    continue;
                nm_.written[c.op + "  (" + lib + "/" + row->info->name + " wrote into '" +
                            name_of(k) + "')"]++;
                ++st_.input_written;
                wrote_input = true;
                std::printf("  %-22s %-44s FAIL  %s/%s wrote into its input '%s'\n",
                            c.op.c_str(), g.str().c_str(), lib.c_str(), row->info->name,
                            name_of(k));
            }
        }
    }
    cov_[fi][lib].reached.insert(row->info->name);
    ran_keys_.insert(run.cap);

    /* ---------------------------------------------------------------- against the recording */
    double worst_rel = -1, worst_abs = 0, worst_max_rel = 0, worst_tol = 0;
    bool worst_bad = false, all_ok = true, all_vacuous = true;
    int64_t nonfinite_gap = 0, id_wrong = 0, ties = 0;
    int n_out = 0;
    std::string per_opd;
    std::vector<kb::Digest> digests;
    size_t ci = 0;
    for (size_t k = 0; k < n_opd && !o_.timing_only; ++k) {
        const KfOperand& d = c.opd[k];
        if (d.absent || !d.have_ref || !written(k)) continue;
        const uint64_t pos_seed = kf_output_seed(c.seed, (int)k);
        kb::Digest mine;
        if (from_cache) {
            mine = cached[ci++];
        } else {
            PhaseTimer pt(&ph_, kb::PH_COMPARE);
            std::vector<float> got;
            rad_widen(own[k].data(), d.dtype, d.ref_n, &got);
            if ((int64_t)got.size() < d.ref_n) continue;
            kb::digest_of(got.data(), d.ref_n, pos_seed, !d.sketch.empty() || cacheable, &mine);
        }
        if (cacheable && !from_cache) digests.push_back(mine);

        PhaseTimer pt(&ph_, kb::PH_COMPARE);
        const char* nm = name_of(k);
        const double tk = o_.tol_override >= 0 ? o_.tol_override
                                               : kb::tolerance_for(c.op, g, nm).rel_l2;
        /* Where the op lets a tie go either way, the ids are judged by the values they index. */
        kb::TieView tv;
        const kb::TieView* tie = nullptr;
        if (const kb::TieRule* rule = kb::tie_rule(c.op, nm)) {
            for (size_t j = 0; j < n_opd; ++j) {
                if (std::strcmp(name_of(j), rule->input) != 0 || !in[j].bytes) continue;
                const KfOperand& id = c.opd[j];
                if (id.shape.size() < 2) break;
                tv.in = in[j].bytes->data();
                tv.in_dtype = id.dtype;
                tv.in_rows = id.shape[0];
                tv.in_width = rad_numel(id.shape) / id.shape[0];
                tv.out_width = d.shape.empty() || d.shape[0] <= 0 ? 1
                                                                   : rad_numel(d.shape) / d.shape[0];
                tv.lane_stride = rule->lane_stride;
                long long off = 0;
                if (g.get_i("vocab_off", &off)) tv.id_base = off;
                tie = &tv;
                break;
            }
        }
        /* The digest above is computed with the sketch only where the recording has one. */
        kb::Digest cmp = mine;
        if (d.sketch.empty()) cmp.sketch.clear();
        const kb::OpdVerdict v = kb::judge_operand(d, cmp, pos_seed, tk, exact, tie);
        per_opd += fmt("%s%s=%.3e", per_opd.empty() ? "" : " ", nm, v.rel);
        if (v.index && (v.index_mismatch || v.ties))
            per_opd += fmt("(%lld id(s) wrong, %lld tie(s))", (long long)v.index_mismatch,
                           (long long)v.ties);
        nonfinite_gap += v.nonfinite_gap;
        id_wrong += v.index_mismatch;
        ties += v.ties;
        all_ok = all_ok && !v.bad;
        all_vacuous = all_vacuous && v.vacuous;
        const double tshow = exact && !v.index ? 0.0 : tk;
        if (kb::takes_worst(v.rel, v.bad, worst_rel, worst_bad)) {
            worst_rel = v.rel; worst_abs = v.max_abs; worst_max_rel = v.max_rel; worst_tol = tshow;
            worst_bad = v.bad;
        }
        ++n_out;
    }
    if (cacheable && !from_cache && launched && digests.size() == n_ref) {
        PhaseTimer pt(&ph_, kb::PH_CACHE);
        cache_.put(ckey, digests);
    }
    if (from_cache) ++st_.cached;
    if (n_out == 0 && !o_.timing_only) { ++st_.skipped; release(); return; }

    const double tol = worst_tol;
    const bool ok = all_ok;
    const bool vacuous = !o_.timing_only && ok && all_vacuous;
    if (o_.timing_only) {
        ++st_.timed;
    } else if (vacuous) {
        ++st_.vacuous;
        bool undrawn = false;
        for (const KfOperand& d : c.opd) undrawn = undrawn || d.undrawn;
        nm_.vacuous[c.op + "  (" + lib + "/" + row->info->name +
                    (undrawn ? ": recorded before weight tables were drawn -- record the fixture "
                               "again)"
                             : ")")]++;
    } else {
        ++st_.checked;
        st_.worst = kb::worst_of(st_.worst, worst_rel);
        if (ok) ++st_.passed; else ++st_.failed;
    }

    /* ---------------------------------------------------------------- and the clock
     *
     * A device case is handed to the timer with its operands, which then belong to it; it is
     * timed in the timing pass (kbench_timer.h). A host case is timed here. */
    int64_t timing_id = -1;
    kb::TimedCase host_case;
    bool host_timed = false;
    if (o_.bench && launched && !wrote_input && (o_.timing_only || (ok && !vacuous))) {
        kb::TimedCase tc;
        tc.info = row->info;
        tc.g = g;
        tc.seed = c.seed;
        tc.what = c.op + " " + row->info->name + " " + g.str();
        for (size_t k = 0; k < n_opd; ++k) {
            kb::TimedOperand d;
            d.t = t[k];
            d.role = role_of(k);
            if (!c.opd[k].absent && t[k].data && in[k].bytes) {
                d.bytes = dom == RAD_DOMAIN_DEVICE ? dev[k].bytes
                                                   : (int64_t)(own[k].empty() ? in[k].bytes->size()
                                                                              : own[k].size());
                if (d.role == RAD_OPD_INOUT) d.pristine = in[k].bytes->data();
                if (d.role == RAD_OPD_IN)
                    kb::timing_input(c.opd[k], g, kf_operand_seed(c.seed, (int)k) ^ kTimingSalt,
                                     &d.variant);
            }
            tc.opd.push_back(std::move(d));
        }
        if (dom == RAD_DOMAIN_DEVICE) {
            tc.scratch = scr.p;
            tc.scratch_bytes = scr.bytes;
            for (DevBuf& d : dev) {
                if (!d.base || !d.owned) continue;
                tc.frees.push_back(d.base);
                tc.held += d.bytes + 2 * RED_BYTES;
                d.owned = false;
            }
            if (scr.base) {
                tc.frees.push_back(scr.base);
                tc.held += scr.bytes + 2 * RED_BYTES;
                scr.base = nullptr;
            }
            if (!timer_) timer_ = std::make_unique<kb::DeviceTimer>(o_.timer, stream, &ph_);
            timing_id = timer_->add(std::move(tc));
        } else {
            tc.scratch = host_scratch.empty() ? nullptr : host_scratch.data();
            tc.scratch_bytes = (int64_t)host_scratch.size();
            host_case = std::move(tc);
            host_timed = true;
        }
    }
    release();

    Row rw;
    rw.op      = c.op;
    rw.kernel  = row->info->name;
    rw.plugin  = lib;
    rw.geom    = g.str();
    rw.domain  = domain_name(dom);
    rw.verdict = o_.timing_only ? "timed" : vacuous ? "vacuous" : ok ? "ok" : "**FAIL**";
    rw.rel_l2  = worst_rel;
    rw.tol     = tol;
    rw.timing_id = timing_id;
    rw.band    = c.band;
    for (const KfOperand& d : c.opd)
        if (!d.absent) rw.bytes += rad_dtype_bytes(d.dtype, rad_numel(d.shape));
    rows_.push_back(std::move(rw));

    if (!o_.timing_only && (!ok || o_.verbose))
        std::printf("  %-22s %-44s %-8s %-28s rel=%.3e max_abs=%.3e max_rel=%.3e tol=%.1e %s%s%s\n",
                    c.op.c_str(), g.str().c_str(), domain_name(dom), row->info->name,
                    worst_rel, worst_abs, worst_max_rel, tol,
                    ok ? (vacuous ? "vacuous" : "ok") : "FAIL", exact ? " (exact)" : "",
                    from_cache ? " (cached)" : "");
    if (!o_.timing_only && !ok && n_out > 1) std::printf("      per operand: %s\n", per_opd.c_str());
    if (!o_.timing_only && !ok && nonfinite_gap)
        std::printf("      %lld element(s) are finite on one side and not the other (a failure "
                    "at any tolerance)\n", (long long)nonfinite_gap);
    if (!o_.timing_only && !ok && id_wrong)
        std::printf("      %lld id(s) differ from the reference's%s\n", (long long)id_wrong,
                    ties ? fmt(" (%lld more are ties)", (long long)ties).c_str() : "");
    if (host_timed) {
        PhaseTimer pt(&ph_, kb::PH_TIMING);
        apply_timing(rows_.back(), kb::time_host(host_case, o_.timer, host_footprint_, checked_s));
    }
}

/* The reference's answer to `base`, written into `out` (a copy of `base` with every written
 * operand's reference replaced): the case drawn as a replay draws it, the reference's host row
 * launched, each output reduced with its sketch. RAD_OK, or an error with `why`. */
int Replay::record_case(const KfCase& base, KfCase* out, std::string* why) {
    *out = base;
    const RadOpSchema* schema = reg_->schema(base.op);
    if (!schema) { *why = "no plugin declares a schema for it"; return RAD_E_NOSCHEMA; }
    if ((int)base.opd.size() != schema->n_operands) {
        *why = fmt("the case has %zu operand(s), the schema %d", base.opd.size(),
                   schema->n_operands);
        return RAD_E_SCHEMA;
    }
    const Geometry g = case_geometry(base);
    const Registry::RowFilter accept = weight_filter(base, schema);
    std::vector<Run> rr = rows_for_case(*reg_, ref_name_, base.op, g, RAD_DOMAIN_HOST, accept,
                                        false);
    if (rr.empty()) { *why = ref_name_ + " has no host row for it"; return RAD_E_NOKERNEL; }
    const Run& run = rr[0];
    if (kb::case_bytes(base.opd) < 0 || kb::case_bytes(base.opd) > o_.max_case_bytes) {
        *why = "over the --max-bytes budget";
        return RAD_E_NOMEM;
    }
    std::vector<Drawn> in;
    if (!draw_case(base, &in, why)) return RAD_E_SHAPE;

    const size_t n = base.opd.size();
    std::vector<Bytes> own(n);
    std::vector<RadTensor> t(n);
    for (size_t k = 0; k < n; ++k) {
        const KfOperand& d = base.opd[k];
        if (d.absent || !in[k].bytes) continue;
        t[k] = host_tensor(d, in[k].bytes->data());
        const int role = schema->operands[k].role;
        if (role == RAD_OPD_OUT || role == RAD_OPD_INOUT) {
            own[k] = *in[k].bytes;
            t[k].data = own[k].data();
        }
    }
    RadArgs a = case_args(t, run.g);
    std::vector<uint8_t> scratch;
    if (run.row->info->scratch) {
        const int64_t need = run.row->info->scratch(&a);
        if (need > 0) {
            scratch.assign((size_t)need, 0);
            a.scratch = scratch.data();
            a.scratch_bytes = need;
        }
    }
    {
        const auto l0 = std::chrono::steady_clock::now();
        const int s = run.row->info->launch(&a, nullptr);
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count();
        ph_.add(kb::PH_LAUNCH, sec);
        ph_.launch_by_lib[ref_name_] += sec;
        if (s < 0) {
            *why = std::string(run.row->info->name) + " refused it: " + rad_strerror(s);
            return s;
        }
    }
    PhaseTimer pt(&ph_, kb::PH_COMPARE);
    for (size_t k = 0; k < n; ++k) {
        KfOperand& d = out->opd[k];
        const int role = schema->operands[k].role;
        d.have_ref = false;
        d.sample.clear();
        d.sketch.clear();
        if (d.absent || (role != RAD_OPD_OUT && role != RAD_OPD_INOUT)) continue;
        const int64_t numel = rad_numel(d.shape);
        std::vector<float> v;
        rad_widen(own[k].data(), d.dtype, numel, &v);
        kb::Digest dg;
        kb::digest_of(v.data(), numel, kf_output_seed(base.seed, (int)k), true, &dg);
        d.have_ref = true;
        d.ref_n = numel;
        d.ref_sumsq = dg.sumsq;
        d.ref_nonfinite = dg.nonfinite;
        d.sample = std::move(dg.sample);
        d.sketch = std::move(dg.sketch);
    }
    return RAD_OK;
}

/* The recording date, shown in reports so a stale fixture is visible. */
std::string today() {
    char day[32] = { 0 };
    const std::time_t now = std::time(nullptr);
    std::strftime(day, sizeof day, "%Y-%m-%d", std::gmtime(&now));
    return day;
}

int Replay::rerecord() {
    Kfixture out;
    out.oracle = ref_name_;
    out.recorded = today();
    int dropped = 0;
    for (size_t fi = 0; fi < fx_.size(); ++fi) {
        out.source += (out.source.empty() ? "" : " + ") + fx_[fi].source;
        for (const KfCase& c : fx_[fi].cases) {
            if (!o_.op_filters.empty()) {
                bool hit = false;
                for (const std::string& f : o_.op_filters)
                    if (c.op.rfind(f, 0) == 0) { hit = true; break; }
                if (!hit) continue;
            }
            KfCase rc;
            std::string why;
            if (record_case(c, &rc, &why) != RAD_OK) {
                nm_.skipped[c.op + "  (not recorded: " + why + ")"]++;
                ++dropped;
                continue;
            }
            out.cases.push_back(std::move(rc));
        }
    }
    for (const auto& [what, n] : nm_.skipped) std::printf("  %-70s x%d\n", what.c_str(), n);
    const int rc = out.save(o_.rerecord_path);
    if (rc < 0) {
        RAD_ERR("--rerecord %s: %s", o_.rerecord_path.c_str(), rad_strerror(rc));
        return 1;
    }
    std::printf("recorded %zu case(s) to %s with %s's answers (%d dropped)\n", out.cases.size(),
                o_.rerecord_path.c_str(), ref_name_.c_str(), dropped);
    std::printf("%s\n", kb::phase_line(ph_, std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - t0_)
                                                .count()).c_str());
    return 0;
}

/* ================================================================== the row-driven fixture
 *
 * A recording reaches only the rows its container's geometry selects; a row behind another band,
 * tuned point or dtype never runs. For every row of every library under test, this moves an
 * existing case of the row's op into the row's constraints (kb::satisfy_constraints) and records
 * the reference's answer there. Only cases whose operand shapes the reference can describe from the
 * parameters alone are moved. A weight of another shape takes the encoding its own row's
 * description implies (derive_enc), where that is readable; the rest are listed with the reason.
 * Replay the result with --all-rows, since a row
 * a higher-priority row shadows is reached only that way. No container and no card are needed. */
/* The affine encoding a row's own description of a weight implies, where the base case's is of
 * another shape than the row states: codes at the weight's declared dtype, a scale at the next
 * operand's, one block per `group` of K. That is what a container declares for the formats the
 * geometry alone states (libref/ref_shapes.cpp, gemm_q_shape: i8a8 is [N][K] codes against [N][K/
 * group] bf16), and the plane extents follow from the weight's [rows, cols] as a container's do
 * (rad_enc_plane_dims). Nothing is invented beyond that: the row states its own operands, so a
 * shape or dtype this derivation cannot read leaves the case out, and a case whose bytes the row
 * then refuses is reported with the row's reason instead of being recorded. */
bool derive_enc(std::vector<KfOperand>* opd, size_t k, const Geometry& g) {
    KfOperand& d = (*opd)[k];
    if (d.role != RAD_OPD_WEIGHT || d.shape.size() != 2 || d.absent) return false;
    long long group = 0;
    if (!g.get_i("group", &group) || group <= 0) return false;
    if (k + 1 >= opd->size()) return false;
    const KfOperand& s = (*opd)[k + 1];
    if (s.role != RAD_OPD_WEIGHT || s.absent) return false;
    const int64_t rows = d.shape[0], cols = d.shape[1];
    if (rows <= 0 || cols <= 0) return false;
    /* The scale is the next operand: one value per (row, group of K), and nothing else. */
    if (s.dtype == RAD_I8 || s.shape.size() != 2 || s.shape[0] != rows ||
        s.shape[1] != (cols + group - 1) / group)
        return false;
    d.has_enc = true;
    d.enc = rad_enc_affine(d.dtype, s.dtype, 1, group);
    d.sel = { 0, 1 };
    d.sel_rows.assign(2, 0);
    d.sel_cols.assign(2, 0);
    for (size_t j = 0; j < 2; ++j)
        rad_enc_plane_dims(&d.enc.plane[d.sel[j]], rows, cols, &d.sel_rows[j], &d.sel_cols[j]);
    return rad_enc_valid(&d.enc) != 0;
}

int Replay::record_rows() {
    Kfixture out;
    out.oracle = ref_name_;
    out.recorded = today();
    for (size_t fi = 0; fi < fx_.size(); ++fi)
        out.source += (out.source.empty() ? "rows from " : " + ") + fx_[fi].source;
    std::set<std::string> seen;
    int rows_total = 0, rows_recorded = 0;
    std::map<std::string, int> why_not;
    for (const KernelRow& r : reg_->rows()) {
        if (!r.info || !r.info->launch || !r.info->op) continue;
        if (r.plugin == ref_name_) continue;
        if (std::find(libs_.begin(), libs_.end(), r.plugin) == libs_.end()) continue;
        if (!o_.op_filters.empty()) {
            bool hit = false;
            for (const std::string& f : o_.op_filters)
                if (std::string(r.info->op).rfind(f, 0) == 0) { hit = true; break; }
            if (!hit) continue;
        }
        const RadOpSchema* schema = reg_->schema(r.info->op);
        if (!schema) continue;
        ++rows_total;
        std::string last = "no recorded case of its op";
        bool done = false;
        for (size_t fi = 0; fi < fx_.size() && !done; ++fi) {
            for (const KfCase& base : fx_[fi].cases) {
                if (done) break;
                if (base.op != r.info->op || (int)base.opd.size() != schema->n_operands) continue;
                Geometry g = case_geometry(base);
                if (!kb::satisfy_constraints(r.info, &g)) {
                    last = "no case of its op can be moved into its constraints";
                    continue;
                }
                std::vector<Run> rr = rows_for_case(*reg_, ref_name_, base.op, g,
                                                    RAD_DOMAIN_HOST, weight_filter(base, schema),
                                                    false);
                if (rr.empty()) { last = ref_name_ + " has no row there"; continue; }
                std::vector<OpdShape> sh;
                if (rad_operand_shapes_hook(rr[0].row->info, g, schema->n_operands, &sh) < 0 ||
                    sh.size() != base.opd.size()) {
                    last = ref_name_ + " cannot describe its operands from the parameters alone";
                    continue;
                }
                KfCase c;
                c.op = base.op;
                c.band = "row " + std::string(r.info->name);
                c.m = base.m;
                for (const char* key : { "M", "q_len" }) {
                    long long v = 0;
                    if (g.get_i(key, &v)) { c.m = v; break; }
                }
                bool weights_kept = true;
                c.opd.resize(sh.size());
                for (size_t k = 0; k < sh.size(); ++k) {
                    KfOperand& d = c.opd[k];
                    const KfOperand& b = base.opd[k];
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
                /* Weights second: derive_enc reads the scale operand beside the codes, which the
                 * loop above has only just filled. */
                for (size_t k = 0; k < sh.size(); ++k) {
                    KfOperand& d = c.opd[k];
                    if (d.role != RAD_OPD_WEIGHT && d.role != RAD_OPD_WTAB) continue;
                    const KfOperand& b = base.opd[k];
                    if (d.absent == b.absent && d.dtype == b.dtype && d.shape == b.shape) {
                        d.has_enc = b.has_enc;
                        d.enc = b.enc;
                        d.sel = b.sel;
                        d.sel_rows = b.sel_rows;
                        d.sel_cols = b.sel_cols;
                        continue;
                    }
                    if (!derive_enc(&c.opd, k, g)) {
                        weights_kept = false;
                        break;
                    }
                    /* Its scale plane is the next operand, which keeps its plain draw: a tensor
                     * of the declared [N][K/group] reads as those scale values. */
                    ++k;
                }
                if (!weights_kept) {
                    last = "its geometry changes a weight's shape";
                    continue;
                }
                for (int i = 0; i < g.n_params(); ++i) {
                    const RadParam& p = g.params()[i];
                    if (!p.key) continue;
                    KfParam kp;
                    kp.key = p.key;
                    kp.kind = p.kind;
                    if (p.kind == RAD_P_INT)      kp.i = p.ival;
                    else if (p.kind == RAD_P_F64) kp.d = p.dval;
                    else if (p.kind == RAD_P_STR) kp.s = p.sval ? p.sval : "";
                    else continue;
                    c.params.push_back(std::move(kp));
                }
                c.seed = rad_case_seed(base.seed, r.info->name, 0, r.info->domain, 0);
                const std::string key = kb::cap_key(c, "", "", 0, g);
                if (seen.count(key)) { done = true; ++rows_recorded; break; }
                KfCase rc;
                std::string why;
                if (record_case(c, &rc, &why) != RAD_OK) { last = why; continue; }
                seen.insert(key);
                out.cases.push_back(std::move(rc));
                ++rows_recorded;
                done = true;
            }
        }
        if (!done) why_not[std::string(r.plugin) + "/" + r.info->name + "  (" + last + ")"]++;
    }
    if (!why_not.empty()) {
        std::printf("rows without a case\n");
        for (const auto& [what, n] : why_not) std::printf("  %s\n", what.c_str());
    }
    const int rc = out.save(o_.record_rows_path);
    if (rc < 0) {
        RAD_ERR("--record-rows %s: %s", o_.record_rows_path.c_str(), rad_strerror(rc));
        return 1;
    }
    std::printf("recorded %zu case(s) for %d of %d row(s) to %s; replay it with --all-rows\n",
                out.cases.size(), rows_recorded, rows_total, o_.record_rows_path.c_str());
    return 0;
}

int Replay::finish() {
    /* The timing pass, before the weight cache it borrows from is freed. */
    BenchInfo bench;
    if (o_.bench) bench = finish_timing(timer_.get(), rows_, host_footprint_, o_.timer.footprint >= 0);
    timer_.reset();
    for (auto& [k2, d] : dcache_) if (d.base) rad_dev_free(d.base, RAD_MEM_DEVICE);
    dcache_.clear();

    print_summary(st_, o_.timing_only, nm_);
    print_row_coverage(census(*reg_, ref_name_), rows_);

    /* The ratchet: per fixture and library under test, the rows a replay reaches must not
     * shrink. Meaningless under an --op filter, which reaches fewer on purpose. */
    int lost = 0;
    if (!o_.coverage_dir.empty() && o_.op_filters.empty()) {
        for (size_t fi = 0; fi < fx_.size(); ++fi) {
            for (const std::string& lib : libs_) {
                const Coverage& cv = cov_[fi][lib];
                const std::string path =
                    o_.coverage_dir + "/" + kb::rows_file_name(o_.fixtures[fi], lib);
                bool exists = false;
                const std::set<std::string> listed = kb::read_rows_file(path, &exists);
                if (o_.coverage_write) {
                    std::set<std::string> keep = cv.reached;
                    for (const std::string& r : listed) if (cv.unknown.count(r)) keep.insert(r);
                    if (kb::write_rows_file(path, keep) == RAD_OK)
                        std::printf("coverage: wrote %zu row(s) to %s\n", keep.size(),
                                    path.c_str());
                    else
                        RAD_ERR("--coverage-write %s: %s", path.c_str(), std::strerror(errno));
                    continue;
                }
                if (!exists) {
                    std::printf("coverage: no ratchet for %s on %s (%s; --coverage-write "
                                "creates it)\n", lib.c_str(), o_.fixtures[fi].c_str(),
                                path.c_str());
                    continue;
                }
                const kb::Ratchet r = kb::ratchet(listed, cv.reached, cv.unknown);
                for (const std::string& s : r.lost)
                    std::printf("coverage: %s/%s is no longer reached by %s\n", lib.c_str(),
                                s.c_str(), o_.fixtures[fi].c_str());
                if (!r.gained.empty())
                    std::printf("coverage: %zu row(s) of %s newly reached by %s (add them with "
                                "--coverage-write)\n", r.gained.size(), lib.c_str(),
                                o_.fixtures[fi].c_str());
                lost += (int)r.lost.size();
            }
        }
    }

    if (!o_.report_path.empty()) {
        PhaseTimer pt(&ph_, kb::PH_REPORT);
        std::vector<std::string> graph_ops;
        std::string prov;
        for (size_t fi = 0; fi < fx_.size(); ++fi) {
            for (const KfCase& c : fx_[fi].cases) graph_ops.push_back(c.op);
            prov += (prov.empty() ? "fixture: `" : "; `") + o_.fixtures[fi] + "`, " +
                    fmt("%zu", fx_[fi].cases.size()) + " case(s) recorded " +
                    (fx_[fi].recorded.empty() ? "?" : fx_[fi].recorded) + " from `" +
                    (fx_[fi].source.empty() ? "?" : fx_[fi].source) + "`";
        }
        std::sort(graph_ops.begin(), graph_ops.end());
        graph_ops.erase(std::unique(graph_ops.begin(), graph_ops.end()), graph_ops.end());
        std::map<std::string, int> gaps = nm_.skipped;
        for (const auto& [w, n] : nm_.unserved) gaps["unserved: " + w] += n;
        for (const auto& [w, n] : nm_.stale) gaps["stale: " + w] += n;
        for (const auto& [w, n] : nm_.vacuous) gaps["vacuous: " + w] += n;
        if (write_report(o_.report_path, rows_, gaps, nm_.fuse_skipped, nm_.diverged,
                         nm_.overruns, o_.analysis_path, prov, 0, 0, st_, ref_name_,
                         plugins_.libraries(), "", graph_ops, census(*reg_, ref_name_),
                         o_.bench ? &bench : nullptr) >= 0)
            std::printf("report written to %s (%zu row(s))\n", o_.report_path.c_str(),
                        rows_.size());
    }
    std::printf("%s\n", kb::phase_line(ph_, std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - t0_)
                                                .count()).c_str());
    if (lost) return 1;
    return kb::exit_code(st_, o_.strict);
}

void usage() {
    std::printf(
        "rad-kbench -- correctness, completeness and speed of every kernel library, in one report\n"
        "\n"
        "usage: rad-kbench [options]                     replay the recorded fixture (no model)\n"
        "       rad-kbench -m <model.rad> [options]      check against a real container\n"
        "       rad-kbench -m <model.rad> --record FILE  record a new fixture\n"
        "       rad-kbench -m <model.rad> --skeleton FILE  write a metadata-only copy to record from\n"
        "\n"
        "  With no -m it replays every kernel every loaded library supplies against the reference\n"
        "  library's recorded answers on seeded inputs, inside guarded allocations. No model and no\n"
        "  card. With -m it checks the same kernels against a real container's weights in the\n"
        "  layout the resolved kernel asked for, which also tests the artifact and the layout hooks.\n"
        "\n"
        "      --fixture FILE the recording to replay (default $RADIANCE_HOME/kernels.rkb);\n"
        "                     repeatable, and the report covers the union\n"
        "      --record FILE  with -m: write a fixture of the reference's answer at every point of\n"
        "                     every declared op the reference serves, whichever library resolved\n"
        "                     it. Weights are drawn from seeds, not read from the container, so any\n"
        "                     machine can reproduce the reference, and no card is needed.\n"
        "      --skeleton FILE  with -m: write a sparse copy of the container, its tables intact and\n"
        "                     its data a hole. Recording reads only the tables, so a skeleton of a\n"
        "                     30 GB model records the same fixture in a few megabytes.\n"
        "      --rerecord FILE  replay mode: write the fixtures' cases again with the reference's\n"
        "                     answers, in the current format\n"
        "      --record-rows FILE  replay mode: move the fixtures' cases into the constraints of\n"
        "                     every row of the libraries under test and record the reference there,\n"
        "                     a fixture that reaches rows no container selects (replay it with\n"
        "                     --all-rows)\n"
        "      --analysis FILE  a markdown fragment from tools/kbench-analysis.sh, spliced into\n"
        "                     the report as its static-and-dynamic-analysis section\n"
        "  -m, --model FILE   the .rad. Weight operands are read from it, in the layout the\n"
        "                     resolved kernel asked for.\n"
        "      --max-ctx N    the deepest context this run considers: declare runs at it and the\n"
        "                     attention sweep builds to it. Default: the checkpoint's training\n"
        "                     context for declare, 16384 for the sweep. An architecture whose\n"
        "                     graph depends on the bound (Qwen4-Exp sizes its QSA buffers by it)\n"
        "                     is checked at this bound. Attention is swept over context depth as\n"
        "                     well as query length, which covers the block table, the KV split\n"
        "                     and the window.\n"
        "      --home DIR     $RADIANCE_HOME\n"
        "      --kernels A:B  the kernel libraries to load, first wins; default every library found\n"
        "      --op NAME      check only this op (repeatable prefix match)\n"
        "      --ref NAME     the oracle plugin's name (default: the library that declares itself a\n"
        "                     reference implementation)\n"
        "      --no-ref       replay: leave the reference out of the libraries under test. Its\n"
        "                     answers are what the fixture holds; replaying it is its own check\n"
        "                     (the kbench_ref test), at tolerance zero.\n"
        "      --strict       replay: a stale fixture fails the run -- a case whose operand count\n"
        "                     or schema moved, which the reference no longer serves, or which it\n"
        "                     now describes differently from its recording\n"
        "      --all-rows     replay: run every row of a library whose constraints hold, not only\n"
        "                     the one selection picks\n"
        "      --coverage DIR replay: fail when a row a fixture reached (DIR/kbench_rows_<fixture>_\n"
        "                     <library>.txt) is no longer reached; rows a memory skip kept from\n"
        "                     running are not counted lost\n"
        "      --coverage-write  with --coverage: write those files from this run\n"
        "      --no-cache     replay: run the reference instead of reusing its cached answers. The\n"
        "                     cache is keyed by the reference's code, its inputs and the case, and\n"
        "                     the comparison runs either way.\n"
        "      --cache-dir DIR  the reference cache (default $RADIANCE_HOME/cache/kbench)\n"
        "      --seed N       base seed; every case derives its own from it, so changing a\n"
        "                     filter does not change any other case's inputs\n"
        "      --tol X        override every tolerance with X (for bisecting, not for passing)\n"
        "      --max-tok N    the max_tok declare is run at (default 512; small on purpose)\n"
        "      --mm-max-patches N  the encoder pass declare is run at, so a media tower's\n"
        "                     kernels are in the graph: default 2048 (rounded to whole output\n"
        "                     rows) when the container carries an encoder, 0 to leave it out\n"
        "      --no-fusion    skip the fusion oracle: a kernel that declares the chain it\n"
        "                     replaces is otherwise run against that chain, out of its own\n"
        "                     plugin, and the outputs compared byte for byte\n"
        "      --max-cases N  runs per distinct (op, kernel, library, domain, geometry, operand\n"
        "                     shapes); 0 means every one (default 4). A 62-layer model presents the\n"
        "                     same kernel at the same shape sixty times and only the weights differ.\n"
        "      --max-bytes MIB  skip a case whose operands would need more than this, by name\n"
        "                     (default: half the memory available at start, at most 8192). One\n"
        "                     case holds every operand as drawn, again as the kernel's copy and\n"
        "                     again on the device; without a bound a small machine dies in the OOM\n"
        "                     killer without naming the op. Also bounds the weight caches.\n"
        "      --bench        time each case that passes, on the bytes it was checked on. Device\n"
        "                     cases are timed after the checks, back to back after one sustained\n"
        "                     warm-up: device timestamps around a batch of launches queued behind\n"
        "                     a gate, weights read cold, in-place operands restored, until the\n"
        "                     median is known to 1%%. Host cases are timed where they are checked.\n"
        "                     The report's speed section says how, with the calibration.\n"
        "      --timing-only  time every device kernel and check nothing: no reference run, no\n"
        "                     comparison, no margins, no fusion chain, host rows skipped. For\n"
        "                     comparing builds a full run has already checked. Implies --bench.\n"
        "      --bench-flush MIB  the rotation footprint: each launch reads its weights (and\n"
        "                     inputs of MIB/32 or more) from one of a set of copies spanning MIB,\n"
        "                     so from memory, as the engine streams them. Default: 256 on a card\n"
        "                     (twice its L2 plus last-level cache, 8 + 64 MiB on the R9700, with\n"
        "                     room), twice the last-level cache and at least 256 for host rows.\n"
        "                     0 times warm: every launch rereads one set of operands. Implies\n"
        "                     --bench.\n"
        "      --bench-iters N  launches in every timed batch; 0 (default) sizes each batch to a\n"
        "                     millisecond and fifty launch floors\n"
        "      --bench-reps N   the most batches a case takes (default 20); fewer when the\n"
        "                     median's 95%% interval is within 1%% after six\n"
        "      --bench-warm-ms N  sustained load before each timing pass, in ms (default 2000;\n"
        "                     0 for none). The cards idle in a low-power state and the clock ramp\n"
        "                     is longer than a short batch.\n"
        "      --report FILE  write the markdown report: a summary of every loaded library, the\n"
        "                     op-by-library completeness matrix, every case and its error, the\n"
        "                     speed table, and every skip named\n"
        "      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at, as it is on the\n"
        "                     engine: an architecture whose attention kernels serve only one width\n"
        "                     declares no attention at the other. Default bf16\n"
        "      --num-speculative-tokens N\n"
        "                     draft depth to declare at (default: the container's own). It puts\n"
        "                     the drafter's kernels in the checked graph.\n"
        "  -v, --verbose      print every case, not only the failures\n"
        "  -h, --help\n");
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string model, home = rad_home(), ref_name;
    /* --op is repeatable; an op matches if any entry is its prefix. Empty means every op. */
    std::vector<std::string> op_filters;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    uint64_t seed = 20260904;
    double   tol_override = -1;
    int64_t  max_tok = 512;
    /* Encoder pass size for declare, so a media tower's kernels are in the graph. -1 is auto: a
     * pass smaller than a serve's when the container carries an encoder (radiance.encoder). */
    int64_t  mm_max_patches = -1;
    /* --max-ctx (0 = not given) sets both the declare context and the attention sweep depth;
     * an architecture whose graph depends on the bound (Qwen4-Exp's QSA) cannot be declared at its
     * training context. sweep_max_ctx is also a memory budget, since a paged KV cache at that
     * depth is allocated per case (67 MB for the 27B at 16384). The sweep takes 2048 and 16384:
     * one past the window where the block table rebases, one well past it. */
    int64_t  max_ctx_flag = 0;
    int      kv_dtype = RAD_DT_INVALID;
    int64_t  sweep_max_ctx = 16384;
    /* --record writes a fixture; with no model and no --record the tool replays one. */
    std::string record_path, skeleton_path;
    std::vector<std::string> fixture_paths;
    /* Replay-only settings (ReplayOpts). */
    ReplayOpts ro;
    /* A markdown fragment from tools/kbench-analysis.sh, spliced into the report verbatim. */
    std::string analysis_path;
    /* Runs per distinct (op, kernel, geometry, tuned point). A 62-layer model repeats each with
     * only the weights differing, and every case costs a host oracle run (a 512 x 17408 x 5120
     * GEMM), so the default samples. 0 means every case, which is what to use before a release. */
    int      max_cases = 4;
    /* The most one CASE may allocate, counting every operand built for the kernel, for the oracle,
     * and a re-laid weight's third copy. Above every weight in the models this tree serves and
     * below the n-gram embedding table, which is 51.2 GB on its own.
     *
     * 0 means "ask the machine", and that is the default because this runs in ctest on an
     * arbitrary machine. The biggest case in the tree's own fixture needs 7.3 GiB by the
     * estimate above, and a fixed ceiling higher than what the machine has is not a bound at
     * all -- it is the OOM killer with extra steps, which is exit 137, two lines of log, and no
     * indication which op it died on. A machine with room gets the full 8 GiB ceiling. */
    int64_t  max_case_bytes = 0;
    /* On by default: a fusion claims to be substitutable for its chain. --no-fusion is for
     * bisecting. */
    bool     fuse_check = true;
    bool     bench = false;
    /* Time without checking, for comparing builds a full run has already checked; skips the host
     * reference, where most of a run's time goes. */
    bool     timing_only = false;
    /* The timing pass (kb::TimerOpts): --bench-flush, --bench-iters, --bench-reps,
     * --bench-warm-ms. */
    kb::TimerOpts topts;
    std::string report_path;
    int      n_spec = -1;   /* -1 = the container's own operating point */
    bool     verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* w) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "rad-kbench: %s needs a value\n", w);
                                 std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-m" || a == "--model") model = next("--model");
        else if (a == "--home")    home = next("--home");
        else if (a == "--kernels") hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--op")      op_filters.push_back(next("--op"));
        else if (a == "--ref")     ref_name = next("--ref");
        else if (a == "--seed")    seed = (uint64_t)std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--tol")     tol_override = std::strtod(next("--tol"), nullptr);
        else if (a == "--max-tok") max_tok = std::atoll(next("--max-tok"));
        else if (a == "--max-ctx") max_ctx_flag = std::atoll(next("--max-ctx"));
        else if (a == "--mm-max-patches") mm_max_patches = std::atoll(next("--mm-max-patches"));
        else if (a == "--kv-cache-dtype") {
            const std::string v = next("--kv-cache-dtype");
            if (v == "fp8")       kv_dtype = RAD_F8E4M3;
            else if (v != "bf16") { std::fprintf(stderr, "rad-kbench: --kv-cache-dtype takes bf16 "
                                                         "or fp8, not '%s'\n", v.c_str());
                                    return 2; }
        }
        else if (a == "--record")  record_path  = next("--record");
        else if (a == "--fixture") fixture_paths.push_back(next("--fixture"));
        else if (a == "--skeleton") skeleton_path = next("--skeleton");
        else if (a == "--rerecord") ro.rerecord_path = next("--rerecord");
        else if (a == "--record-rows") ro.record_rows_path = next("--record-rows");
        else if (a == "--strict")  ro.strict = true;
        else if (a == "--no-ref")  ro.no_ref = true;
        else if (a == "--no-cache") ro.no_cache = true;
        else if (a == "--cache-dir") ro.cache_dir = next("--cache-dir");
        else if (a == "--all-rows") ro.all_rows = true;
        else if (a == "--coverage") ro.coverage_dir = next("--coverage");
        else if (a == "--coverage-write") ro.coverage_write = true;
        else if (a == "--analysis") analysis_path = next("--analysis");
        else if (a == "--max-cases") max_cases = std::atoi(next("--max-cases"));
        else if (a == "--max-bytes")
            max_case_bytes = (int64_t)std::atoll(next("--max-bytes")) << 20;
        else if (a == "--no-fusion") fuse_check = false;
        else if (a == "--bench")   bench = true;
        else if (a == "--timing-only") timing_only = true;
        else if (a == "--bench-flush") {
            topts.footprint = std::max<int64_t>(0, (int64_t)std::atoll(next("--bench-flush"))) << 20;
            bench = true;
        }
        else if (a == "--bench-iters") topts.fixed = std::max(0, std::atoi(next("--bench-iters")));
        else if (a == "--bench-reps")  topts.max_reps = std::max(1, std::atoi(next("--bench-reps")));
        else if (a == "--bench-warm-ms") topts.warm_ms = std::max(0, std::atoi(next("--bench-warm-ms")));
        else if (a == "--report")  report_path = next("--report");
        else if (a == "--num-speculative-tokens") n_spec = std::atoi(next("--num-speculative-tokens"));
        else if (a == "-v" || a == "--verbose") { verbose = true; log_set_level(Log::Debug); }
        else { std::fprintf(stderr, "rad-kbench: unrecognised argument '%s'. Try --help.\n",
                            a.c_str()); return 2; }
    }

    if (timing_only) {
        if (!record_path.empty()) {
            std::fprintf(stderr, "rad-kbench: --timing-only runs no reference, so it has nothing to "
                                 "record; drop one of --timing-only and --record\n");
            return 2;
        }
        bench = true;          /* it is the clock or nothing */
        fuse_check = false;    /* the fusion oracle is a check too */
    }

    /* Default case budget: half the memory available now (a snapshot; the caches add to one
     * case's estimate), clamped to [512 MiB, 8 GiB]. Cases over it are skipped by name. A warning
     * is printed when it is below 8 GiB. */
    if (max_case_bytes <= 0) {
        std::string meminfo;
        if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
            char buf[4096];
            size_t got;
            while ((got = std::fread(buf, 1, sizeof buf, mf)) > 0) meminfo.append(buf, got);
            std::fclose(mf);
        }
        int64_t avail = 0;
        const int64_t want = kb::default_case_budget(meminfo, &avail);
        max_case_bytes = want;
        if (want < ((int64_t)8 << 30))
            RAD_WARN("case budget is %s, taken from the %s this machine has available -- cases "
                     "over it are named and skipped. --max-bytes MIB overrides.",
                     humanb(want).c_str(), humanb(avail).c_str());
    }

    /* ================================================== which of the two jobs this is
     *
     * No model: replay a recorded fixture, in seconds on any machine. `-m`: check against a real
     * container's weights in the layout each resolved kernel asked for, which tests the artifact
     * and the layout hooks (and is how fixtures are recorded). */
    if (model.empty()) {
        if (!skeleton_path.empty() || !record_path.empty()) {
            std::fprintf(stderr, "rad-kbench: --%s needs -m <model.rad>\n",
                         skeleton_path.empty() ? "record" : "skeleton");
            return 2;
        }
        if (fixture_paths.empty()) fixture_paths.push_back(rad_home_file(home, "kernels.rkb"));
        ro.home = home;
        ro.hierarchy = hierarchy;
        ro.ref_arg = ref_name;
        ro.op_filters = op_filters;
        ro.tol_override = tol_override;
        ro.max_cases = max_cases;
        ro.max_case_bytes = max_case_bytes;
        ro.bench = bench;
        ro.timing_only = timing_only;
        ro.verbose = verbose;
        ro.timer = topts;
        ro.timer.budget = max_case_bytes;
        ro.timer.verbose = verbose;
        ro.report_path = report_path;
        ro.analysis_path = analysis_path;
        ro.fixtures = fixture_paths;
        Replay r(ro);
        return r.run();
    }
    if (!ro.rerecord_path.empty() || !ro.record_rows_path.empty() || !fixture_paths.empty()) {
        std::fprintf(stderr, "rad-kbench: --fixture, --rerecord and --record-rows replay a "
                             "fixture; drop -m\n");
        return 2;
    }

    if (!skeleton_path.empty()) {
        std::string err;
        if (kb::write_skeleton(model, skeleton_path, &err) != RAD_OK) {
            RAD_ERR("--skeleton: %s", err.c_str());
            return 1;
        }
        std::printf("wrote %s: the tables of %s, with the data region a hole\n",
                    skeleton_path.c_str(), model.c_str());
        return 0;
    }

    /* Where this pass's time goes, printed at its end as a replay's is. */
    Phases ph;
    const auto pass_t0 = std::chrono::steady_clock::now();
    PhaseTimer load_t(&ph, kb::PH_LOAD);

    RadFile f;
    if (f.open(model.c_str()) < 0) return 1;

    /* ---------------------------------------------------------------- plugins and declare */
    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, f.str(f.header().arch_id),
                               f.str(f.header().quant), &plugins) < 0) return 1;
    ref_name = resolve_reference(rad_tools_registry(), ref_name);
    if (ref_name.empty()) return 1;


    RadModelMeta meta{};
    std::vector<std::string> kk, kv;
    std::vector<const char*> kp, vp;
    meta.arch_id = f.str(f.header().arch_id);
    meta.name    = f.str(f.header().model_name);
    meta.quant   = f.str(f.header().quant);
    auto mi = [&](const char* k, int64_t d) { int64_t v = d; f.get_i(k, &v); return v; };
    auto mf = [&](const char* k, double d) { double v = d; f.get_f(k, &v); return (float)v; };
    meta.n_layers = mi("n_layers", 0);  meta.n_embd = mi("n_embd", 0);
    meta.n_head   = mi("n_head", 0);    meta.n_head_kv = mi("n_head_kv", 0);
    meta.head_dim = mi("head_dim", 0);  meta.n_ff = mi("n_ff", 0);
    meta.n_vocab  = mi("n_vocab", 0);   meta.n_ctx_train = mi("n_ctx_train", 0);
    meta.n_expert = mi("n_expert", 0);  meta.n_expert_used = mi("n_expert_used", 0);
    meta.n_expert_shared = mi("n_expert_shared", 0);
    meta.rms_eps = mf("rms_eps", 1e-6); meta.rope_theta = mf("rope_theta", 10000.f);
    meta.rope_scale = 1.f;   /* the rotary's scaling is the free-form rope keys (rad_builder.h) */

    /* GDN head counts (see ShapeCtx); zero for a non-hybrid container, so the gdn_* recipes
     * decline. Read through the string table too: rad-convert stores the checkpoint config
     * verbatim, so JSON numbers arrive as strings and get_i alone would answer 0. */
    auto ms = [&](const char* k, int64_t d) {
        int64_t v = d;
        if (f.get_i(k, &v)) return v;
        const char* s = f.get_s(k, nullptr);
        if (!s || !*s) return d;
        char* end = nullptr;
        const long long p = std::strtoll(s, &end, 10);
        return (end && end != s) ? (int64_t)p : d;
    };
    ShapeCtx sctx;
    sctx.n_head_k = ms("linear_num_key_heads",   ms("qwen35.ssm.group_count", 0));
    sctx.n_head_v = ms("linear_num_value_heads", ms("qwen35.ssm.time_step_rank", 0));
    sctx.n_seq    = 1;
    RAD_DEBUG("gdn head counts from the container: n_head_k=%lld n_head_v=%lld",
              (long long)sctx.n_head_k, (long long)sctx.n_head_v);
    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const RadFileKV& e = f.meta_at(i);
        if (e.type != RAD_P_STR) continue;
        kk.push_back(f.str(e.key));
        kv.push_back(f.str(e.v.s));
    }
    for (auto& s : kk) kp.push_back(s.c_str());
    for (auto& s : kv) vp.push_back(s.c_str());
    meta.n_kv = (int)kk.size(); meta.kv_key = kp.data(); meta.kv_val = vp.data();

    /* Declare with the drafter so its kernels (conv, selector, MTP head, draft GEMMs, `row_topk`,
     * ...) are checked; max_spec 0 would leave them out. The default depth is the plugin's
     * (rad_arch_probe), as in the engine and rad-tune, so the checked graph is the served one. */
    RadBuildCtx ctx{};
    ctx.rank = 0; ctx.world_size = 1; ctx.max_tok = max_tok;
    RadArchProbe probe{};
    (void)rad_tools_probe(plugins, meta, &probe);
    ctx.max_spec = n_spec >= 0 ? n_spec : probe.draft_depth;
    if (mm_max_patches < 0) {
        const char* enc = rad_meta_gets(&meta, "radiance.encoder", "");
        mm_max_patches = (enc && *enc) ? 2048 : 0;
    }
    if (mm_max_patches > 0 && probe.enc_merge > 1)
        mm_max_patches -= mm_max_patches % probe.enc_merge;
    ctx.max_enc_patches = mm_max_patches > 0 ? mm_max_patches : 0;
    {
        const char* enc = rad_meta_gets(&meta, "radiance.encoder", "");
        if (mm_max_patches != 0 && enc && std::strstr(enc, "audio") && probe.audio_frames > 0)
            ctx.max_audio_frames = probe.audio_frames;
    }
    if (ctx.max_enc_patches > 0)
        RAD_INFO("  encoder pass %lld patches (--mm-max-patches)", (long long)ctx.max_enc_patches);
    ctx.max_seqs = 8;
    ctx.max_ctx = max_ctx_flag > 0 ? max_ctx_flag : meta.n_ctx_train;
    ctx.kv_dtype = kv_dtype;
    ctx.scope = "";
    if (max_ctx_flag > 0) sweep_max_ctx = max_ctx_flag;
    if (ctx.max_spec > 0)
        RAD_INFO("  draft depth  %d (the container's own; --num-speculative-tokens overrides)",
                 ctx.max_spec);

    /* Also declare the sampler chain, which the engine (not the architecture plugin) declares,
     * so its twelve stages (`sample_penalties` ... `sample_merge_topk`) are checked. libref's shape
     * recipes for them use fill laws that exercise the narrowing stages on some rows. The engine's
     * sampler takes the token buffer as an argument (core/engine.cpp) for this call. Failure is
     * not fatal; the rest of the graph is still checked. A replay has sampler cases only if its
     * fixture was recorded with the chain declared. */
    Sampler sampler;
    auto also = [&](RadBuilder* b, int64_t logits_width) -> int {
        SamplerConfig sc{};
        sc.n_vocab       = meta.n_vocab;
        sc.n_vocab_local = logits_width > 0 ? logits_width : meta.n_vocab;
        sc.max_seqs      = ctx.max_seqs;
        sc.max_spec      = ctx.max_spec;
        sc.rank          = ctx.rank;
        sc.world_size    = ctx.world_size;
        const int s = sampler.declare(b, sc);
        if (s < 0)
            RAD_WARN("the sampler chain did not declare (%s); its twelve stages are not in this "
                     "pass", rad_strerror(s));
        return RAD_OK;
    };

    Program prog;
    if (rad_tools_declare(plugins, meta, ctx, &prog, also, nullptr,
                          rad_container_sources(f)) < 0) return 1;
    load_t.stop();

    Registry& reg = rad_tools_registry();

    /* The oracle's host row for an op, found by scanning rows; selection would return the kernel
     * under test. */
    auto ref_for = [&](const std::string& op, const Geometry& g) -> const KernelRow* {
        return reg.select_in(ref_name, op, g, RAD_DOMAIN_HOST);
    };

    std::printf("rad-kbench %s\n", model.c_str());
    std::printf("  oracle       %s (host domain)\n", ref_name.c_str());
    std::printf("  libraries    %s\n", plugins.libraries().c_str());
    std::printf("  ops declared %zu, seed %llu\n\n", prog.ops.size() - 1,
                (unsigned long long)seed);

    /* The fixture being recorded (--record), as a side effect of the container pass so it holds
     * exactly the cases that ran. */
    const bool recording = !record_path.empty();
    Kfixture fixture;
    /* The container's file name only; the fixture is checked in. */
    fixture.source = model.substr(model.find_last_of('/') + 1);
    fixture.oracle = ref_name;
    Stats st;
    Named nm;
    std::map<std::string, int>& skipped_ops = nm.skipped;
    /* One counter per distinct (op, kernel, plugin, domain, geometry, tuned point), the report's
     * grouping key. */
    std::map<std::string, int> group_seen;
    /* Fusion checks (byte identity against a chain) are tallied separately from the reference
     * comparison. */
    std::map<std::string, int>& fuse_skipped = nm.fuse_skipped;
    /* Cases where kernel and oracle describe an output differently; neither pass nor fail. */
    std::map<std::string, int>& diverged = nm.diverged;
    /* Kernels caught writing outside an operand or their scratch. Counted separately: a kernel can
     * be numerically correct and still corrupt a neighbour, which here lands in unused memory. */
    std::map<std::string, int>& overruns = nm.overruns;
    int fuse_passed = 0, fuse_failed = 0;
    std::vector<Row> report_rows;
    /* --bench: the device timer (made at the first passing device case) and the rotation footprint
     * host rows are timed with. */
    std::unique_ptr<kb::DeviceTimer> timer;
    topts.budget = max_case_bytes;
    topts.verbose = verbose;
    const int64_t host_footprint =
        !bench ? 0 : topts.footprint >= 0 ? topts.footprint
                                          : kb::default_footprint(kb::host_llc_bytes());

    for (size_t oi = 1; oi < prog.ops.size(); ++oi) {
        const OpInfo& o = prog.ops[oi];
        if (!op_filters.empty()) {
            bool hit = false;
            for (const std::string& f : op_filters) if (o.op.rfind(f, 0) == 0) { hit = true; break; }
            if (!hit) continue;
        }
        if (!o.schema) continue;

        /* ================================================== the points this op is tested at
         *
         * Each band is swept at its lower edge, upper edge and geometric interior points; the
         * lower edge (shortest call the kernel claims) is where tile-tail bugs show, e.g. M = 65
         * in a (64, 512] band.
         *
         * Context depth is swept separately: for attention the query length is the ranged key, and
         * `ref_attn_paged_shape` makes the context equal to the query, so without a depth sweep a
         * 10-row decode only sees 10 tokens of KV. Block tables, split-K over KV and window
         * masking all depend on depth. Any op whose operand description responds to depth is
         * swept, detected rather than listed. */
        std::vector<int64_t> ms;
        {
            const bool ranged = !o.ranged_key.empty();
            for (size_t b = 0; b < o.bands.size(); ++b) {
                int64_t hi = o.bands[b].hi;
                if (hi <= 0 || hi > (1 << 20)) hi = max_tok;
                if (!ranged) { ms.push_back(1); continue; }
                hi = std::min<int64_t>(hi, max_tok);
                int64_t lo = b ? o.bands[b - 1].hi + 1 : std::max<int64_t>(o.range_lo, 1);
                if (lo <= 0) lo = 1;
                lo = std::min(lo, hi);
                /* lo, hi and a geometric interior (65..512 -> 65, 130, 260, 512): tile and split
                 * counts move with the ratio. */
                ms.push_back(lo);
                for (int64_t m = lo * 2; m < hi; m *= 2) ms.push_back(m);
                if (hi != lo) ms.push_back(hi);
            }
            std::sort(ms.begin(), ms.end());
            ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
        }
        /* The band of each point, so it is checked against the kernel the selector would pick. */
        auto band_of = [&](int64_t m) -> size_t {
            for (size_t b = 0; b < o.bands.size(); ++b) {
                int64_t hi = o.bands[b].hi;
                if (hi <= 0 || hi > (1 << 20)) hi = max_tok;
                if (m <= hi) return b;
            }
            return o.bands.empty() ? 0 : o.bands.size() - 1;
        };

        /* Does this op's operand description depend on the context depth? Probed against the
         * oracle, whose description defines the op. */
        std::vector<int64_t> ctxs{ 0 };   /* 0 means "whatever the recipe's own default is" */
        if (!o.bands.empty()) {
            const Resolved* r0 = nullptr;
            for (size_t b = 0; b < o.bands.size() && !r0; ++b)
                for (int d = 0; d < RAD_N_DOMAINS && !r0; ++d)
                    if (o.bands[b].dom[d] && o.bands[b].dom[d].row) r0 = &o.bands[b].dom[d];
            const KernelRow* orc = r0 ? ref_for(o.op, r0->geom) : nullptr;
            if (orc) {
                Geometry ga = r0->geom, gb = r0->geom;
                ga.set_i("ctx", 64); gb.set_i("ctx", 4096);
                std::vector<OpdShape> sa, sb;
                if (rad_operand_shapes_hook(orc->info, ga, o.schema->n_operands, &sa) >= 0 &&
                    rad_operand_shapes_hook(orc->info, gb, o.schema->n_operands, &sb) >= 0 &&
                    sa.size() == sb.size()) {
                    for (size_t k = 0; k < sa.size(); ++k)
                        if (sa[k].shape != sb[k].shape) {
                            for (int64_t c : { (int64_t)2048, (int64_t)16384 })
                                if (c <= sweep_max_ctx) ctxs.push_back(c);
                            break;
                        }
                }
            }
        }

        /* The (M, depth) points as one flat list; each M's band is looked up. */
        struct CasePt { int64_t m; int64_t ctx; };
        std::vector<CasePt> pts;
        for (int64_t m : ms) for (int64_t c : ctxs) pts.push_back({ m, c });

        for (const CasePt& pt : pts) {
            const size_t b = band_of(pt.m);
            /* A recording takes the reference's answer once per point, whichever library resolved
             * it and in whichever domain: a host-only machine resolves every op to the reference
             * itself, and the answer does not depend on the kernel under test. */
            bool point_recorded = false;
            for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
                const Resolved& r = o.bands[b].dom[dom];
                if (!r || !r.row || !r.row->info || !r.row->info->launch) continue;
                /* Checking ref against itself proves nothing and costs a lot at 151K vocab, so a
                 * point the reference resolved is visited only to record it. */
                const bool record_only = r.row->plugin == ref_name;
                if (record_only && (!recording || point_recorded)) continue;
                const bool record_here = recording && !point_recorded;
                /* `r.geom` is the selector's resolution (band upper bound plus tuned axes). The
                 * point's M goes into a copy that everything downstream reads (shape hooks, launch
                 * params, report, cap key), so params and operand sizes agree. The tuned axes still
                 * apply: the point is inside their band. */
                const int64_t M = pt.m;
                Geometry g = r.geom;
                if (!o.ranged_key.empty()) g.set_i(o.ranged_key, M);

                const KernelRow* oracle = ref_for(o.op, g);
                /* Without an oracle, a fused kernel that declares `replaces` is still checked,
                 * byte for byte against its own plugin's chain. A timing-only run times device
                 * kernels only; one without an oracle uses its own operand description. */
                if (timing_only && dom != RAD_DOMAIN_DEVICE) continue;
                const bool fuse_only = !oracle && fuse_check && dom == RAD_DOMAIN_DEVICE &&
                                       r.row->info->replaces && !record_only;
                if (!oracle && !fuse_only && !timing_only) {
                    /* An op outside libref's conventional vocabulary has no oracle (spec §17). */
                    skipped_ops[o.op + "  (no " + ref_name + " kernel: no oracle exists for it)"]++;
                    ++st.skipped;
                    continue;
                }

                /* Cap runs per (kernel, geometry, tuned point, depth). Counted after the skips so
                 * the skip inventory stays complete. */
                if (max_cases > 0) {
                    const std::string gk = o.op + "\x1f" + r.row->info->name + "\x1f" +
                                           r.row->plugin + "\x1f" + std::to_string(dom) + "\x1f" +
                                           g.str() + "\x1f" + r.choice + "\x1f" +
                                           std::to_string(pt.ctx);
                    if (++group_seen[gk] > max_cases) { ++st.capped; continue; }
                }

                /* Operand shapes come from the oracle's description (reference semantics define
                 * the op, and a drifted kernel must not describe its own test case), then the
                 * kernel's hook, then the opshapes.h table. If all decline, the case is skipped.
                 *
                 * The GDN head counts (ShapeCtx) go to the hook as ordinary parameters, since
                 * RadShapeFn takes only a geometry. They are set on a copy, `hgeom`, because `g` is
                 * what the kernel was selected on. */
                Geometry hgeom = g;
                if (sctx.n_head_k > 0) hgeom.set_i("n_head_k", sctx.n_head_k);
                if (sctx.n_head_v > 0) hgeom.set_i("n_head_v", sctx.n_head_v);
                if (sctx.n_seq   > 0) hgeom.set_i("n_seq",   sctx.n_seq);
                /* The depth also goes in as a parameter: no op declares `ctx` (paged kernels read
                 * it from `seqused` and the block table), so the description needs it to size the
                 * cache. Zero keeps the recipe's default. */
                if (pt.ctx > 0) hgeom.set_i("ctx", pt.ctx);

                std::vector<OpdShape> sh;
                int shst = oracle ? rad_operand_shapes_hook(oracle->info, hgeom,
                                                            o.schema->n_operands, &sh)
                                  : RAD_E_UNSUPPORTED;
                if (shst < 0)
                    shst = rad_operand_shapes_hook(r.row->info, hgeom, o.schema->n_operands, &sh);
                if (shst < 0)
                    shst = rad_operand_shapes(o.op, g, o.schema->n_operands, M, &sh, &sctx);
                if (shst < 0) {
                    skipped_ops[o.op + "  (neither the oracle nor the kernel describes its "
                                       "operands, and opshapes.h has no recipe)"]++;
                    ++st.skipped;
                    continue;
                }
                /* An optional weight the declaration leaves unbound (a norm with no gain, a
                 * LayerNorm with no bias) is absent, as at launch. The weight list is positional
                 * over WEIGHT-role operands, so an unbound one still takes its slot. */
                {
                    size_t wpos = 0;
                    for (size_t k = 0; k < sh.size() && k < (size_t)o.schema->n_operands; ++k) {
                        const RadOperandSpec& os = o.schema->operands[k];
                        if (os.role != RAD_OPD_WEIGHT) continue;
                        const rad_weight h = wpos < o.weights.size() ? o.weights[wpos] : 0;
                        ++wpos;
                        if (!h && os.optional) sh[k].absent = true;
                    }
                }

                /* If the kernel's own description of an output differs from the oracle's, the two
                 * libraries implement different ops under one name, and a numeric comparison would
                 * be meaningless. Reported as a divergence with both descriptions. Inputs are not
                 * compared: a differently read input shows up as a wrong output. */
                if (oracle && r.row->info->opd_shape) {
                    std::vector<OpdShape> ksh;
                    if (rad_operand_shapes_hook(r.row->info, hgeom, o.schema->n_operands,
                                                &ksh) >= 0 && ksh.size() == sh.size()) {
                        std::string why;
                        for (size_t k = 0; k < sh.size() && why.empty(); ++k) {
                            const int role = o.schema->operands[k].role;
                            if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                            if (sh[k].absent || ksh[k].absent) continue;
                            if (sh[k].dtype == ksh[k].dtype && sh[k].shape == ksh[k].shape)
                                continue;
                            const char* nm = o.schema->operands[k].name
                                                 ? o.schema->operands[k].name : "?";
                            why = std::string("'") + nm + "': " + ref_name + " says " +
                                  shape_str(sh[k]) + ", " + r.row->info->name + " says " +
                                  shape_str(ksh[k]);
                        }
                        if (!why.empty()) {
                            diverged[o.op + "  (" + std::string(r.row->info->name) + " and " +
                                     ref_name + " describe the same output differently -- " + why +
                                     ")"]++;
                            ++st.diverged;
                            continue;
                        }
                    }
                }

                /* The case in its on-disk form (`kop`), which everything below reads, as a replay
                 * does, so recording and replay share one description. The case seed is stored and
                 * operand seeds derive from it, so a replay reproduces the bytes from the file. */
                const uint64_t case_seed =
                    rad_case_seed(seed, o.op, (int)b, dom, 0) ^ ((uint64_t)M << 32) ^
                    (uint64_t)pt.ctx;
                std::vector<KfOperand> kop((size_t)o.schema->n_operands);
                for (size_t k = 0; k < sh.size() && k < kop.size(); ++k) {
                    KfOperand& d = kop[k];
                    d.role       = o.schema->operands[k].role;
                    d.dtype      = sh[k].dtype;
                    d.shape      = sh[k].shape;
                    d.absent     = sh[k].absent;
                    d.fill       = sh[k].fill;
                    d.is_index   = sh[k].is_index;
                    d.idx_unique = sh[k].idx_unique;
                    d.idx_cu     = sh[k].idx_cu;
                    d.idx_prefix = sh[k].idx_prefix;
                    d.idx_history = sh[k].idx_history;
                    d.idx_max    = sh[k].idx_max;
                    d.idx_const  = sh[k].idx_const;
                    d.fill_chunk = sh[k].fill_chunk;
                }
                /* A weight table records its first expert's encoding and selection: every expert
                 * of a run shares one (the container's fixed-stride addressing requires it), and
                 * the replay draws each expert's planes from it (kf_draw_table). */
                for (size_t k = 0; k < kop.size() && k < sh.size(); ++k) {
                    if (kop[k].role != RAD_OPD_WTAB || kop[k].absent) continue;
                    for (size_t wi = 0; wi < o.weights.size() && wi < o.weight_opd.size(); ++wi) {
                        if (o.weight_opd[wi] != (int)k) continue;
                        const rad_weight h = o.weights[wi];
                        if (h && h < prog.weights.size()) kf_operand_enc(prog.weights[h], &kop[k]);
                        break;
                    }
                }

                /* Skip by name a case over the budget before allocating: every operand is built
                 * for kernel and oracle, a re-laid weight a third time (Qwen4-Exp's 51.2 GB
                 * n-gram table would need ~150 GB). */
                {
                    int64_t want = 0;
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent)
                            want += 3 * rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
                    if (want > max_case_bytes) {
                        skipped_ops[o.op + "  (one case needs " + humanb(want) +
                                    ", over the " + humanb(max_case_bytes) +
                                    " budget -- raise --max-bytes to check it)"]++;
                        ++st.skipped;
                        continue;
                    }
                }

                std::vector<std::vector<uint8_t>> bufs(sh.size()), ref_out(sh.size());
                std::vector<RadTensor> t_test(sh.size()), t_ref(sh.size());
                int wk = 0;
                bool from_container = false, unusable = false;
                PhaseTimer draw_t(&ph, kb::PH_DRAW);

                for (size_t k = 0; k < sh.size() && !unusable; ++k) {
                    const int64_t n = rad_numel(sh[k].shape);
                    const int role = o.schema->operands[k].role;

                    /* An absent optional operand: a null RadTensor on both sides, not compared. */
                    if (sh[k].absent) {
                        /* An absent weight still consumes its handle (the list is positional). */
                        if (role == RAD_OPD_WEIGHT) ++wk;
                        t_test[k] = RadTensor{};
                        t_ref[k]  = RadTensor{};
                        continue;
                    }

                    bufs[k].assign((size_t)rad_dtype_bytes(sh[k].dtype, n), 0);
                    if (bufs[k].empty() && n > 0) { unusable = true; break; }

                    /* The tensor description both sides share. */
                    auto mk = [&](std::vector<uint8_t>& store) {
                        RadTensor t{};
                        t.data  = store.data();
                        t.dtype = sh[k].dtype;
                        t.rank  = (uint32_t)sh[k].shape.size();
                        for (size_t d = 0; d < sh[k].shape.size() && d < RAD_MAX_RANK; ++d)
                            t.shape[d] = sh[k].shape[d];
                        rad_tensor_pack(&t);
                        return t;
                    };

                    if (role == RAD_OPD_WEIGHT) {
                        /* Weights are the only operands the two sides read differently: the kernel
                         * reads its layout hooks' arrangement of the planes, the oracle the tensor
                         * they decode to.
                         *
                         * A recording always draws the planes (and records the encoding) so it can
                         * be replayed without the checkpoint. A container pass reads them, except
                         * for weights the container lacks (a declaration wider than the artifact,
                         * e.g. a 4-layer bring-up slice), which are drawn and reported as
                         * synthesised.
                         *
                         * `t_test` is described as Ctx::refresh_weights does (kf_describe_laid). */
                        const rad_weight h = wk < (int)o.weights.size() ? o.weights[(size_t)wk++]
                                                                       : 0;
                        const WeightInfo* wi = h ? &prog.weights[h] : nullptr;
                        const char* opd_name = o.schema->operands[k].name
                                                   ? o.schema->operands[k].name : "?";
                        if (!wi) {
                            skipped_ops[o.op + "  (operand '" + opd_name + "' is a weight and "
                                        "the declaration bound none to it)"]++;
                            unusable = true;
                            break;
                        }
                        kf_operand_enc(*wi, &kop[k]);
                        const RadFileEntry* e = recording ? nullptr : f.find(wi->source);
                        std::vector<std::vector<uint8_t>> pl;
                        if (!e) {
                            kf_draw_planes(kop[k], kf_weight_seed(kop[k], k), &pl);
                        } else {
                            /* One rank of one: each selected plane is the whole plane. */
                            for (int j = 0; j < wi->n_sel; ++j) {
                                const RadFilePlane& fp = f.plane(*e, wi->sel[j]);
                                const int64_t want =
                                    wi->sel_rows[j] *
                                    rad_enc_row_bytes(wi->enc.plane[wi->sel[j]].dtype,
                                                      wi->sel_cols[j]);
                                if ((int64_t)fp.bytes != want) {
                                    skipped_ops[o.op + "  ('" + wi->source + "' plane '" +
                                                wi->enc.plane[wi->sel[j]].role + "' holds " +
                                                std::to_string(fp.bytes) + " bytes and the "
                                                "declaration reads " + std::to_string(want) +
                                                ")"]++;
                                    unusable = true;
                                    break;
                                }
                                const uint8_t* src = (const uint8_t*)f.data(fp);
                                pl.emplace_back(src, src + fp.bytes);
                            }
                            if (unusable) break;
                            from_container = true;
                        }
                        std::string why;
                        if (kf_planes_logical(kop[k], pl, &ref_out[k], &why) != RAD_OK) {
                            skipped_ops[o.op + "  ('" + wi->source + "' as " +
                                        enc_name(wi->enc) + " does not make the operand '" +
                                        opd_name + "' the reference reads: " + why + ")"]++;
                            unusable = true;
                            break;
                        }
                        t_ref[k] = mk(ref_out[k]);
                        RadLayout lay{};
                        std::vector<uint8_t> laid;
                        const int lrc = kf_lay_weight(r.row->info, g.params(), g.n_params(),
                                                      (int)k, kop[k], pl, &lay, &laid);
                        if (lrc == RAD_OK) {
                            bufs[k].swap(laid);
                            kf_describe_laid(kop[k], lay, bufs[k].data(), &t_test[k]);
                        } else if (lrc == RAD_E_UNSUPPORTED && pl.size() == 1) {
                            /* read as stored: the plane, widened where the operand is wider */
                            bufs[k] = ref_out[k];
                            t_test[k] = mk(bufs[k]);
                        } else {
                            skipped_ops[o.op + "  (" + std::string(r.row->info->name) +
                                        " cannot lay out operand '" + opd_name + "' from " +
                                        enc_name(wi->enc) + ": " +
                                        (lrc == RAD_E_UNSUPPORTED ? "it reads several planes "
                                                                    "as stored"
                                                                  : rad_strerror(lrc)) +
                                        ")"]++;
                            unusable = true;
                            break;
                        }
                        continue;
                    } else if (role == RAD_OPD_WTAB) {
                        /* A weight table, stacked [n_expert, ...]: each expert's selected planes
                         * from the container, decoded as the reference reads them, or drawn when
                         * recording or when an expert is missing (a declaration wider than the
                         * artifact). */
                        std::vector<rad_weight> run;
                        for (size_t wi = 0; wi < o.weights.size() && wi < o.weight_opd.size(); ++wi)
                            if (o.weight_opd[wi] == (int)k) run.push_back(o.weights[wi]);
                        bool read = !recording && kop[k].has_enc && !kop[k].shape.empty() &&
                                    (int64_t)run.size() == kop[k].shape[0];
                        const int64_t per = read ? rad_dtype_bytes(kop[k].dtype, n / kop[k].shape[0])
                                                 : 0;
                        for (size_t e = 0; read && e < run.size(); ++e) {
                            const WeightInfo* wi = run[e] ? &prog.weights[run[e]] : nullptr;
                            const RadFileEntry* fe = wi ? f.find(wi->source) : nullptr;
                            if (!fe) { read = false; break; }
                            std::vector<std::vector<uint8_t>> pl;
                            for (int j = 0; j < wi->n_sel && read; ++j) {
                                const RadFilePlane& fp = f.plane(*fe, wi->sel[j]);
                                const uint8_t* src = (const uint8_t*)f.data(fp);
                                pl.emplace_back(src, src + fp.bytes);
                            }
                            KfOperand one = kop[k];
                            kf_operand_enc(*wi, &one);
                            one.shape.assign(kop[k].shape.begin() + 1, kop[k].shape.end());
                            std::vector<uint8_t> logical;
                            std::string why;
                            if (kf_planes_logical(one, pl, &logical, &why) != RAD_OK ||
                                (int64_t)logical.size() != per) {
                                read = false;
                                break;
                            }
                            std::memcpy(bufs[k].data() + per * (int64_t)e, logical.data(),
                                        (size_t)per);
                        }
                        if (read) from_container = true;
                        else kf_draw(kop, k, M, kf_vocab_rows(kop), 0, &bufs[k]);
                    } else if (role == RAD_OPD_IN || role == RAD_OPD_INOUT) {
                        /* The fixture's draw, the single implementation recorder and replay
                         * share. */
                        kf_draw(kop, k, M, kf_vocab_rows(kop), kf_operand_seed(case_seed, (int)k),
                                &bufs[k]);
                    }

                    ref_out[k] = bufs[k];   /* same inputs, separate outputs */
                    t_test[k] = mk(bufs[k]);
                    t_ref[k]  = mk(ref_out[k]);
                }
                draw_t.stop();
                if (unusable) { ++st.skipped; continue; }

                std::vector<uint8_t> scratch_a, scratch_b;
                auto launch = [&](const KernelRow* row, std::vector<RadTensor>& tv,
                                  std::vector<uint8_t>& scratch) {
                    RadArgs a = case_args(tv, g);
                    if (row->info->scratch) {
                        const int64_t need = row->info->scratch(&a);
                        if (need > 0) { scratch.assign((size_t)need, 0);
                                        a.scratch = scratch.data(); a.scratch_bytes = need; }
                    }
                    return row->info->launch(&a, nullptr);
                };

                /* ---- Device kernels need device memory: the host vectors above suit the oracle
                 * (domain HOST), but a device kernel handed one may silently write nothing, and
                 * zero inputs would then compare as a pass. So each operand is uploaded to its own
                 * allocation on a real stream, and outputs are copied back into the host buffers
                 * the comparison reads. Both sides see identical input bytes. */
                /* Every device allocation is guarded (see "red zones"). */
                std::vector<DevBuf> dev(sh.size());
                DevBuf dev_scratch;
                RadStream stream = nullptr;
                bool dev_ok = true;
                if (dom == RAD_DOMAIN_DEVICE) {
                    if (!(stream = run_stream())) dev_ok = false;
                    for (size_t k = 0; k < sh.size() && dev_ok; ++k) {
                        if (sh[k].absent || bufs[k].empty()) continue;
                        if (!guarded_alloc(dev[k], (int64_t)bufs[k].size(), stream)) {
                            dev_ok = false; break;
                        }
                        if (rad_memcpy_async(dev[k].p, bufs[k].data(), dev[k].bytes, stream) < 0)
                            dev_ok = false;
                        t_test[k].data = dev[k].p;
                    }
                    if (dev_ok && rad_stream_sync(stream) < 0) dev_ok = false;
                }

                /* INOUT operands' original bytes, kept for the fusion check and the timing pass: the
                 * chain shares the fused run's inputs and weights, but the fused launch overwrites
                 * INOUT operands, so the chain must start from a snapshot, and the timing pass
                 * restores them from it before every launch. */
                std::vector<std::vector<uint8_t>> inout0;
                const bool want_fuse = fuse_check && dom == RAD_DOMAIN_DEVICE && dev_ok &&
                                       r.row->info->replaces;
                if (want_fuse || (bench && !record_only)) {
                    inout0.resize(sh.size());
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent && o.schema->operands[k].role == RAD_OPD_INOUT)
                            inout0[k] = bufs[k];
                }

                auto dev_release = [&]() {
                    if (stream) (void)rad_stream_sync(stream);
                    for (auto& d : dev) if (d.base && d.owned) rad_dev_free(d.base, RAD_MEM_DEVICE);
                    if (dev_scratch.base) rad_dev_free(dev_scratch.base, RAD_MEM_DEVICE);
                };

                if (!dev_ok) {
                    std::printf("  %-22s %-40s SKIP  no device memory for this case: %s\n",
                                o.op.c_str(), g.str().c_str(), rad_dev_last_error());
                    dev_release();
                    ++st.skipped;
                    continue;
                }

                /* Device launch, with device scratch. */
                auto launch_dev = [&](const KernelRow* row, std::vector<RadTensor>& tv) -> int {
                    RadArgs a = case_args(tv, g);
                    if (row->info->scratch) {
                        const int64_t need = row->info->scratch(&a);
                        if (need > 0) {
                            /* Guarded too: the size comes from the kernel's own RadScratchFn, so an
                             * overrun means the kernel disagrees with itself. */
                            if (!guarded_alloc(dev_scratch, need, stream)) return RAD_E_NOMEM;
                            if (rad_memset_async(dev_scratch.p, 0, need, stream) < 0)
                                return RAD_E_DEVICE;
                            a.scratch = dev_scratch.p;
                            a.scratch_bytes = need;
                        }
                    }
                    const int s = row->info->launch(&a, stream);
                    if (s < 0) return s;
                    return rad_stream_sync(stream);
                };

                /* The oracle's launch counts toward its library's share of the launch phase. */
                const auto tref0 = std::chrono::steady_clock::now();
                const int s_ref  = oracle && !timing_only ? launch(oracle, t_ref, scratch_b) : RAD_OK;
                if (oracle && !timing_only && s_ref >= 0) {
                    const double sec = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - tref0).count();
                    ph.add(kb::PH_LAUNCH, sec);
                    ph.launch_by_lib[ref_name] += sec;
                }
                double test_s = 0;
                const auto ttest0 = std::chrono::steady_clock::now();
                const int s_test = record_only ? s_ref
                                   : dom == RAD_DOMAIN_DEVICE ? launch_dev(r.row, t_test)
                                                              : launch(r.row, t_test, scratch_a);
                if (!record_only) {
                    const double sec = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - ttest0).count();
                    test_s = sec;
                    ph.add(kb::PH_LAUNCH, sec);
                    ph.launch_by_lib[r.row->plugin] += sec;
                }
                if (record_only)
                    for (size_t k = 0; k < sh.size(); ++k) bufs[k] = ref_out[k];

                /* Copy the checked outputs back before timing: in-place ops (`rope`,
                 * `gdn_conv_update`) would otherwise be applied once per timed launch and fail
                 * with --bench. A timing-only run reads nothing back. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0 && !timing_only && !record_only) {
                    for (size_t k = 0; k < sh.size(); ++k) {
                        if (!dev[k].p) continue;
                        const int role = o.schema->operands[k].role;
                        if (role == RAD_OPD_OUT || role == RAD_OPD_INOUT)
                            rad_memcpy_async(bufs[k].data(), dev[k].p, dev[k].bytes, stream);
                    }
                    rad_stream_sync(stream);
                }

                /* Check the margins before the timed loop, so an overrun is attributed to the
                 * checked launch. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0 && !timing_only && !record_only)
                    check_margins(dev, dev_scratch, o.schema, o.op, r.row->info->name, g, stream,
                                  &overruns, &st);

                /* ================================================== the fusion oracle
                 *
                 * Run the chain this kernel declares it replaces, from the same plugin, and memcmp
                 * the outputs: a fusion claims byte identity with its chain. This also covers fused
                 * ops libref lacks (e.g. every fp8 quantiser fold).
                 *
                 * Inputs and weights are the fused run's device buffers. Outputs get fresh
                 * allocations; INOUT operands get fresh ones restored from the pre-launch snapshot.
                 * A temporary (a buffer passed between chain steps, e.g. gemm_nt_q_gated's [M, 2n]
                 * product) is sized by the step's own opd_shape hook, or the check is declined. */
                if (want_fuse && s_test >= 0) {
                    std::string why;
                    std::vector<DevBuf> fdev(sh.size());
                    std::vector<DevBuf> ftmp(RAD_MAX_FUSE_TMP);
                    std::vector<RadTensor> ftmp_t(RAD_MAX_FUSE_TMP);
                    std::vector<std::vector<uint8_t>> fout(sh.size());
                    DevBuf fscratch;

                    /* Fresh device memory for everything the chain writes. */
                    for (size_t k = 0; k < sh.size() && why.empty(); ++k) {
                        if (sh[k].absent || bufs[k].empty()) continue;
                        const int role = o.schema->operands[k].role;
                        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                        fdev[k].bytes = (int64_t)bufs[k].size();
                        fdev[k].p = rad_dev_alloc(fdev[k].bytes, RAD_MEM_DEVICE);
                        if (!fdev[k].p) { why = "no device memory for the chain's outputs"; break; }
                        /* INOUT starts from the fused run's starting bytes; OUT starts zeroed, so
                         * an unwritten output shows as a difference. */
                        const void* src = role == RAD_OPD_INOUT && !inout0[k].empty()
                                              ? (const void*)inout0[k].data() : nullptr;
                        if (src) {
                            if (rad_memcpy_async(fdev[k].p, src, fdev[k].bytes, stream) < 0)
                                why = "upload failed";
                        } else if (rad_memset_async(fdev[k].p, 0, fdev[k].bytes, stream) < 0) {
                            why = "memset failed";
                        }
                    }
                    if (why.empty() && rad_stream_sync(stream) < 0) why = "stream sync failed";

                    /* Walk the chain; the hook ends it by returning RAD_E_NOTFOUND. */
                    for (int step = 0; step < RAD_MAX_FUSE_OPD && why.empty(); ++step) {
                        RadFuseStep fs{};
                        /* The actual set of present operands, not RAD_FUSE_PRESENT_ALL. */
                        uint64_t present = 0;
                        for (size_t k = 0; k < sh.size() && k < 64; ++k)
                            if (!sh[k].absent) present |= (uint64_t)1 << k;
                        const int fst = r.row->info->replaces(g.params(), g.n_params(),
                                                              present, step, &fs);
                        if (fst == RAD_E_NOTFOUND) {
                            if (step == 0) why = "declares an empty chain";
                            break;
                        }
                        if (fst < 0) {
                            why = std::string("the chain declines this geometry: ") +
                                  rad_strerror(fst);
                            break;
                        }
                        if (!fs.op) { why = "a step with no op name"; break; }

                        Geometry sg;
                        for (int pi = 0; pi < fs.n_p && pi < RAD_MAX_FUSE_PARAMS; ++pi) {
                            const RadParam& sp = fs.p[pi];
                            if (!sp.key) continue;
                            if (sp.kind == RAD_P_INT)      sg.set_i(sp.key, sp.ival);
                            else if (sp.kind == RAD_P_STR) sg.set_s(sp.key, sp.sval ? sp.sval : "");
                            else if (sp.kind == RAD_P_F64) sg.set_f(sp.key, sp.dval);
                        }

                        /* The fused kernel's own library: a fusion claims byte identity with
                         * the chain there, and selection would return the fused kernel itself. */
                        const KernelRow* srow =
                            reg.select_in(r.row->plugin, fs.op, sg, RAD_DOMAIN_DEVICE);
                        if (!srow) {
                            why = std::string("no ") + r.row->plugin + " kernel for step '" +
                                  fs.op + "' at " + sg.str();
                            break;
                        }
                        const RadOpSchema* ss = reg.schema(fs.op);
                        if (!ss) { why = std::string("step '") + fs.op + "' has no schema"; break; }
                        /* Operand-count mismatch means the hook has drifted from the schema. */
                        if (fs.n_from != ss->n_operands) {
                            why = fmt("step '%s' wires %d operands, its schema declares %d",
                                      fs.op, fs.n_from, ss->n_operands);
                            break;
                        }

                        /* The step's own description, for any temporary it names. */
                        std::vector<OpdShape> ssh;
                        const bool have_ssh =
                            rad_operand_shapes_hook(srow->info, sg, ss->n_operands, &ssh) >= 0;

                        std::vector<RadTensor> st_t((size_t)ss->n_operands);
                        for (int oi = 0; oi < ss->n_operands && why.empty(); ++oi) {
                            const int16_t src = fs.from[oi];
                            if (src == RAD_FUSE_NONE) { st_t[(size_t)oi] = RadTensor{}; continue; }
                            if (RAD_FUSE_IS_TMP(src)) {
                                const int ti = RAD_FUSE_TMP_INDEX(src);
                                if (ti < 0 || ti >= RAD_MAX_FUSE_TMP) {
                                    why = "temporary index out of range"; break;
                                }
                                if (!ftmp[(size_t)ti].p) {
                                    if (!have_ssh || (size_t)oi >= ssh.size()) {
                                        why = fmt("step '%s' names temporary %d and does not "
                                                  "describe its operands", fs.op, ti);
                                        break;
                                    }
                                    const OpdShape& d = ssh[(size_t)oi];
                                    const int64_t nb =
                                        rad_dtype_bytes(d.dtype, rad_numel(d.shape));
                                    if (nb <= 0) { why = "a temporary of no size"; break; }
                                    ftmp[(size_t)ti].p = rad_dev_alloc(nb, RAD_MEM_DEVICE);
                                    if (!ftmp[(size_t)ti].p) { why = "no memory for a temporary";
                                                               break; }
                                    ftmp[(size_t)ti].bytes = nb;
                                    rad_memset_async(ftmp[(size_t)ti].p, 0, nb, stream);
                                    RadTensor t{};
                                    t.data = ftmp[(size_t)ti].p;
                                    t.dtype = d.dtype;
                                    t.rank = (uint32_t)d.shape.size();
                                    for (size_t dd = 0; dd < d.shape.size() && dd < RAD_MAX_RANK;
                                         ++dd)
                                        t.shape[dd] = d.shape[dd];
                                    rad_tensor_pack(&t);
                                    ftmp_t[(size_t)ti] = t;
                                }
                                st_t[(size_t)oi] = ftmp_t[(size_t)ti];
                                continue;
                            }
                            if (src < 0 || (size_t)src >= sh.size()) {
                                why = "an operand index outside the fused kernel's list"; break;
                            }
                            if (sh[(size_t)src].absent) { st_t[(size_t)oi] = RadTensor{}; continue; }
                            /* A shared weight is in the fused kernel's layout (e.g. fragment order),
                             * so a step must use the same layout hook; a row-major reader such as
                             * `gemm_fp8a16_nt_m1` would misread it. Decline otherwise. */
                            if (o.schema->operands[(size_t)src].role == RAD_OPD_WEIGHT &&
                                srow->info->layout != r.row->info->layout) {
                                why = fmt("step '%s' resolved to %s, which reads '%s' in a "
                                          "different layout from the fused kernel", fs.op,
                                          srow->info->name,
                                          o.schema->operands[(size_t)src].name
                                              ? o.schema->operands[(size_t)src].name : "?");
                                break;
                            }
                            /* The fused kernel's tensor, pointed at the chain's own buffer for
                             * outputs and the shared one for inputs. */
                            RadTensor t = t_test[(size_t)src];
                            t.data = fdev[(size_t)src].p ? fdev[(size_t)src].p
                                                         : dev[(size_t)src].p;
                            if (!t.data) { why = "a step operand with no buffer"; break; }
                            st_t[(size_t)oi] = t;
                        }
                        if (!why.empty()) break;

                        RadArgs sa = case_args(st_t, sg);
                        if (srow->info->scratch) {
                            const int64_t need = srow->info->scratch(&sa);
                            if (need > fscratch.bytes) {
                                if (fscratch.p) rad_dev_free(fscratch.p, RAD_MEM_DEVICE);
                                fscratch.p = rad_dev_alloc(need, RAD_MEM_DEVICE);
                                fscratch.bytes = fscratch.p ? need : 0;
                            }
                            if (need > 0 && !fscratch.p) { why = "no scratch for a step"; break; }
                            if (need > 0) rad_memset_async(fscratch.p, 0, need, stream);
                            sa.scratch = fscratch.p; sa.scratch_bytes = fscratch.bytes;
                        }
                        const int sst = srow->info->launch(&sa, stream);
                        if (sst < 0) {
                            why = fmt("step '%s' (%s) refused: %s", fs.op, srow->info->name,
                                      rad_strerror(sst));
                            break;
                        }
                        if (rad_stream_sync(stream) < 0) { why = "a step faulted"; break; }
                    }

                    /* Copy the chain's outputs back and compare with the fused kernel's (in bufs[]). */
                    size_t n_cmp = 0, n_diff = 0;
                    std::string first_diff;
                    if (why.empty()) {
                        for (size_t k = 0; k < sh.size(); ++k) {
                            if (!fdev[k].p) continue;
                            fout[k].assign((size_t)fdev[k].bytes, 0);
                            rad_memcpy_async(fout[k].data(), fdev[k].p, fdev[k].bytes, stream);
                        }
                        if (rad_stream_sync(stream) < 0) why = "copy-back failed";
                    }
                    if (why.empty()) {
                        for (size_t k = 0; k < sh.size(); ++k) {
                            if (fout[k].empty() || fout[k].size() != bufs[k].size()) continue;
                            ++n_cmp;
                            if (std::memcmp(fout[k].data(), bufs[k].data(), fout[k].size()) != 0) {
                                ++n_diff;
                                if (first_diff.empty())
                                    first_diff = o.schema->operands[k].name
                                                     ? o.schema->operands[k].name : "?";
                            }
                        }
                        if (n_cmp == 0) why = "nothing comparable came back";
                    }

                    for (auto& d : fdev) if (d.p) rad_dev_free(d.p, RAD_MEM_DEVICE);
                    for (auto& d : ftmp) if (d.p) rad_dev_free(d.p, RAD_MEM_DEVICE);
                    if (fscratch.p) rad_dev_free(fscratch.p, RAD_MEM_DEVICE);

                    if (!why.empty()) {
                        fuse_skipped[o.op + "  (" + std::string(r.row->info->name) + ": " +
                                     why + ")"]++;
                    } else if (n_diff) {
                        ++fuse_failed;
                        std::printf("  %-22s %-40s fusion differs from the chain it declares: "
                                    "%zu of %zu operand(s), first '%s'  [%s]\n",
                                    o.op.c_str(), g.str().c_str(), n_diff, n_cmp,
                                    first_diff.c_str(), r.row->info->name);
                    } else {
                        ++fuse_passed;
                    }
                }


                /* The timing pass takes a passing device case's buffers over and times it later
                 * (kbench_timer.h); a host case is timed here, on its own copies of what it
                 * writes, so the comparison below still reads the checked launch's output. */
                int64_t timing_id = -1;
                kb::TimedCase host_case;
                bool host_pending = false;
                if (bench && s_test >= 0 && !record_only && !fuse_only && (!oracle || s_ref >= 0)) {
                    kb::TimedCase tc;
                    tc.info = r.row->info;
                    tc.g = g;
                    tc.seed = case_seed;
                    tc.what = o.op + " " + r.row->info->name + " " + g.str();
                    for (size_t k = 0; k < sh.size(); ++k) {
                        kb::TimedOperand d;
                        d.t = t_test[k];
                        d.role = o.schema->operands[k].role;
                        if (!sh[k].absent && t_test[k].data) {
                            d.bytes = dom == RAD_DOMAIN_DEVICE ? dev[k].bytes
                                                               : (int64_t)bufs[k].size();
                            if (d.role == RAD_OPD_INOUT && k < inout0.size() && !inout0[k].empty())
                                d.pristine = inout0[k].data();
                            if (d.role == RAD_OPD_IN)
                                kb::timing_input(kop[k], g,
                                                 kf_operand_seed(case_seed, (int)k) ^ kTimingSalt,
                                                 &d.variant);
                        }
                        tc.opd.push_back(std::move(d));
                    }
                    if (dom == RAD_DOMAIN_DEVICE) {
                        tc.scratch = dev_scratch.p;
                        tc.scratch_bytes = dev_scratch.bytes;
                        for (DevBuf& d : dev) {
                            if (!d.base || !d.owned) continue;
                            tc.frees.push_back(d.base);
                            tc.held += d.bytes + 2 * RED_BYTES;
                            d.owned = false;
                        }
                        if (dev_scratch.base) {
                            tc.frees.push_back(dev_scratch.base);
                            tc.held += dev_scratch.bytes + 2 * RED_BYTES;
                            dev_scratch.base = nullptr;
                        }
                        if (!timer) timer = std::make_unique<kb::DeviceTimer>(topts, stream, &ph);
                        timing_id = timer->add(std::move(tc));
                    } else {
                        tc.scratch = scratch_a.empty() ? nullptr : scratch_a.data();
                        tc.scratch_bytes = (int64_t)scratch_a.size();
                        host_case = std::move(tc);
                        host_pending = true;
                    }
                }
                /* A case is timed only if it passed: a host case after its verdict, a device
                 * case handed over now and dropped if the verdict fails it. */
                auto time_row = [&](Row& rw, bool passed) {
                    if (!passed) {
                        if (timing_id >= 0 && timer) timer->drop(timing_id);
                        return;
                    }
                    rw.timing_id = timing_id;
                    if (host_pending) {
                        PhaseTimer pt(&ph, kb::PH_TIMING);
                        apply_timing(rw, kb::time_host(host_case, topts, host_footprint, test_s));
                    }
                };

                /* Repoint the tensors at the host buffers (captured before timing), so the rest
                 * is domain-independent. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0 && !record_only)
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (dev[k].p) t_test[k].data = bufs[k].data();
                dev_release();

                /* A fusion-only case has no reference; the chain comparison was its verdict. */
                if (fuse_only) continue;

                if (s_ref < 0) {
                    std::printf("  %-22s %-40s oracle refused: %s\n", o.op.c_str(),
                                g.str().c_str(), rad_strerror(s_ref));
                    ++st.skipped;
                    continue;
                }
                if (s_test < 0) {
                    /* Refusing a geometry it was selected for is a failure, not a skip
                     * (spec §17). */
                    std::printf("  %-22s %-40s FAIL  %s/%s refused the geometry it was selected "
                                "for: %s\n", o.op.c_str(), g.str().c_str(),
                                r.row->plugin.c_str(), r.row->info->name, rad_strerror(s_test));
                    ++st.failed;
                    continue;
                }

                if (timing_only) {
                    ++st.timed;
                    Row rw;
                    rw.op = o.op;
                    rw.kernel = r.row->info->name;
                    rw.plugin = r.row->plugin;
                    rw.geom = g.str();
                    rw.domain = domain_name(dom);
                    rw.choice = r.choice;
                    rw.verdict = "timed";
                    time_row(rw, true);
                    rw.synthesised = !from_container;
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent) rw.bytes += (int64_t)bufs[k].size();
                    report_rows.push_back(std::move(rw));
                    continue;
                }

                /* Compare every OUT and INOUT operand and name the worst; which output is wrong is
                 * most of the diagnosis. */
                PhaseTimer cmp_t(&ph, kb::PH_COMPARE);
                Err worst;
                /* Non-finite mismatches are summed over operands; they are not in rel_l2, so taking
                 * them from the worst operand alone could drop them. */
                int64_t nonfinite = 0;
                /* Each operand is judged against its own tolerance; see takes_worst. */
                double worst_tol = tol_override >= 0
                                       ? tol_override
                                       : kb::tolerance_for(o.op, g, nullptr).rel_l2;
                bool worst_bad = false, all_ok = true;

                std::string per_opd;
                int n_out = 0;
                for (size_t k = 0; k < sh.size(); ++k) {
                    const int role = o.schema->operands[k].role;
                    if (sh[k].absent) continue;
                    if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                    const int64_t n = rad_numel(sh[k].shape);
                    std::vector<float> a, bref;
                    rad_widen(bufs[k].data(), sh[k].dtype, n, &a);
                    rad_widen(ref_out[k].data(), sh[k].dtype, n, &bref);
                    const Err e = kb::compare(a.data(), bref.data(), n);
                    const char* nm = o.schema->operands[k].name ? o.schema->operands[k].name : "?";
                    per_opd += fmt("%s%s=%.3e", per_opd.empty() ? "" : " ", nm, e.rel_l2);
                    if (e.n_nonfinite_mismatch)
                        per_opd += fmt("(%lld non-finite)", (long long)e.n_nonfinite_mismatch);
                    nonfinite += e.n_nonfinite_mismatch;
                    const double tk = tol_override >= 0
                                          ? tol_override
                                          : kb::tolerance_for(o.op, g, nm).rel_l2;
                    /* Ids are right or wrong (kbench_core.h). Where the op lets a tie go either
                     * way, an id indexing a value equal to the reference id's is accepted. */
                    int64_t id_wrong = 0;
                    if (kb::is_index_dtype(sh[k].dtype)) {
                        kb::TieView tv;
                        const kb::TieView* tie = nullptr;
                        if (const kb::TieRule* rule = kb::tie_rule(o.op, nm)) {
                            for (size_t j = 0; j < sh.size(); ++j) {
                                const char* jn = o.schema->operands[j].name;
                                if (!jn || std::strcmp(jn, rule->input) != 0 || sh[j].absent ||
                                    sh[j].shape.size() < 2)
                                    continue;
                                tv.in = ref_out[j].data();
                                tv.in_dtype = sh[j].dtype;
                                tv.in_rows = sh[j].shape[0];
                                tv.in_width = rad_numel(sh[j].shape) / sh[j].shape[0];
                                tv.out_width = sh[k].shape.empty() || sh[k].shape[0] <= 0
                                                   ? 1 : n / sh[k].shape[0];
                                tv.lane_stride = rule->lane_stride;
                                long long off = 0;
                                if (g.get_i("vocab_off", &off)) tv.id_base = off;
                                tie = &tv;
                                break;
                            }
                        }
                        for (int64_t i = 0; i < n; ++i)
                            if (!kb::index_matches(tie, i, (int64_t)a[(size_t)i],
                                                   (int64_t)bref[(size_t)i]))
                                ++id_wrong;
                        if (id_wrong) per_opd += fmt("(%lld id(s) wrong)", (long long)id_wrong);
                    }
                    /* Written so NaN fails (`rel > tol || rel < 0` would pass it). */
                    const bool bad = kb::is_index_dtype(sh[k].dtype)
                                         ? id_wrong != 0
                                         : !(e.rel_l2 >= 0 && e.rel_l2 <= tk);
                    all_ok = all_ok && !bad;
                    if (takes_worst(e.rel_l2, bad, worst.rel_l2, worst_bad)) {
                        worst = e;
                        worst_tol = tk;
                        worst_bad = bad;
                    }
                    ++n_out;
                    /* When recording, reduce the oracle's widened output into the fixture here,
                     * while it is in hand. */
                    if (record_here) {
                        KfOperand& d = kop[k];
                        d.have_ref = true;
                        d.ref_n    = n;
                        kf_reduce(bref.data(), n, kf_output_seed(case_seed, (int)k),
                                  &d.ref_sumsq, &d.ref_nonfinite, &d.sample);
                        kf_sketch(bref.data(), n, kf_output_seed(case_seed, (int)k), &d.sketch);
                    }
                }

                if (n_out == 0) {
                    if (timing_id >= 0 && timer) timer->drop(timing_id);
                    ++st.skipped;
                    continue;
                }
                worst.n_nonfinite_mismatch = nonfinite;

                const double tol = worst_tol;
                const bool ok = worst.n_nonfinite_mismatch == 0 && all_ok;

                if (!record_only) {
                    ++st.checked;
                    st.worst = kb::worst_of(st.worst, worst.rel_l2);
                    if (ok) ++st.passed; else ++st.failed;
                }

                if (!record_only) {
                    Row rw;
                    rw.op = o.op;
                    rw.kernel = r.row->info->name;
                    rw.plugin = r.row->plugin;
                    rw.geom = g.str();
                    rw.domain = domain_name(dom);
                    rw.choice = r.choice;
                    rw.verdict = ok ? "ok" : "**FAIL**";
                    rw.rel_l2 = worst.rel_l2;
                    rw.tol = tol;
                    time_row(rw, ok);
                    rw.synthesised = !from_container;
                    /* Total bytes of the operands given to one launch (not a FLOP count, not a
                     * bytes-read figure). */
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent) rw.bytes += (int64_t)bufs[k].size();
                    report_rows.push_back(std::move(rw));
                }
                if (record_here) {
                    point_recorded = true;
                    {
                        KfCase c;
                        c.op   = o.op;
                        c.seed = case_seed;
                        c.m    = M;
                        c.band = o.bands[b].span;
                        c.opd  = kop;
                        /* Stored typed (see KfParam). */
                        for (int pi2 = 0; pi2 < g.n_params(); ++pi2) {
                            const RadParam& sp = g.params()[pi2];
                            if (!sp.key) continue;
                            KfParam kp;
                            kp.key  = sp.key;
                            kp.kind = sp.kind;
                            if (sp.kind == RAD_P_INT)      kp.i = sp.ival;
                            else if (sp.kind == RAD_P_F64) kp.d = sp.dval;
                            else if (sp.kind == RAD_P_STR) kp.s = sp.sval ? sp.sval : "";
                            else continue;   /* a RANGE is not a point and cannot be replayed */
                            c.params.push_back(std::move(kp));
                        }
                        fixture.cases.push_back(std::move(c));
                    }
                }

                if (!record_only && (!ok || verbose)) {
                    std::printf("  %-22s %-44s %-8s %-16s rel_l2=%.3e max_abs=%.3e max_rel=%.3e "
                                "tol=%.1e %s%s\n",
                                o.op.c_str(), g.str().c_str(), domain_name(dom),
                                r.row->info->name, worst.rel_l2, worst.max_abs, worst.max_rel,
                                tol, ok ? "ok" : "FAIL",
                                from_container ? "" : "  [weights synthesised: not in the "
                                                      "container]");
                    if (n_out > 1)
                        std::printf("      per operand: %s\n", per_opd.c_str());
                    if (worst.n_nonfinite_mismatch)
                        std::printf("      %lld element(s) are finite on one side and not the "
                                    "other (a failure at any tolerance)\n",
                                    (long long)worst.n_nonfinite_mismatch);
                }
            }
        }
    }

    /* The fusion check's own tally (byte identity, a different claim from tolerance). */
    if (fuse_check && (fuse_passed || fuse_failed || !fuse_skipped.empty())) {
        std::printf("\nfusions: %d byte-identical to their declared chain, %d differ\n",
                    fuse_passed, fuse_failed);
        if (!fuse_skipped.empty()) {
            std::printf("  not checked\n");
            for (const auto& [what, n] : fuse_skipped)
                std::printf("    %-70s x%d\n", what.c_str(), n);
        }
    }

    st.fuse_failed = fuse_failed;
    BenchInfo binfo;
    if (bench) binfo = finish_timing(timer.get(), report_rows, host_footprint, topts.footprint >= 0);
    timer.reset();
    print_summary(st, timing_only, nm);
    print_row_coverage(census(reg, ref_name), report_rows);
    if (st.failed == 0 && st.checked > 0)
        std::printf("every resolved kernel agrees with %s on this container's bytes\n",
                    ref_name.c_str());
    if (st.checked == 0 && !timing_only)
        std::printf("Nothing was checked: every op resolved to %s, or no op has a shape "
                    "recipe.\n", ref_name.c_str());
    if (!report_path.empty()) {
        /* The distinct ops this container's program declares (prog.ops repeats per layer). */
        std::vector<std::string> graph_ops;
        {
            std::set<std::string> seen;
            for (size_t oi = 1; oi < prog.ops.size(); ++oi)
                if (prog.ops[oi].schema) seen.insert(prog.ops[oi].op);
            graph_ops.assign(seen.begin(), seen.end());
        }
        if (write_report(report_path, report_rows, skipped_ops, fuse_skipped, diverged, overruns,
                         analysis_path,
                         "container: `" + model + "` (weights read from it, in the layout each\n  resolved kernel asked for)",
                         fuse_passed, fuse_failed, st, ref_name,
                         plugins.libraries(), plugins.arch_plugin, graph_ops,
                         census(reg, ref_name), bench ? &binfo : nullptr) >= 0)
            std::printf("report written to %s (%zu row(s))\n", report_path.c_str(),
                        report_rows.size());
    }
    if (recording) {
        fixture.recorded = today();
        const int rc = fixture.save(record_path);
        if (rc < 0) { RAD_ERR("--record %s: %s", record_path.c_str(), rad_strerror(rc)); return 1; }
        std::printf("recorded %zu case(s) to %s\n", fixture.cases.size(), record_path.c_str());
    }
    std::printf("%s\n", kb::phase_line(ph, std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - pass_t0)
                                               .count()).c_str());
    /* An overrun or a fusion that differs from its chain is a failure as much as a wrong
     * number. */
    return kb::exit_code(st, false);
}
