/* kbench_core.cpp -- rad-kbench's rules (kbench_core.h). */
#include "kbench_core.h"
#include "format/radfile.h"
#include "rad_format.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {
namespace kb {

/* ================================================================== elementwise comparison */
Err compare(const float* got, const float* ref, int64_t n) {
    Err e;
    e.n = n;
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        const bool fg = std::isfinite(got[i]), fr = std::isfinite(ref[i]);
        if (fg != fr) { ++e.n_nonfinite_mismatch; continue; }
        if (!fg) continue;
        const double d = (double)got[i] - (double)ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        const double scale = std::fabs((double)ref[i]);
        if (scale > 1e-30) e.max_rel = std::max(e.max_rel, std::fabs(d) / scale);
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    e.rel_l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    return e;
}

bool takes_worst(double rel, bool bad, double worst_rel, bool worst_bad) {
    if (std::isnan(worst_rel)) return false;
    return (bad && !worst_bad) || (bad == worst_bad && !(rel <= worst_rel));
}

double worst_of(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::nan("");
    return a > b ? a : b;
}

/* ================================================================== tolerances */
namespace {

/* Most specific first. The lint in tests/kbench_test.cpp walks every op of both fixtures through
 * this table and refuses one that lands on the catch-all, so a new op gets a row of its own. */
const std::vector<TolRow> g_tol = {
    /* Reductions over K: the error grows with the reduction length and the tiling order. The
     * gated GEMM also writes an E4M3 twin of its result, which a rounding difference in the GEMM
     * moves by whole code steps (see g_opd_tol). */
    { "gemm_nt_q_gated", "*",     3e-2 },
    { "gemm_nt",      "bf16",     8e-3 },
    { "gemm_nt",      "f16",      4e-3 },
    { "gemm_nt",      "f32",      2e-5 },
    /* Both sides read the same E4M3 codes and scales; only the summation order differs. */
    { "gemm_nt",      "fp8a8",    5e-3 },
    { "gemm_nt",      "*",        3e-2 },   /* w4a8, w4a16, mxfp4a8: the quant dominates */
    { "logits_gemm",  "bf16",     8e-3 },
    { "logits_gemm",  "f32",      2e-5 },
    /* A dot product per candidate against the bf16 head: a GEMM's error. */
    { "logit_rerank", "*",        8e-3 },
    { "moe_gemm",     "*",        3e-2 },
    /* Runtime LoRA's shrink is a K-long reduction into an f32 `t` whatever `a` is. The codes are
     * decoded alike on both sides, so only the summation order differs (and libr4d's WMMA row
     * scales each 32-wide step's partial rather than each element): gemm_nt's f32 tolerance.
     * Expand's R-long sum lands in `y` and takes the dtype rows below; libr4d's tiled expand
     * computes libref's bytes. */
    { "lora_shrink",  "*",        2e-5 },
    /* Attention: a softmax over a long context plus a second reduction. */
    { "attn_",        "bf16",     2e-2 },
    { "attn_",        "f32",      1e-4 },
    { "attn_",        "*",        3e-2 },
    /* The table is ids, counts and offsets, compared exactly; the masked read is attention over the
     * same pages in the same order, so it takes attention's tolerance. */
    { "qsa_table",    "*",        0.0  },
    { "qsa_attn",     "bf16",     2e-2 },
    { "qsa_attn",     "f32",      1e-4 },
    { "qsa_attn",     "*",        3e-2 },
    /* Elementwise and norms: one pass, so anything but a rounding difference is a bug. */
    { "rmsnorm",      "bf16",     4e-3 },
    { "rmsnorm",      "f32",      1e-6 },
    { "layernorm",    "f32",      1e-6 },
    { "softmax",      "f32",      1e-6 },
    /* Two implementations of the same quantiser must agree, so this is tight. */
    { "quant_act",    "*",        1e-6 },
    { "dequant",      "*",        1e-6 },
    { "cast",         "*",        1e-6 },
    /* Copies, and stores that round each element once the same way: bit-exact or broken. A
     * top-k's values and pairs are copies of its input. */
    { "gather_rows",  "*",        0.0  },
    { "scatter_rows", "*",        0.0  },
    { "embed_lookup", "*",        0.0  },
    { "kv_store",     "*",        0.0  },
    { "row_topk",     "*",        0.0  },
    { "pair_argmax",  "*",        0.0  },
    { "all_gather",   "*",        0.0  },
    /* Rotation tables in f32, and the rotation itself on bf16 rows. */
    { "rope_table",   "*",        1e-5 },
    { "rope",         "*",        2e-3 },
    /* Gated delta net. The chunked scan accumulates its state across chunks and is the loosest;
     * the solve, the convolution and the norm are one pass each. */
    { "gdn_chunk_scan",    "*",   2e-2 },
    { "gdn_kkt_solve",     "*",   2e-3 },
    { "gdn_conv_update",   "*",   2e-3 },
    { "gdn_gated_rmsnorm", "*",   2e-3 },
    { "gdn_",              "*",   2e-2 },
    /* Sampler stages on f32 logits: elementwise, or copies with a decision. sample_logprobs sums
     * exp() over the vocabulary, which a different order moves in the last bits. */
    { "sample_logprobs", "*",     1e-5 },
    { "sample_",      "*",        1e-6 },
    /* Collectives. */
    { "all_reduce",   "f32",      1e-6 },
    { "all_reduce",   "*",        4e-3 },
    { "*",            "f32",      1e-5 },
    { "*",            "bf16",     6e-3 },
    { "*",            "f16",      3e-3 },
    { "*",            "*",        2e-2 },
};

/* Per-operand overrides, consulted first, for outputs derived from another output.
 *
 * The fp8 twins of a computed row: E4M3 keeps three mantissa bits, so a rounding difference d in
 * the bf16 input moves about d / 2^-3 of the codes by one step of about 2^-3, giving rel_l2 near
 * sqrt(d / 8): 2.7e-2 at the row's 6e-3 tolerance, 1e-2 at a typical 1e-3. A twin quantised from
 * wrong values or with the wrong rotation lands near 1. */
struct OpdTol { const char* op; const char* operand; double rel_l2; };
const OpdTol g_opd_tol[] = {
    { "hc_read", "q",  3e-2 },
    { "hc_read", "rq", 3e-2 },
};

}  /* namespace */

const std::vector<TolRow>& tolerance_table() { return g_tol; }

const char* activation_dtype(const Geometry& g) {
    for (const char* k : { "dtype", "q_dtype", "kv_dtype" })
        if (const char* v = g.get_s(k)) return v;
    return nullptr;
}

TolPick tolerance_for(const std::string& op, const Geometry& g, const char* operand) {
    TolPick p;
    if (operand)
        for (const OpdTol& t : g_opd_tol)
            if (op == t.op && std::strcmp(operand, t.operand) == 0) {
                p.rel_l2 = t.rel_l2;
                return p;
            }
    const char* dt = activation_dtype(g);
    if (!dt) dt = "*";
    for (const TolRow& t : g_tol) {
        const bool op_match = std::strcmp(t.op, "*") == 0 || op.rfind(t.op, 0) == 0;
        const bool dt_match = std::strcmp(t.dtype, "*") == 0 || std::strcmp(t.dtype, dt) == 0;
        if (op_match && dt_match) {
            p.rel_l2 = t.rel_l2;
            p.row = &t;
            p.catch_all = !std::strcmp(t.op, "*") && !std::strcmp(t.dtype, "*");
            return p;
        }
    }
    p.rel_l2 = 2e-2;
    p.catch_all = true;
    return p;
}

/* ================================================================== index outputs */
bool is_index_dtype(uint32_t dt) {
    return dt == RAD_I16 || dt == RAD_I32 || dt == RAD_I64 || dt == RAD_U32 || dt == RAD_U16 ||
           dt == RAD_BOOL;
}

namespace {

/* The ops whose contract leaves the winner of a tie open. Every other index output is exact:
 * row_topk, row_topk_merge, pair_argmax, logit_rerank and sample_logprobs state "ties to the
 * lower id" in docs/OPS.md. */
const TieRule g_ties[] = {
    { "sample_argmax", "token",      "logits", 1 },
    { "router_topk",   "expert_ids", "logits", 1 },
    { "sample_topk",   "cand_idx",   "logits", 1 },
    { "sample_topk",   "pairs",      "logits", 2 },
};

}  /* namespace */

const TieRule* tie_rule(const std::string& op, const char* output) {
    if (!output) return nullptr;
    for (const TieRule& t : g_ties)
        if (op == t.op && std::strcmp(output, t.output) == 0) return &t;
    return nullptr;
}

bool index_matches(const TieView* tie, int64_t pos, int64_t got, int64_t want) {
    if (got == want) return true;
    if (!tie || !tie->in || tie->in_width <= 0 || tie->out_width <= 0) return false;
    const int ls = tie->lane_stride > 0 ? tie->lane_stride : 1;
    if (pos % ls != 0) return false;               /* a value lane: equal or wrong */
    const int64_t row = pos / tie->out_width;
    if (row < 0 || row >= tie->in_rows) return false;
    const int64_t g = got - tie->id_base, w = want - tie->id_base;
    if (g < 0 || w < 0 || g >= tie->in_width || w >= tie->in_width) return false;
    const float vg = rad_load_f32(tie->in, tie->in_dtype, row * tie->in_width + g);
    const float vw = rad_load_f32(tie->in, tie->in_dtype, row * tie->in_width + w);
    return vg == vw;
}

/* ================================================================== the replay's verdict */
void digest_of(const float* v, int64_t n, uint64_t seed, bool sketch, Digest* out) {
    out->n = n;
    kf_reduce(v, n, seed, &out->sumsq, &out->nonfinite, &out->sample);
    if (sketch) kf_sketch(v, n, seed, &out->sketch);
    else out->sketch.clear();
}

Digest digest_of(const KfOperand& rec) {
    Digest d;
    d.n = rec.ref_n;
    d.sumsq = rec.ref_sumsq;
    d.nonfinite = rec.ref_nonfinite;
    d.sample = rec.sample;
    d.sketch = rec.sketch;
    return d;
}

namespace {

/* The sketch's estimate of the full-tensor rel_l2: each projection of the difference has the
 * difference's squared norm as its expectation. -1 when either side has no sketch. */
double sketch_rel(const std::vector<double>& mine, const std::vector<double>& ref,
                  double ref_sumsq) {
    if (mine.size() != (size_t)KF_SKETCH || ref.size() != (size_t)KF_SKETCH) return -1;
    double dn = 0;
    for (int j = 0; j < KF_SKETCH; ++j) {
        const double d = mine[(size_t)j] - ref[(size_t)j];
        dn += d * d;
    }
    if (!(ref_sumsq > 0)) return dn > 0 ? 1.0 : 0.0;
    return std::sqrt(dn / KF_SKETCH) / std::sqrt(ref_sumsq);
}

/* The norm bound for an index output whose op allows ties: a tie at an unsampled position moves
 * the norm by one id's difference, a wrong kernel by far more. */
constexpr double kTieNormTol = 1e-3;

}  /* namespace */

OpdVerdict judge_operand(const KfOperand& rec, const Digest& mine, uint64_t pos_seed, double tol,
                         bool exact, const TieView* tie) {
    OpdVerdict v;
    v.index = is_index_dtype(rec.dtype);
    std::vector<int64_t> pos;
    if (v.index) kf_sample_positions(pos_seed, rec.ref_n, &pos);

    double num = 0, den = 0;
    const size_t ns = std::min(mine.sample.size(), rec.sample.size());
    for (size_t i = 0; i < ns; ++i) {
        const float a = mine.sample[i], b = rec.sample[i];
        const bool fa = std::isfinite(a), fb = std::isfinite(b);
        if (fa != fb) { ++v.nonfinite_gap; continue; }
        if (!fa) {
            /* Both non-finite: a NaN where the reference has an infinity, or the opposite sign,
             * is still a different value. */
            if (std::isnan(a) != std::isnan(b) || (!std::isnan(a) && a != b)) ++v.nonfinite_gap;
            continue;
        }
        const double diff = (double)a - (double)b;
        v.max_abs = std::max(v.max_abs, std::fabs(diff));
        const double sc = std::fabs((double)b);
        if (sc > 1e-30) v.max_rel = std::max(v.max_rel, std::fabs(diff) / sc);
        num += diff * diff;
        den += (double)b * (double)b;
        if (v.index && diff != 0.0) {
            const int64_t p = i < pos.size() ? pos[i] : (int64_t)i;
            if (index_matches(tie, p, (int64_t)a, (int64_t)b)) ++v.ties;
            else ++v.index_mismatch;
        }
    }
    if (mine.sample.size() != rec.sample.size()) v.nonfinite_gap += 1;
    v.sample_rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    v.norm_rel = rec.ref_sumsq > 0
                     ? std::fabs(std::sqrt(mine.sumsq) - std::sqrt(rec.ref_sumsq)) /
                           std::sqrt(rec.ref_sumsq)
                     : (mine.sumsq > 0 ? 1.0 : 0.0);
    v.nonfinite_gap += std::llabs((long long)(mine.nonfinite - rec.ref_nonfinite));
    const bool ties_allowed = v.index && tie != nullptr;
    if (!ties_allowed) v.sketch_rel = sketch_rel(mine.sketch, rec.sketch, rec.ref_sumsq);
    v.rel = std::max(v.sample_rel, std::max(v.norm_rel, v.sketch_rel));
    if (std::isnan(v.sample_rel) || std::isnan(v.norm_rel) || std::isnan(v.sketch_rel))
        v.rel = std::nan("");

    v.vacuous = rec.ref_sumsq == 0 && rec.ref_nonfinite == 0 && mine.sumsq == 0 &&
                mine.nonfinite == 0;

    if (v.index) {
        /* Ids are right or wrong. Without a tie rule the norm and the sketch are exact too, since
         * both sides reduce identical values in one order; with one, a tie at an unsampled
         * position may move the norm by one id. */
        bool ok = v.index_mismatch == 0 && v.nonfinite_gap == 0;
        if (ties_allowed) ok = ok && v.norm_rel <= kTieNormTol;
        else ok = ok && mine.sumsq == rec.ref_sumsq && !(v.sketch_rel > 0);
        v.bad = !ok;
        return v;
    }
    const double t = exact ? 0.0 : tol;
    /* Written so NaN fails (`rel > tol || rel < 0` would pass it). */
    v.bad = !(v.rel >= 0 && v.rel <= t) || v.nonfinite_gap != 0;
    return v;
}

/* ================================================================== keys */
std::string cap_key(const KfCase& c, const std::string& kernel, const std::string& lib, int dom,
                    const Geometry& g) {
    std::string k = c.op + "\x1f" + kernel + "\x1f" + lib + "\x1f" + std::to_string(dom) +
                    "\x1f" + g.str();
    for (const KfOperand& d : c.opd) {
        k += "\x1f";
        if (d.absent) { k += "-"; continue; }
        k += std::to_string(d.dtype) + ":";
        for (int64_t x : d.shape) k += std::to_string(x) + ",";
    }
    return k;
}

uint64_t host_weight_key(const KfOperand& d, size_t k) {
    const uint64_t wk = kf_weight_seed(d, k);
    uint64_t ck = wk;
    if (d.has_enc) {
        const uint8_t* e = (const uint8_t*)&d.enc;
        for (size_t i = 0; i < sizeof d.enc; ++i) ck = (ck ^ e[i]) * 0x100000001b3ull;
        for (int s : d.sel) ck = (ck ^ (uint64_t)(uint32_t)s) * 0x100000001b3ull;
        uint64_t mix = ck ^ 0x5bd1e995ull;
        ck = rad_splitmix(mix);
    }
    return ck;
}

std::string device_weight_key(const std::string& lib, const std::string& kernel,
                              const KfOperand& d, size_t k, size_t bytes) {
    return lib + "\x1f" + kernel + "\x1f" + std::to_string(host_weight_key(d, k)) + "\x1f" +
           std::to_string(bytes);
}

/* ================================================================== memory */
int64_t case_bytes(const std::vector<KfOperand>& opd) {
    int64_t want = 0;
    for (const KfOperand& d : opd) {
        if (d.absent) continue;
        int64_t n = 1;
        for (int64_t x : d.shape) {
            if (x <= 0 || n > INT64_MAX / x) return -1;
            n *= x;
        }
        const int bits = rad_dtype_bits(d.dtype);
        if (bits <= 0 || n > INT64_MAX / 3 / bits) return -1;
        const int64_t b = rad_dtype_bytes(d.dtype, n);
        if (want > INT64_MAX - 3 * b) return -1;
        want += 3 * b;
    }
    return want;
}

int64_t default_case_budget(const std::string& meminfo, int64_t* avail) {
    int64_t kb = -1;
    const size_t at = meminfo.find("MemAvailable:");
    if (at != std::string::npos) {
        const char* p = meminfo.c_str() + at + std::strlen("MemAvailable:");
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end != p && v >= 0) kb = v;
    }
    const int64_t bytes = kb >= 0 ? kb * 1024 : 0;
    if (avail) *avail = bytes;
    int64_t want = bytes > 0 ? bytes / 2 : ((int64_t)8 << 30);
    want = std::min<int64_t>(want, (int64_t)8 << 30);
    want = std::max<int64_t>(want, (int64_t)512 << 20);
    return want;
}

/* ================================================================== red zones */
Overrun scan_margins(const uint8_t* lo, const uint8_t* hi, int64_t red_bytes, uint8_t fill) {
    Overrun o;
    for (int64_t i = 0; i < red_bytes; ++i) {
        if (lo[i] != fill) o.before = std::max(o.before, red_bytes - i);
        if (hi[i] != fill) o.after  = std::max(o.after, i + 1);
    }
    return o;
}

/* ================================================================== what a run found */
std::string bench_refusal(const Stats& st, const std::set<std::string>& measured,
                          const std::map<std::string, int>& skipped,
                          const std::map<std::string, int>& stale) {
    if (measured.empty())
        return "no op of this run has a time to report (nothing was checked, or --timing-only was "
               "left off and every case was skipped)";
    if (st.checked == 0) return "no case of this run was checked against a reference";
    /* A gap line is named `op  (why)`; the op is what the gate can attribute it to. */
    auto op_of = [](const std::string& key) {
        const size_t at = key.find("  ");
        return at == std::string::npos ? key : key.substr(0, at);
    };
    std::set<std::string> ops;
    int n = 0;
    for (const std::map<std::string, int>* m : { &skipped, &stale })
        for (const auto& [key, count] : *m) {
            const std::string op = op_of(key);
            if (!measured.count(op)) continue;
            ops.insert(op);
            n += count;
        }
    if (ops.empty()) return std::string();
    std::string names;
    for (const std::string& o : ops) names += (names.empty() ? "" : ", ") + o;
    return fmt("%d case(s) of %s were skipped or are stale, so %s not fully checked and %s time "
               "cannot stand for the op", n, names.c_str(), ops.size() == 1 ? "it is" : "they are",
               ops.size() == 1 ? "its" : "their");
}

/* The stale decision, kept out of the replay so it can be held to a case no plugin set here can
 * build: a library that names itself as the recording and has no host row for the op. The tool
 * reaches it only when a plugin's rows disappear between recording and replay, which is what makes
 * it worth stating rather than assuming. */
std::string stale_reason(const std::string& recorded, const std::string& oracle, bool oracle_row,
                         const std::string& differs) {
    if (recorded.empty()) return std::string();   /* no recorded library: `--ref` answers */
    if (oracle.empty())
        return recorded + " recorded this case and has no row for it now, and no loaded library "
                          "declares a reference for it";
    if (oracle != recorded) return std::string();  /* a substitute, not a stale case */
    if (!oracle_row) return oracle + " recorded this case and has no row for it now";
    if (!differs.empty())
        return oracle + " now describes an output differently from its recording -- " + differs;
    return std::string();
}

int exit_code(const Stats& st, bool strict) {
    if (st.failed || st.overrun || st.fuse_failed || st.input_written) return 1;
    if (strict && st.stale) return 1;
    return 0;
}

/* ================================================================== where the time went */
double PhaseTimer::stop() {
    if (done_) return 0;
    done_ = true;
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    if (ph_) ph_->add(p_, s);
    return s;
}

std::string phase_line(const Phases& ph, double wall_s) {
    static const char* names[PH_N] = { "load", "draw", "hash", "layout", "upload", "launch",
                                       "readback", "compare", "cache", "timing", "report" };
    std::string s = "time:";
    bool first = true;
    for (int p = 0; p < PH_N; ++p) {
        if (ph.s[p] < 0.0005 && p != PH_LAUNCH) continue;
        s += fmt("%s %s %.2f s", first ? "" : ",", names[p], ph.s[p]);
        first = false;
        if (p == PH_LAUNCH && !ph.launch_by_lib.empty()) {
            s += " (";
            bool f2 = true;
            for (const auto& [lib, t] : ph.launch_by_lib) {
                s += fmt("%s%s %.2f s", f2 ? "" : ", ", lib.c_str(), t);
                f2 = false;
            }
            s += ")";
        }
    }
    s += fmt("; %.2f s wall", wall_s);
    return s;
}

/* ================================================================== timing */
TimeStats time_stats(std::vector<double> v) {
    TimeStats s;
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }),
            v.end());
    const int64_t n = (int64_t)v.size();
    s.n = (int)n;
    if (n == 0) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    s.median = n % 2 ? v[(size_t)(n / 2)] : 0.5 * (v[(size_t)(n / 2 - 1)] + v[(size_t)(n / 2)]);
    /* ceil(0.9 n) in integers: 0.9 * 20 is 18.000000000000004 in doubles. */
    s.p90 = v[(size_t)((9 * n + 9) / 10 - 1)];
    if (n < 2) return s;
    const double h = 0.98 * std::sqrt((double)n);
    const int64_t lo = std::max<int64_t>(1, (int64_t)std::floor((double)n / 2.0 - h));
    const int64_t hi = std::min<int64_t>(n, (int64_t)std::ceil((double)n / 2.0 + 1.0 + h));
    s.ci_lo = v[(size_t)(lo - 1)];
    s.ci_hi = v[(size_t)(hi - 1)];
    if (s.median > 0) s.ci_pct = 100.0 * (s.ci_hi - s.ci_lo) / 2.0 / s.median;
    return s;
}

bool timing_done(const std::vector<double>& v, int min_n, int max_n, double ci_pct) {
    const int n = (int)v.size();
    if (n >= max_n) return true;
    if (n < min_n) return false;
    const TimeStats s = time_stats(v);
    return s.ci_pct >= 0 && s.ci_pct <= ci_pct;
}

double batch_target_us(double floor_us) {
    return std::max(1000.0, 50.0 * (floor_us > 0 ? floor_us : 0.0));
}

std::string timing_caveat(double median_us, double floor_us, double restore_pct) {
    char b[160];
    if (median_us > 0 && floor_us > 0 && median_us < 0.95 * floor_us) {
        std::snprintf(b, sizeof b, "under the launch floor (%.2f us): an order of magnitude",
                      floor_us);
        return b;
    }
    if (restore_pct >= 75.0) {
        std::snprintf(b, sizeof b, "restore-dominated (%.0f%% of the batch): an order of magnitude",
                      restore_pct);
        return b;
    }
    return "";
}

int64_t batch_launches(double est_us, double target_us, int64_t room, int64_t fixed) {
    /* A host-side bound when nothing else limits the batch (a launch that runs on the calling
     * thread has no queue to fill). */
    constexpr int64_t kMaxBatch = 4096;
    int64_t n = 1;
    if (fixed > 0) n = fixed;
    else if (est_us > 0 && target_us > 0) n = (int64_t)std::ceil(target_us / est_us);
    n = std::min(std::max<int64_t>(n, 1), kMaxBatch);
    if (room >= 1) n = std::min(n, room);
    return n;
}

int64_t rotation_stride(int64_t bytes) {
    return bytes <= 0 ? 0 : (bytes + 255) & ~(int64_t)255;
}

int64_t rotation_copies(int64_t per_launch, int64_t footprint) {
    if (footprint <= 0 || per_launch <= 0 || per_launch >= footprint) return 1;
    /* R copies leave R - 1 others between two uses of one: (R - 1) * per_launch >= footprint. */
    return (footprint + per_launch - 1) / per_launch + 1;
}

int64_t default_footprint(int64_t cache_bytes) {
    return std::max<int64_t>((int64_t)256 << 20, 2 * std::max<int64_t>(cache_bytes, 0));
}

bool rotated(int role, int64_t bytes, int64_t big_input) {
    if (role == RAD_OPD_WEIGHT || role == RAD_OPD_WTAB) return true;
    return role == RAD_OPD_IN && big_input > 0 && bytes >= big_input;
}

namespace {

/* A double in [0, 1) from the stream. */
double unit(uint64_t& s) { return (double)(rad_splitmix(s) >> 11) * 0x1.0p-53; }

/* A seeded order of [0, n): Fisher-Yates. */
std::vector<int64_t> seeded_order(int64_t n, uint64_t seed) {
    std::vector<int64_t> o((size_t)std::max<int64_t>(n, 0));
    for (int64_t i = 0; i < n; ++i) o[(size_t)i] = i;
    uint64_t s = seed;
    for (int64_t i = n - 1; i > 0; --i)
        std::swap(o[(size_t)i], o[(size_t)(rad_splitmix(s) % (uint64_t)(i + 1))]);
    return o;
}

/* The skew of skewed_counts: Zipf with this exponent over the ranks. */
constexpr double kRouteSkew = 0.5;

/* Each expert's weight, rank r (from 1) getting r^-kRouteSkew, ranks in a seeded order. */
std::vector<double> route_weights(int64_t n, uint64_t seed) {
    const std::vector<int64_t> order = seeded_order(n, seed);
    std::vector<double> w((size_t)n, 0.0);
    for (int64_t r = 0; r < n; ++r)
        w[(size_t)order[(size_t)r]] = std::pow((double)(r + 1), -kRouteSkew);
    return w;
}

}  /* namespace */

void table_permutation(int64_t n, uint64_t seed, int64_t l, std::vector<int64_t>* out) {
    *out = seeded_order(n, rad_splitmix(seed) ^ (0xD6E8FEB86659FD93ull * (uint64_t)(l + 1)));
}

int64_t table_count(int64_t copies) {
    constexpr int64_t kTables = 64;
    if (copies < 1) copies = 1;
    return copies * ((kTables + copies - 1) / copies);
}

void table_pointers(const std::vector<uint64_t>& bases, int64_t n, int64_t entry, uint64_t seed,
                    int64_t l, std::vector<uint64_t>* out) {
    if (bases.empty() || n <= 0) return;
    std::vector<int64_t> perm;
    table_permutation(n, seed, l, &perm);
    const uint64_t base = bases[(size_t)(l % (int64_t)bases.size())];
    for (int64_t e = 0; e < n; ++e) out->push_back(base + (uint64_t)(perm[(size_t)e] * entry));
}

std::vector<int64_t> skewed_counts(int64_t buckets, int64_t total, int64_t cap, uint64_t seed) {
    std::vector<int64_t> c((size_t)std::max<int64_t>(buckets, 0), 0);
    if (buckets <= 0 || total <= 0) return c;
    if (cap <= 0 || cap > total || cap * buckets < total) cap = total;
    const std::vector<int64_t> order = seeded_order(buckets, seed);
    double sum = 0;
    for (int64_t r = 0; r < buckets; ++r) sum += std::pow((double)(r + 1), -kRouteSkew);
    /* Largest remainder, by rank: hottest first, ties to the hotter rank. */
    std::vector<int64_t> by_rank((size_t)buckets, 0);
    std::vector<std::pair<double, int64_t>> frac;
    frac.reserve((size_t)buckets);
    int64_t placed = 0;
    for (int64_t r = 0; r < buckets; ++r) {
        const double x = (double)total * std::pow((double)(r + 1), -kRouteSkew) / sum;
        by_rank[(size_t)r] = (int64_t)std::floor(x);
        placed += by_rank[(size_t)r];
        frac.push_back({ x - std::floor(x), r });
    }
    std::stable_sort(frac.begin(), frac.end(),
                     [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                         return a.first > b.first;
                     });
    for (size_t i = 0; placed < total && i < frac.size(); ++i, ++placed) ++by_rank[(size_t)frac[i].second];
    /* Over the cap: the excess goes to the hottest experts with room. */
    int64_t excess = 0;
    for (int64_t& x : by_rank)
        if (x > cap) { excess += x - cap; x = cap; }
    for (int64_t r = 0; r < buckets && excess > 0; ++r) {
        const int64_t take = std::min(excess, cap - by_rank[(size_t)r]);
        by_rank[(size_t)r] += take;
        excess -= take;
    }
    for (int64_t r = 0; r < buckets; ++r) c[(size_t)order[(size_t)r]] = by_rank[(size_t)r];
    return c;
}

bool timing_input(const KfOperand& d, const Geometry& g, uint64_t seed, std::vector<uint8_t>* out) {
    if (d.role != RAD_OPD_IN || !d.is_index || d.absent || d.shape.empty()) return false;
    const int64_t n = rad_numel(d.shape);
    if (n <= 0) return false;
    /* `plain` is the uniform draw from a range, which is what the branches below mirror: an operand
     * drawn under another law has no variant to make. */
    const bool plain = !d.idx_unique && !d.idx_cu && !d.idx_prefix && !d.idx_history &&
                       !d.idx_ordered && d.idx_const < 0;
    long long ne = 0, M = 0;
    g.get_i("n_expert", &ne);
    g.get_i("M", &M);
    std::vector<float> v((size_t)n, 0.0f);
    if (d.idx_prefix) {
        /* The histogram kf_draw draws uniformly: n - 1 buckets holding idx_max - 1 rows. */
        const int64_t buckets = n > 1 ? n - 1 : 1;
        const int64_t total = d.idx_max > 0 ? d.idx_max - 1 : 0;
        const std::vector<int64_t> cnt = skewed_counts(buckets, total, M, seed);
        int64_t acc = 0;
        for (int64_t i = 1; i < n; ++i) {
            acc += cnt[(size_t)(i - 1)];
            v[(size_t)i] = (float)acc;
        }
    } else if (plain && ne > 0 && d.idx_max == ne && d.shape.size() == 2 && d.shape[1] <= ne) {
        const int64_t rows = d.shape[0], width = d.shape[1];
        const std::vector<double> w0 = route_weights(ne, seed);
        uint64_t s = rad_splitmix(seed);
        std::vector<double> w;
        for (int64_t r = 0; r < rows; ++r) {
            w = w0;
            for (int64_t j = 0; j < width; ++j) {
                double tot = 0;
                for (double x : w) tot += x;
                double u = unit(s) * tot;
                int64_t pick = -1;
                for (int64_t e = 0; e < ne; ++e) {
                    if (w[(size_t)e] <= 0) continue;
                    pick = e;
                    if (u < w[(size_t)e]) break;
                    u -= w[(size_t)e];
                }
                if (pick < 0) return false;
                w[(size_t)pick] = 0;
                v[(size_t)(r * width + j)] = (float)pick;
            }
        }
    } else if (plain && d.shape.size() >= 2 && d.idx_max == n) {
        const std::vector<int64_t> p = seeded_order(n, seed);
        for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)p[(size_t)i];
    } else {
        return false;
    }
    const int64_t bytes = rad_dtype_bytes(d.dtype, n);
    if (bytes <= 0) return false;
    out->assign((size_t)bytes, 0);
    rad_narrow(v.data(), d.dtype, n, out->data());
    return true;
}

bool parse_gpu_metrics(const uint8_t* p, size_t n, GpuMetrics* out) {
    *out = GpuMetrics{};
    if (!p || n < 4) return false;
    uint16_t size = 0;
    std::memcpy(&size, p, 2);
    out->format = p[2];
    out->content = p[3];
    /* gpu_metrics_v1_3 (drivers/gpu/drm/amd/include/kgd_pp_interface.h): 120 bytes, naturally
     * aligned, so its offsets are those of the packed layout cardclock.py unpacks. */
    if (out->format != 1 || out->content != 3 || size < 120 || n < 120) return false;
    auto u16 = [p](size_t off) -> double {
        uint16_t v = 0;
        std::memcpy(&v, p + off, 2);
        return v == 0xFFFF ? -1.0 : (double)v;
    };
    out->hotspot_c      = u16(6);     /* temperature_hotspot */
    out->gfx_busy_pct   = u16(16);    /* average_gfx_activity */
    out->power_w        = u16(22);    /* average_socket_power */
    out->gfxclk_mhz     = u16(40);    /* average_gfxclk_frequency */
    out->gfxclk_now_mhz = u16(54);    /* current_gfxclk */
    uint64_t indep = 0;
    std::memcpy(&indep, p + 112, 8);  /* indep_throttle_status */
    out->throttle = indep == ~(uint64_t)0 ? 0 : indep;
    return true;
}

std::string throttle_names(uint64_t bits) {
    static const std::pair<int, const char*> kBits[] = {
        { 0, "PPT0" }, { 1, "PPT1" }, { 2, "PPT2" }, { 3, "PPT3" }, { 4, "SPL" }, { 5, "FPPT" },
        { 6, "SPPT" }, { 7, "SPPT_APU" }, { 16, "TDC_GFX" }, { 17, "TDC_SOC" }, { 18, "TDC_MEM" },
        { 19, "TDC_VDD" }, { 20, "TDC_CVIP" }, { 21, "EDC_CPU" }, { 22, "EDC_GFX" }, { 23, "APCC" },
        { 32, "TEMP_GPU" }, { 33, "TEMP_CORE" }, { 34, "TEMP_MEM" }, { 35, "TEMP_EDGE" },
        { 36, "TEMP_HOTSPOT" }, { 37, "TEMP_SOC" }, { 38, "TEMP_VR_GFX" }, { 39, "TEMP_VR_SOC" },
        { 40, "TEMP_VR_MEM0" }, { 41, "TEMP_VR_MEM1" }, { 42, "TEMP_LIQUID0" },
        { 43, "TEMP_LIQUID1" }, { 44, "VRHOT0" }, { 45, "VRHOT1" }, { 46, "PROCHOT_CPU" },
        { 47, "PROCHOT_GFX" }, { 56, "PPM" }, { 57, "FIT" },
    };
    std::string s;
    for (const auto& [bit, name] : kBits) {
        if (!(bits >> bit & 1)) continue;
        s += (s.empty() ? "" : "+") + std::string(name);
        bits &= ~((uint64_t)1 << bit);
    }
    for (int bit = 0; bit < 64; ++bit)
        if (bits >> bit & 1) s += (s.empty() ? "" : "+") + fmt("BIT%d", bit);
    return s;
}

std::string bandwidth_refusal(const std::vector<BwProbe>& probes) {
    if (probes.empty()) return "no bandwidth probe ran";
    const BwProbe* big = &probes[0];
    for (const BwProbe& p : probes) {
        if (!(p.warm_gbs > 0) || !(p.cold_gbs > 0) || !std::isfinite(p.warm_gbs) ||
            !std::isfinite(p.cold_gbs))
            return "the " + humanb(p.bytes) + " read probe measured no rate";
        if (p.bytes > big->bytes) big = &p;
    }
    for (const BwProbe& p : probes) {
        if (&p == big || p.cold_gbs <= 1.5 * big->cold_gbs) continue;
        return fmt("cold reads of %s ran at %.0f GB/s, %.1fx the %.0f GB/s of the %s probe: the "
                   "rotation was served from a cache, so a cold timing would be a cache hit",
                   humanb(p.bytes).c_str(), p.cold_gbs, p.cold_gbs / big->cold_gbs,
                   big->cold_gbs, humanb(big->bytes).c_str());
    }
    const double r = big->warm_gbs / big->cold_gbs;
    if (r > 1.5 || r < 1.0 / 1.5)
        return fmt("the %s probe, past every cache either way, ran at %.0f GB/s warm and %.0f GB/s "
                   "cold: the clock moved during calibration", humanb(big->bytes).c_str(),
                   big->warm_gbs, big->cold_gbs);
    return std::string();
}

/* ================================================================== hashing */
namespace {

constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full,
                   P3 = 0x165667B19E3779F9ull, P4 = 0x85EBCA77C2B2AE63ull,
                   P5 = 0x27D4EB2F165667C5ull;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t round1(uint64_t acc, uint64_t in) {
    acc += in * P2;
    acc = rotl(acc, 31);
    return acc * P1;
}
inline uint64_t merge(uint64_t acc, uint64_t v) {
    acc ^= round1(0, v);
    return acc * P1 + P4;
}

}  /* namespace */

uint64_t hash64(const void* data, size_t len, uint64_t seed) {
    const uint8_t* p = (const uint8_t*)data;
    const uint8_t* const end = p + len;
    uint64_t h;
    if (len >= 32) {
        const uint8_t* const limit = end - 32;
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        do {
            v1 = round1(v1, rd64(p));      p += 8;
            v2 = round1(v2, rd64(p));      p += 8;
            v3 = round1(v3, rd64(p));      p += 8;
            v4 = round1(v4, rd64(p));      p += 8;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += (uint64_t)len;
    while (p + 8 <= end) { h ^= round1(0, rd64(p)); h = rotl(h, 27) * P1 + P4; p += 8; }
    if (p + 4 <= end) { h ^= (uint64_t)rd32(p) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
    while (p < end) { h ^= (*p) * P5; h = rotl(h, 11) * P1; ++p; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

uint64_t hash_combine(uint64_t h, uint64_t v) {
    return hash64(&v, sizeof v, h ^ 0x9E3779B97F4A7C15ull);
}

uint64_t hash_str(uint64_t h, const std::string& s) {
    return hash_combine(hash64(s.data(), s.size(), h), s.size());
}

uint64_t fingerprint(const void* p, size_t n) {
    constexpr size_t kBlock = (size_t)4 << 20;
    const int64_t nb = (int64_t)((n + kBlock - 1) / kBlock);
    if (nb <= 1) return hash_combine(hash64(p, n, 0), (uint64_t)n);
    std::vector<uint64_t> part((size_t)nb);
    rad_for_blocks(nb, [&](int64_t b) {
        const size_t lo = (size_t)b * kBlock;
        part[(size_t)b] = hash64((const uint8_t*)p + lo, std::min(kBlock, n - lo), (uint64_t)b);
    });
    uint64_t h = (uint64_t)n;
    for (uint64_t x : part) h = hash_combine(h, x);
    return h;
}

namespace {

/* The NT_GNU_BUILD_ID descriptor of one loaded object, from its PT_NOTE segments in memory. */
std::string build_id_in(const dl_phdr_info* info) {
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr)& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_NOTE) continue;
        const uint8_t* p = (const uint8_t*)(info->dlpi_addr + ph.p_vaddr);
        const uint8_t* end = p + ph.p_memsz;
        while (p + sizeof(ElfW(Nhdr)) <= end) {
            const ElfW(Nhdr)* nh = (const ElfW(Nhdr)*)p;
            const size_t name_sz = (nh->n_namesz + 3u) & ~3u;
            const size_t desc_sz = (nh->n_descsz + 3u) & ~3u;
            const uint8_t* name = p + sizeof(ElfW(Nhdr));
            const uint8_t* desc = name + name_sz;
            if (desc + nh->n_descsz > end) break;
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 &&
                std::memcmp(name, "GNU", 4) == 0) {
                std::string hex;
                for (uint32_t k = 0; k < nh->n_descsz; ++k) hex += fmt("%02x", desc[k]);
                return hex;
            }
            p = desc + desc_sz;
        }
    }
    return std::string();
}

std::string real_path(const std::string& p) {
    char buf[4096];
    if (::realpath(p.c_str(), buf)) return buf;
    return p;
}

struct Loaded { std::string path, build_id; };

std::vector<Loaded> loaded_objects() {
    std::vector<Loaded> v;
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void* data) -> int {
        auto* out = (std::vector<Loaded>*)data;
        /* The main program has an empty name; the vdso has no file. */
        if (!info->dlpi_name || !info->dlpi_name[0]) return 0;
        out->push_back({ real_path(info->dlpi_name), build_id_in(info) });
        return 0;
    }, &v);
    return v;
}

/* The CPU's model name and feature flags: libm resolves FMA or non-FMA implementations of exp, log
 * and pow at load from them, and those differ in the last bit. */
std::string cpu_identity() {
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return "?";
    std::string model, flags;
    char line[8192];
    while (std::fgets(line, sizeof line, f)) {
        if (model.empty() && !std::strncmp(line, "model name", 10)) model = line;
        if (flags.empty() && !std::strncmp(line, "flags", 5)) flags = line;
        if (!model.empty() && !flags.empty()) break;
    }
    std::fclose(f);
    return model + flags;
}

}  /* namespace */

std::string build_id_of(const std::string& path) {
    const std::string want = real_path(path);
    for (const Loaded& l : loaded_objects())
        if (l.path == want) return l.build_id;
    return std::string();
}

uint64_t code_identity(const std::string& lib_path, const std::vector<std::string>& exclude) {
    const std::string me = real_path(lib_path);
    std::vector<std::string> ex;
    for (const std::string& e : exclude) ex.push_back(real_path(e));
    std::vector<Loaded> objs = loaded_objects();
    bool have_lib = false;
    std::vector<std::string> ids;
    for (const Loaded& l : objs) {
        if (l.path == me) {
            if (l.build_id.empty()) return 0;
            have_lib = true;
        } else if (std::find(ex.begin(), ex.end(), l.path) != ex.end()) {
            continue;
        }
        /* An object without a build-id is named by its path instead; it can then only stale the
         * cache by changing under the same name, which a rebuild of the tree does not do. */
        ids.push_back(l.build_id.empty() ? "path:" + l.path : l.build_id);
    }
    if (!have_lib) return 0;
    std::sort(ids.begin(), ids.end());
    uint64_t h = 0x6b62656e63686964ull;   /* "kbenchid" */
    h = hash_str(h, "lib:" + build_id_of(lib_path));
    for (const std::string& s : ids) h = hash_str(h, s);
    h = hash_str(h, cpu_identity());
    return h ? h : 1;
}

/* ================================================================== the reference cache */
namespace {

constexpr char     kCacheMagic[4] = { 'R', 'K', 'C', '1' };
constexpr uint32_t kCacheVersion = 1;

bool mkdirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    size_t i = 0;
    while (i <= dir.size()) {
        const size_t j = dir.find('/', i);
        cur = dir.substr(0, j == std::string::npos ? dir.size() : j);
        if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    struct stat st{};
    return ::stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/* Removes identity directories other than `keep` beyond the most recent `spare` of them: each
 * holds answers of a libref build that no longer exists. */
void prune(const std::string& root, const std::string& keep, int spare) {
    DIR* d = ::opendir(root.c_str());
    if (!d) return;
    std::vector<std::pair<long long, std::string>> dirs;
    while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == ".." || n == keep || n.size() != 16) continue;
        struct stat st{};
        const std::string p = root + "/" + n;
        if (::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            dirs.push_back({ (long long)st.st_mtime, p });
    }
    ::closedir(d);
    std::sort(dirs.rbegin(), dirs.rend());
    for (size_t i = (size_t)std::max(spare, 0); i < dirs.size(); ++i) {
        DIR* sub = ::opendir(dirs[i].second.c_str());
        if (!sub) continue;
        while (dirent* e = ::readdir(sub)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            ::unlink((dirs[i].second + "/" + n).c_str());
        }
        ::closedir(sub);
        ::rmdir(dirs[i].second.c_str());
    }
}

}  /* namespace */

void RefCache::open(const std::string& dir, uint64_t identity) {
    on_ = false;
    id_ = identity;
    if (dir.empty() || identity == 0) return;
    const std::string sub = fmt("%016llx", (unsigned long long)identity);
    if (!mkdirs(dir + "/" + sub)) {
        RAD_WARN("rad-kbench: cannot create the reference cache under %s (%s); running without it",
                 dir.c_str(), std::strerror(errno));
        return;
    }
    prune(dir, sub, 1);
    dir_ = dir + "/" + sub;
    on_ = true;
}

std::string RefCache::path_of(uint64_t key) const {
    return dir_ + fmt("/%016llx.rkc", (unsigned long long)key);
}

bool RefCache::get(uint64_t key, std::vector<Digest>* outs) const {
    if (!on_) return false;
    FILE* f = std::fopen(path_of(key).c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> b;
    uint8_t buf[65536];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + got);
    std::fclose(f);
    if (b.size() < 4 + 4 + 8 + 4 + 8) return false;
    const size_t body = b.size() - 8;
    uint64_t sum;
    std::memcpy(&sum, b.data() + body, 8);
    if (hash64(b.data(), body, key) != sum) return false;
    size_t o = 0;
    auto take = [&](void* dst, size_t n) {
        if (o + n > body) return false;
        std::memcpy(dst, b.data() + o, n);
        o += n;
        return true;
    };
    char magic[4];
    uint32_t ver = 0, nout = 0;
    uint64_t k2 = 0;
    if (!take(magic, 4) || std::memcmp(magic, kCacheMagic, 4) != 0) return false;
    if (!take(&ver, 4) || ver != kCacheVersion) return false;
    if (!take(&k2, 8) || k2 != key) return false;
    if (!take(&nout, 4) || nout > 64) return false;
    std::vector<Digest> v(nout);
    for (Digest& d : v) {
        uint32_t ns = 0, nk = 0;
        if (!take(&d.n, 8) || !take(&d.sumsq, 8) || !take(&d.nonfinite, 8)) return false;
        if (!take(&ns, 4) || ns > (uint32_t)KF_SAMPLES) return false;
        d.sample.resize(ns);
        if (!take(d.sample.data(), (size_t)ns * 4)) return false;
        if (!take(&nk, 4) || (nk != 0 && nk != (uint32_t)KF_SKETCH)) return false;
        d.sketch.resize(nk);
        if (!take(d.sketch.data(), (size_t)nk * 8)) return false;
    }
    if (o != body) return false;
    *outs = std::move(v);
    ++hits_;
    return true;
}

void RefCache::put(uint64_t key, const std::vector<Digest>& outs) {
    if (!on_) return;
    std::vector<uint8_t> b;
    auto put_raw = [&](const void* p, size_t n) {
        b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n);
    };
    put_raw(kCacheMagic, 4);
    put_raw(&kCacheVersion, 4);
    put_raw(&key, 8);
    const uint32_t nout = (uint32_t)outs.size();
    put_raw(&nout, 4);
    for (const Digest& d : outs) {
        put_raw(&d.n, 8);
        put_raw(&d.sumsq, 8);
        put_raw(&d.nonfinite, 8);
        const uint32_t ns = (uint32_t)d.sample.size(), nk = (uint32_t)d.sketch.size();
        put_raw(&ns, 4);
        put_raw(d.sample.data(), (size_t)ns * 4);
        put_raw(&nk, 4);
        put_raw(d.sketch.data(), (size_t)nk * 8);
    }
    const uint64_t sum = hash64(b.data(), b.size(), key);
    put_raw(&sum, 8);

    const std::string final_path = path_of(key);
    const std::string tmp = final_path + fmt(".%d.tmp", (int)::getpid());
    FILE* f = std::fopen(tmp.c_str(), "wb");
    bool ok = f && std::fwrite(b.data(), 1, b.size(), f) == b.size();
    if (f) ok = (std::fclose(f) == 0) && ok;
    if (ok) ok = ::rename(tmp.c_str(), final_path.c_str()) == 0;
    if (!ok) {
        ::unlink(tmp.c_str());
        if (!warned_) {
            RAD_WARN("rad-kbench: cannot write the reference cache at %s (%s)", dir_.c_str(),
                     std::strerror(errno));
            warned_ = true;
        }
        return;
    }
    ++writes_;
}

uint64_t ref_cache_key(uint64_t identity, const std::string& kernel, const KfCase& c,
                       const Geometry& g, uint64_t input_hash) {
    uint64_t h = hash_combine(identity, kCacheVersion);
    h = hash_combine(h, (uint64_t)KF_SAMPLES);
    h = hash_combine(h, (uint64_t)KF_SKETCH);
    h = hash_str(h, kernel);
    h = hash_str(h, c.op);
    h = hash_combine(h, c.seed);
    h = hash_combine(h, (uint64_t)c.m);
    /* The parameters exactly, not Geometry::str(), which prints a float to six digits. */
    for (int i = 0; i < g.n_params(); ++i) {
        const RadParam& p = g.params()[i];
        h = hash_str(h, p.key ? p.key : "");
        h = hash_combine(h, (uint64_t)p.kind);
        h = hash_combine(h, (uint64_t)p.ival);
        h = hash_combine(h, (uint64_t)p.ihi);
        uint64_t dbits;
        std::memcpy(&dbits, &p.dval, 8);
        h = hash_combine(h, dbits);
        h = hash_str(h, p.sval ? p.sval : "");
    }
    for (const KfOperand& d : c.opd) {
        h = hash_combine(h, (uint64_t)d.role);
        h = hash_combine(h, (uint64_t)d.dtype);
        h = hash_combine(h, d.absent ? 1u : 0u);
        for (int64_t x : d.shape) h = hash_combine(h, (uint64_t)x);
        h = hash_combine(h, d.have_ref ? 1u : 0u);
    }
    return hash_combine(h, input_hash);
}

/* ================================================================== row coverage ratchet */
std::set<std::string> read_rows_file(const std::string& path, bool* exists) {
    std::set<std::string> rows;
    FILE* f = std::fopen(path.c_str(), "r");
    if (exists) *exists = f != nullptr;
    if (!f) return rows;
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
        std::string s = line;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
            s.pop_back();
        size_t a = 0;
        while (a < s.size() && s[a] == ' ') ++a;
        s = s.substr(a);
        if (s.empty() || s[0] == '#') continue;
        rows.insert(s);
    }
    std::fclose(f);
    return rows;
}

int write_rows_file(const std::string& path, const std::set<std::string>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return RAD_E_IO;
    std::fprintf(f, "# Kernel rows a replay of this fixture reaches (rad-kbench --coverage).\n"
                    "# A row listed here and not reached fails the replay; regenerate with\n"
                    "# --coverage-write when a row is added or deliberately retired.\n");
    for (const std::string& r : rows) std::fprintf(f, "%s\n", r.c_str());
    const bool ok = std::fclose(f) == 0;
    return ok ? RAD_OK : RAD_E_IO;
}

Ratchet ratchet(const std::set<std::string>& listed, const std::set<std::string>& reached,
                const std::set<std::string>& unknown) {
    Ratchet r;
    for (const std::string& s : listed)
        if (!reached.count(s) && !unknown.count(s)) r.lost.push_back(s);
    for (const std::string& s : reached)
        if (!listed.count(s)) r.gained.push_back(s);
    return r;
}

std::string rows_file_name(const std::string& fixture_path, const std::string& lib) {
    std::string stem = fixture_path.substr(fixture_path.find_last_of('/') + 1);
    const size_t dot = stem.rfind('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    return "kbench_rows_" + stem + "_" + lib + ".txt";
}

/* ================================================================== synthetic geometry */
bool satisfy_constraints(const RadKernelInfo* ki, Geometry* g) {
    if (!ki) return false;
    static const size_t kFact = std::strlen(RAD_DEVICE_FACT_PREFIX);
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < ki->n_constraints; ++i) {
            const RadConstraint& c = ki->constraints[i];
            if (!c.key || !std::strncmp(c.key, RAD_DEVICE_FACT_PREFIX, kFact)) continue;
            if (constraint_holds(c, *g)) continue;
            long long v = 0;
            const bool has_i = g->get_i(c.key, &v);
            switch (c.op) {
                case RAD_C_EQ: g->set_i(c.key, c.ival); break;
                case RAD_C_LE: g->set_i(c.key, c.ival); break;
                case RAD_C_GE: g->set_i(c.key, c.ival); break;
                case RAD_C_DIV:
                    if (c.ival > 0)
                        g->set_i(c.key, has_i && v > 0 ? (v + c.ival - 1) / c.ival * c.ival
                                                       : c.ival);
                    break;
                case RAD_C_IN: {
                    const std::string set = c.sval ? c.sval : "";
                    const auto toks = split_ws(set);
                    if (toks.empty()) break;
                    const std::string first(toks.front());
                    char* end = nullptr;
                    const long long iv = std::strtoll(first.c_str(), &end, 10);
                    if (has_i && end && *end == '\0') g->set_i(c.key, iv);
                    else g->set_s(c.key, first);
                    break;
                }
                default: break;
            }
        }
    }
    for (int i = 0; i < ki->n_constraints; ++i) {
        const RadConstraint& c = ki->constraints[i];
        if (!c.key || !std::strncmp(c.key, RAD_DEVICE_FACT_PREFIX, kFact)) return false;
        if (!constraint_holds(c, *g)) return false;
    }
    return true;
}

/* ================================================================== skeleton containers */
int write_skeleton(const std::string& src, const std::string& dst, std::string* err) {
    auto fail = [&](int rc, const std::string& why) {
        if (err) *err = why;
        return rc;
    };
    const int in = ::open(src.c_str(), O_RDONLY);
    if (in < 0) return fail(RAD_E_IO, src + ": " + std::strerror(errno));
    RadFileHeader h{};
    if (::pread(in, &h, sizeof h, 0) != (ssize_t)sizeof h || h.magic != RAD_MAGIC) {
        ::close(in);
        return fail(RAD_E_FORMAT, src + ": not a .rad container");
    }
    struct stat st{};
    if (::fstat(in, &st) != 0) { ::close(in); return fail(RAD_E_IO, std::strerror(errno)); }
    const int64_t size = (int64_t)st.st_size;

    /* The header and every table. The blob and anything between tables stay a hole. */
    struct Span { uint64_t off, bytes; };
    const Span spans[] = {
        { 0, sizeof h },
        { h.str_off,   h.str_bytes },
        { h.meta_off,  h.meta_count * sizeof(RadFileKV) },
        { h.dir_off,   h.dir_count * sizeof(RadFileEntry) },
        { h.plane_off, h.plane_count * sizeof(RadFilePlane) },
        { h.enc_off,   h.enc_count * sizeof(RadEncoding) },
        { h.vocab_off, h.vocab_bytes },
        { h.prof_off,  h.prof_count * sizeof(RadFileProfile) },
    };
    for (const Span& s : spans)
        if (s.bytes && (s.off > (uint64_t)size || s.bytes > (uint64_t)size - s.off)) {
            ::close(in);
            return fail(RAD_E_FORMAT, src + ": a table lies outside the file");
        }

    const int out = ::open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) { ::close(in); return fail(RAD_E_IO, dst + ": " + std::strerror(errno)); }
    int rc = RAD_OK;
    std::string why;
    if (::ftruncate(out, size) != 0) { rc = RAD_E_IO; why = std::strerror(errno); }
    std::vector<uint8_t> buf(1 << 20);
    for (const Span& s : spans) {
        for (uint64_t done = 0; rc == RAD_OK && done < s.bytes;) {
            const size_t n = (size_t)std::min<uint64_t>(buf.size(), s.bytes - done);
            const ssize_t r = ::pread(in, buf.data(), n, (off_t)(s.off + done));
            if (r <= 0) { rc = RAD_E_IO; why = "short read"; break; }
            if (::pwrite(out, buf.data(), (size_t)r, (off_t)(s.off + done)) != r) {
                rc = RAD_E_IO;
                why = std::strerror(errno);
                break;
            }
            done += (uint64_t)r;
        }
    }
    ::close(in);
    if (::close(out) != 0 && rc == RAD_OK) { rc = RAD_E_IO; why = std::strerror(errno); }
    if (rc != RAD_OK) {
        ::unlink(dst.c_str());
        return fail(rc, dst + ": " + why);
    }
    return RAD_OK;
}

}  /* namespace kb */
}  /* namespace rad */
