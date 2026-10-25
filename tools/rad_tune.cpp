/* rad-tune -- benchmark every point of each kernel's declared tuning space at the shapes the graph
 * actually issues, and write the cache the selector consults at declare.
 *
 * Constraint matching finds a correct kernel for a geometry; within it, the best tile configuration
 * depends on shape and machine (spec §15). This tool is the single tuning mechanism and tunecache.h
 * the single format, so plugins need no tuning tools of their own.
 *
 * The work list is Program::tune_requests: every (kernel, geometry) pair the declared graph can
 * reach, per band, rather than a swept grid.
 *
 * A candidate's time is the median over reps, not the best (a best-of hides clock-dependent
 * candidates), and the spread is reported so a measurement across a clock ramp is visible.
 */
#include "kfixture.h"
#include "opshapes.h"
#include "format/tunecache.h"

#include "rad_device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::printf(
        "rad-tune -- benchmark each kernel's tunable space at this model's shapes, on this machine\n"
        "\n"
        "usage: rad-tune -m <model.rad> [options]\n"
        "\n"
        "  -m, --model FILE   the .rad whose declared graph supplies the shapes\n"
        "      --home DIR     $RADIANCE_HOME; the cache is written to <home>/tune/<machine>.tune\n"
        "      --kernels A:B  the kernel libraries to load, first wins; default every library found\n"
        "      --op NAME      tune only ops whose name starts with NAME\n"
        "      --machine KEY  override the machine key (for writing a cache for another box)\n"
        "      --reps N       timing reps, median reported (default 5)\n"
        "      --iters N      launches per rep (default 32). One launch is short enough that\n"
        "                     launch overhead and clock ramp would otherwise dominate.\n"
        "      --max-tok N    the max_tok declare is run at (default 8192)\n"
        "      --tp N         tensor-parallel width to declare at (default 1). The shapes a\n"
        "                     sharded deployment runs are not the shapes tp=1 runs.\n"
        "      --num-speculative-tokens N|auto\n"
        "                     draft depth to declare at (default auto: the container's own). It puts\n"
        "                     the drafter's kernels in the measured graph.\n"
        "      --dry-run      list what would be measured and stop\n"
        "  -v, --verbose      print every candidate, not just the winner\n"
        "  -h, --help\n");
}

/* A device allocation that frees itself, because the timing path has several ways out. */
struct DevBuf {
    void* p = nullptr;
    ~DevBuf() { if (p) rad_dev_free(p, RAD_MEM_DEVICE); }
};

/* Maximum relative difference from the defaults. A different split-K count reorders the reduction
 * and moves the last bits (at K=5120 in fp32); 1e-3 is two orders above that and far below a
 * transposed tile or a dropped k slice. */
constexpr double kVariantTol = 1e-3;

/* The candidates: every legal point of the kernel's declared space at this geometry, as the cross
 * product of its axes (a hand-written list of points can silently have holes). `tune_valid` prunes
 * points this geometry cannot run, e.g. a split that does not divide K; a pruned point is a skip,
 * not a failure.
 *
 * The all-defaults point, the baseline an untuned install runs, is moved to index 0 and
 * `*has_defaults` set. It is absent only when tune_valid pruned it.
 */
enum { kMaxCandidates = 4096 };

static std::vector<std::vector<std::pair<std::string, int64_t>>>
tune_candidates(const RadKernelInfo* k, const Geometry& g, int64_t* space_size,
                bool* has_defaults) {
    std::vector<std::vector<std::pair<std::string, int64_t>>> out;
    *has_defaults = false;
    if (!k || k->n_tunables <= 0) return out;

    int64_t total = 1;
    for (int i = 0; i < k->n_tunables; ++i) total *= (int64_t)k->tunables[i].n_values;
    if (space_size) *space_size = total;
    if (total <= 0 || total > (int64_t)kMaxCandidates) return out;

    std::vector<RadParam> cp((size_t)k->n_tunables);
    std::vector<int> idx((size_t)k->n_tunables, 0);
    std::vector<std::pair<std::string, int64_t>> point((size_t)k->n_tunables);

    for (int64_t n = 0; n < total; ++n) {
        int64_t rem = n;
        for (int i = k->n_tunables - 1; i >= 0; --i) {
            idx[(size_t)i] = (int)(rem % k->tunables[i].n_values);
            rem /= k->tunables[i].n_values;
        }
        for (int i = 0; i < k->n_tunables; ++i) {
            point[(size_t)i].first  = k->tunables[i].key;
            point[(size_t)i].second = k->tunables[i].values[idx[(size_t)i]];
            cp[(size_t)i].key  = k->tunables[i].key;
            cp[(size_t)i].kind = RAD_P_INT;
            cp[(size_t)i].ival = point[(size_t)i].second;
        }
        if (k->tune_valid &&
            k->tune_valid(g.params(), g.n_params(), cp.data(), (int)cp.size()) != 1) continue;
        out.push_back(point);
    }

    /* The loader requires each default to be one of its axis's values, so the all-defaults point
     * is in the product unless tune_valid pruned it. */
    for (size_t i = 0; i < out.size(); ++i) {
        bool all_default = true;
        for (int a = 0; a < k->n_tunables; ++a)
            if (out[i][(size_t)a].second != k->tunables[a].deflt) { all_default = false; break; }
        if (all_default) { std::swap(out[0], out[i]); *has_defaults = true; break; }
    }
    return out;
}

/* Every OUT and INOUT operand's bytes after one launch at `g`, which carries the candidate point as
 * ordinary parameters. Used to compare a candidate with the defaults. */
int capture_outputs(const KernelRow& row, std::vector<RadTensor>& t,
                    const Geometry& g, const RadOpSchema* schema,
                    const std::vector<OpdShape>& sh, RadStream stream,
                    std::vector<uint8_t>* out) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = g.params(); a.n_p = g.n_params();
    a.rank = 0; a.world_size = 1;
    DevBuf scratch;
    if (row.info->scratch) {
        const int64_t need = row.info->scratch(&a);
        if (need > 0) {
            scratch.p = rad_dev_alloc(need, RAD_MEM_DEVICE);
            if (!scratch.p) return RAD_E_NOMEM;
            rad_memset_async(scratch.p, 0, need, stream);
            a.scratch = scratch.p;
            a.scratch_bytes = need;
        }
    }
    if (row.info->launch(&a, stream) < 0) return RAD_E_SHAPE;
    if (stream && rad_stream_sync(stream) < 0) return RAD_E_STATE;

    /* Size the destination before copying: growing it between async copies would reallocate it
     * under copies still in flight. */
    size_t total = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        if (!t[k].data) continue;
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
        if (bytes > 0) total += (size_t)bytes;
    }
    out->assign(total, 0);
    size_t at = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        if (!t[k].data) continue;
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
        if (bytes <= 0) continue;
        if (rad_memcpy_async(out->data() + at, t[k].data, bytes, stream) < 0) return RAD_E_STATE;
        at += (size_t)bytes;
    }
    if (stream && rad_stream_sync(stream) < 0) return RAD_E_STATE;
    return RAD_OK;
}

/* The largest relative difference over the captured outputs, widened to f32 through each operand's
 * dtype. */
double max_rel_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                    const std::vector<OpdShape>& sh, const RadOpSchema* schema) {
    if (a.size() != b.size() || a.empty()) return a.size() == b.size() ? 0.0 : 1.0;
    double worst = 0.0;
    size_t at = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        const int64_t n = rad_numel(sh[k].shape);
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, n);
        if (bytes <= 0 || at + (size_t)bytes > a.size()) break;
        std::vector<float> fa, fb;
        rad_widen(a.data() + at, sh[k].dtype, n, &fa);
        rad_widen(b.data() + at, sh[k].dtype, n, &fb);
        if ((int64_t)fa.size() < n || (int64_t)fb.size() < n) break;
        double num = 0, den = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = (double)fa[(size_t)i] - (double)fb[(size_t)i];
            num += d * d;
            den += (double)fa[(size_t)i] * (double)fa[(size_t)i];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
        /* Return NaN directly: `rel > worst` is false for it, so a max would drop it. The caller's
         * `!(err <= tol)` rejects it. */
        if (std::isnan(rel)) return rel;
        if (rel > worst) worst = rel;
        at += (size_t)bytes;
    }
    return worst;
}

/* Median microseconds per launch at `g` (which carries the candidate point). Negative means the
 * kernel refused the shape at its entry point (spec 2.1), which is allowed: tune_valid prunes only
 * what the geometry alone can rule out. */
double time_point(const KernelRow& row, std::vector<RadTensor>& t,
                  const Geometry& g, int reps, int iters,
                  RadStream stream, double* spread_pct) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = g.params(); a.n_p = g.n_params();
    a.rank = 0; a.world_size = 1;
    /* Scratch must be device memory: e.g. a split-K GEMM writes its partial planes there. */
    DevBuf scratch;
    if (row.info->scratch) {
        const int64_t need = row.info->scratch(&a);
        if (need > 0) {
            scratch.p = rad_dev_alloc(need, RAD_MEM_DEVICE);
            if (!scratch.p) return (double)RAD_E_NOMEM;
            rad_memset_async(scratch.p, 0, need, stream);
            a.scratch = scratch.p;
            a.scratch_bytes = need;
        }
    }

    /* A warm-up launch; its status decides whether this candidate serves the shape at all. */
    const int s = row.info->launch(&a, stream);
    if (s < 0) return (double)s;
    if (stream && rad_stream_sync(stream) < 0) return (double)RAD_E_STATE;

    /* Stop the clock only after the device finishes: launches are asynchronous, so without the
     * sync this would time host submission only. */
    std::vector<double> us;
    us.reserve((size_t)reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            if (row.info->launch(&a, stream) < 0) return (double)RAD_E_SHAPE;
        if (stream && rad_stream_sync(stream) < 0) return (double)RAD_E_STATE;
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
    }
    std::sort(us.begin(), us.end());
    if (spread_pct)
        *spread_pct = us[us.size() / 2] > 0
                    ? 100.0 * (us.back() - us.front()) / us[us.size() / 2] : 0.0;
    return us[us.size() / 2];
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string model, home = rad_home(), op_filter, machine;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    int      reps = 5, iters = 32;
    int64_t  max_tok = 8192;
    int      tp = 1, spec = -1;          /* -1 = the container's own drafter depth */
    bool     dry = false, verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* w) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "rad-tune: %s needs a value\n", w);
                                 std::exit(2); }
            return argv[++i];
        };
        auto num = [&](const char* w, int64_t lo, int64_t hi) {
            return rad_flag_int("rad-tune", w, next(w), lo, hi);
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-m" || a == "--model") model = next("--model");
        else if (a == "--home")    home = next("--home");
        else if (a == "--kernels") hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--op")      op_filter = next("--op");
        else if (a == "--machine") machine = next("--machine");
        else if (a == "--reps")    reps = (int)num("--reps", 1, INT32_MAX);
        else if (a == "--iters")   iters = (int)num("--iters", 1, INT32_MAX);
        else if (a == "--max-tok") max_tok = num("--max-tok", 1, INT64_MAX);
        else if (a == "--tp")      tp   = (int)num("--tp", 1, INT32_MAX);
        else if (a == "--num-speculative-tokens") {
            /* `auto` is the container's own depth, as the engine's flag of this name reads it. */
            const char* v = next("--num-speculative-tokens");
            spec = std::string(v) == "auto"
                       ? -1 : (int)rad_flag_int("rad-tune", "--num-speculative-tokens", v, 0,
                                                INT32_MAX);
        }
        else if (a == "--dry-run") dry = true;
        else if (a == "-v" || a == "--verbose") verbose = true;
        else { std::fprintf(stderr, "rad-tune: unrecognised argument '%s'. Try --help.\n",
                            a.c_str()); return 2; }
    }
    if (model.empty()) { usage(); return 2; }

    RadFile f;
    if (f.open(model.c_str()) < 0) return 1;

    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, f.str(f.header().arch_id),
                               f.str(f.header().quant), &plugins) < 0) return 1;

    RadModelMeta meta{};
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

    std::vector<std::string> kk, kv;
    std::vector<const char*> kp, vp;
    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const RadFileKV& e = f.meta_at(i);
        if (e.type != RAD_P_STR) continue;
        kk.push_back(f.str(e.key));
        kv.push_back(f.str(e.v.s));
    }
    for (auto& s : kk) kp.push_back(s.c_str());
    for (auto& s : kv) vp.push_back(s.c_str());
    meta.n_kv = (int)kk.size(); meta.kv_key = kp.data(); meta.kv_val = vp.data();

    /* Declare the graph the deployment runs: the tensor-parallel width sets every sharded N, and
     * the speculative depth decides whether the drafter (and its shapes) is in the graph. */
    RadBuildCtx ctx{};
    ctx.rank = 0; ctx.world_size = tp; ctx.max_tok = max_tok;
    RadArchProbe probe{};
    (void)rad_tools_probe(plugins, meta, &probe);
    ctx.max_spec = spec >= 0 ? spec : probe.draft_depth;
    ctx.max_seqs = 256; ctx.max_ctx = meta.n_ctx_train; ctx.scope = "";

    /* Every rank declares concurrently on its own thread and device, as Engine does: kernel init()
     * runs from declare, and a collective's init is a rendezvous (libr4d's all-reduce waits for
     * its peers), so declaring rank 0 alone would hang. Only rank 0's program is measured. */
    Program prog;
    {
        std::vector<Program> progs((size_t)tp);
        std::vector<int>     st((size_t)tp, 0);
        std::vector<std::thread> th;
        for (int ri = 0; ri < tp; ++ri) {
            th.emplace_back([&, ri] {
                if (tp > 1 && rad_dev_set(ri) < 0) {
                    std::fprintf(stderr, "rad-tune: --tp %d needs %d devices; there is no device "
                                         "%d to declare rank %d on\n", tp, tp, ri, ri);
                    st[(size_t)ri] = RAD_E_DEVICE;
                    return;
                }
                RadBuildCtx bc = ctx;
                bc.rank = ri;
                st[(size_t)ri] = rad_tools_declare(plugins, meta, bc, &progs[(size_t)ri], {},
                                                   nullptr, rad_container_sources(f));
            });
        }
        for (auto& t : th) t.join();
        if (tp > 1) rad_dev_set(0);
        for (int ri = 0; ri < tp; ++ri) if (st[(size_t)ri] < 0) return 1;
        prog = std::move(progs[0]);
    }

    if (machine.empty()) machine = rad_machine_key();

    /* Load from the first home on the search path with a cache for this machine; save to the
     * first home on the path. */
    const std::string rel = TuneCache::rel_path(machine);
    const std::string src = rad_home_file(home, rel);
    const std::string save_home = rad_home_dirs(home).front();
    TuneCache cache;
    cache.load(src.substr(0, src.size() - rel.size() - 1), machine);   /* missing is normal */
    const size_t before = cache.rows().size();

    std::printf("rad-tune %s\n", model.c_str());
    std::printf("  machine      %s\n", machine.c_str());
    std::printf("  cache        %s (%zu existing row(s))\n", src.c_str(), before);
    std::printf("  requests     %zu (kernel, shape) pair(s) the declared graph reaches\n\n",
                prog.tune_requests.size());

    Registry& reg = rad_tools_registry();
    auto row_for = [&](const std::string& plugin, const std::string& kernel) -> const KernelRow* {
        for (const KernelRow& r : reg.rows())
            if (r.plugin == plugin && r.info && kernel == r.info->name) return &r;
        return nullptr;
    };

    /* tune_requests lacks the operand list, so schemas are looked up by the kernel's op name. */
    int measured = 0, skipped = 0, single = 0;
    std::map<std::string, int> skip_why;

    /* One explicit stream for the run; timing on the null stream would measure host submission.
     * Without ROCm the device layer is a synchronous host stub, so the same path works. */
    RadStream stream = nullptr;
    if (rad_stream_create(&stream, 0) < 0) {
        std::fprintf(stderr, "rad-tune: could not create a stream: %s\n", rad_dev_last_error());
        return 1;
    }

    for (const Program::TuneRequest& tr : prog.tune_requests) {
        const KernelRow* row = row_for(tr.plugin, tr.kernel);
        if (!row || !row->info || !row->info->launch) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (no such kernel row)"]++;
            ++skipped;
            continue;
        }
        const std::string op = row->info->op;
        if (!op_filter.empty() && op.rfind(op_filter, 0) != 0) continue;

        /* A kernel with at most one candidate has nothing to choose; no row is written. */
        int64_t space = 0;
        bool has_defaults = false;
        const auto cands = tune_candidates(row->info, tr.geom, &space, &has_defaults);
        const int n_var = (int)cands.size();
        if (n_var <= 1) { ++single; continue; }
        if (!has_defaults) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (tune_valid rejects the defaults: no "
                     "reference to check candidates against)"]++;
            ++skipped;
            continue;
        }

        const RadOpSchema* schema = reg.schema(op);
        if (!schema) { skip_why[op + "  (no schema)"]++; ++skipped; continue; }

        int64_t M = 1;
        long long mv = 0;
        for (const char* k : { "M", "q_len" }) if (tr.geom.get_i(k, &mv)) { M = (int64_t)mv; break; }
        if (M <= 0) M = 1;

        /* Operand shapes as rad-kbench takes them: the reference library's description (which
         * defines the op), then the kernel's own, then the fallback table in opshapes.h. */
        std::vector<OpdShape> sh;
        int shst = RAD_E_UNSUPPORTED;
        for (const Plugin& p : reg.plugins()) {
            if (p.kind != RAD_PLUGIN_KERNEL || !p.reference) continue;
            if (const KernelRow* rr = reg.select_in(p.name, op, tr.geom, RAD_DOMAIN_HOST))
                shst = rad_operand_shapes_hook(rr->info, tr.geom, schema->n_operands, &sh);
            break;
        }
        if (shst < 0) shst = rad_operand_shapes_hook(row->info, tr.geom, schema->n_operands, &sh);
        if (shst < 0) shst = rad_operand_shapes(op, tr.geom, schema->n_operands, M, &sh);
        if (shst < 0) {
            skip_why[op + "  (neither the reference nor the kernel describes its operands, and "
                          "tools/opshapes.h has no recipe)"]++;
            ++skipped;
            continue;
        }

        if (dry) {
            std::printf("  %-22s %-46s %s/%s  %d of %lld point(s)\n", op.c_str(),
                        tr.geom.str().c_str(), tr.plugin.c_str(), tr.kernel.c_str(), n_var,
                        (long long)space);
            continue;
        }

        /* Inputs come from kf_draw, the same draw rad-kbench checks with, so each operand stays in
         * the domain its recipe states (constant `seqused`, cumulative lengths, distinct scatter
         * destinations, in-range block tables) and candidates are timed on realistic work. */
        std::vector<KfOperand> kop(sh.size());
        for (size_t k = 0; k < sh.size(); ++k) {
            KfOperand& d = kop[k];
            d.role       = k < (size_t)schema->n_operands ? schema->operands[k].role : RAD_OPD_IN;
            d.dtype      = sh[k].dtype;
            d.shape      = sh[k].shape;
            d.absent     = sh[k].absent;
            d.fill       = sh[k].fill;
            d.is_index   = sh[k].is_index;
            d.idx_unique = sh[k].idx_unique;
            d.idx_cu     = sh[k].idx_cu;
            d.idx_prefix = sh[k].idx_prefix;
            d.idx_history = sh[k].idx_history;
            d.idx_ordered = sh[k].idx_ordered;
            d.idx_max    = sh[k].idx_max;
            d.idx_const  = sh[k].idx_const;
            d.fill_chunk = sh[k].fill_chunk;
        }
        const int64_t vocab_rows = kf_vocab_rows(kop);

        /* One set of device buffers shared by all candidates, so they all see the same bytes.
         * `store` is the host-side draw, only ever a copy source; RadTensor::data must be the
         * device allocation (a host address would fail silently). */
        std::vector<std::vector<uint8_t>> store(sh.size());
        std::vector<DevBuf>               dev(sh.size());
        std::vector<RadTensor>            t(sh.size());
        bool ok = true;
        for (size_t k = 0; k < sh.size() && ok; ++k) {
            const int64_t n = rad_numel(sh[k].shape);
            const int64_t bytes = rad_dtype_bytes(sh[k].dtype, n);
            if (bytes <= 0) { ok = false; break; }
            kf_draw(kop, k, M, vocab_rows, rad_case_seed(1, op, 0, row->info->domain, (int)k),
                    &store[k]);
            dev[k].p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
            if (!dev[k].p) { ok = false; break; }
            if (rad_memcpy_async(dev[k].p, store[k].data(), bytes, stream) < 0) { ok = false; break; }
            t[k].data = dev[k].p;
            t[k].dtype = sh[k].dtype;
            t[k].rank = (uint32_t)sh[k].shape.size();
            for (size_t d = 0; d < sh[k].shape.size() && d < RAD_MAX_RANK; ++d)
                t[k].shape[d] = sh[k].shape[d];
            rad_tensor_pack(&t[k]);
        }
        if (!ok) { skip_why[op + "  (an operand has a plugin-private dtype the core cannot "
                                 "size)"]++; ++skipped; continue; }

        /* One geometry per candidate, built up front. As at issue, the chosen values are appended
         * to the resolved geometry, so a launch reads `sk` via rad_args_geti() like `M`. */
        std::vector<Geometry> gv((size_t)n_var);
        for (int v = 0; v < n_var; ++v) {
            gv[(size_t)v] = tr.geom;
            for (const auto& kv : cands[(size_t)v]) gv[(size_t)v].set_i(kv.first.c_str(), kv.second);
        }

        /* Median over `reps`, with the spread. */
        std::vector<double> best_us((size_t)n_var, -1.0), spread((size_t)n_var, 0.0);
        for (int v = 0; v < n_var; ++v) {
            double sp = 0;
            best_us[(size_t)v] = time_point(*row, t, gv[(size_t)v], reps, iters, stream, &sp);
            spread[(size_t)v] = sp;
        }

        /* ---- each candidate must agree with the defaults before it can win on speed.
         *
         * No other check exercises non-default tile configurations, and a subtly wrong winner
         * would be served from the cache, showing up only as degraded output quality. Candidate 0,
         * the all-defaults point (what an untuned install runs, spec §15), is the reference.
         * kVariantTol allows the last-bit changes of a reordered reduction. */
        std::vector<uint8_t> ref;
        std::vector<double>  verr((size_t)n_var, 0.0);
        if (best_us[0] <= 0 || capture_outputs(*row, t, gv[0], schema, sh, stream, &ref) < 0) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (the defaults refused the shape: no "
                     "reference to check candidates against)"]++;
            ++skipped;
            continue;
        }
        for (int v = 1; v < n_var; ++v) {
            if (best_us[(size_t)v] <= 0) continue;
            std::vector<uint8_t> got;
            if (capture_outputs(*row, t, gv[(size_t)v], schema, sh, stream, &got) < 0) {
                best_us[(size_t)v] = (double)RAD_E_STATE;
                continue;
            }
            verr[(size_t)v] = max_rel_diff(ref, got, sh, schema);
            if (!(verr[(size_t)v] <= kVariantTol)) best_us[(size_t)v] = (double)RAD_E_SHAPE;
        }

        int win = -1;
        for (int v = 0; v < n_var; ++v)
            if (best_us[(size_t)v] > 0 && (win < 0 || best_us[(size_t)v] < best_us[(size_t)win]))
                win = v;

        if (win < 0) {
            std::printf("  %-22s %-46s no candidate served it (%s/%s, %d tried of %lld)\n",
                        op.c_str(), tr.geom.str().c_str(), tr.plugin.c_str(), tr.kernel.c_str(),
                        n_var, (long long)space);
            ++skipped;
            continue;
        }

        const std::string wname = TuneCache::canon_choice(cands[(size_t)win]);

        if (verbose)
            for (int v = 0; v < n_var; ++v) {
                const std::string nm = TuneCache::canon_choice(cands[(size_t)v]);
                if (best_us[(size_t)v] > 0)
                    std::printf("      %-32s %9.3f us  spread %4.0f%%  vs default %8.2g%s\n",
                                nm.c_str(), best_us[(size_t)v], spread[(size_t)v], verr[(size_t)v],
                                v == win ? "  <-" : "");
                else if (!(verr[(size_t)v] <= kVariantTol))
                    std::printf("      %-32s rejected: disagrees with the defaults by %.3g\n",
                                nm.c_str(), verr[(size_t)v]);
                else
                    std::printf("      %-32s refused: %s\n", nm.c_str(),
                                rad_strerror((int)best_us[(size_t)v]));
            }

        TuneRow tr_out;
        tr_out.kernel   = tr.kernel;
        tr_out.plugin   = tr.plugin;
        tr_out.choice   = wname;
        /* Key on the geometry without the point (the chosen axes are in `choice`): the selector
         * looks the instantiation up before it knows what to choose. */
        tr_out.geom     = tr.geom_key;
        tr_out.us       = best_us[(size_t)win];
        tr_out.measured = TuneCache::now_iso8601();
        cache.put(tr_out);
        ++measured;

        std::printf("  %-22s %-46s %-28s %9.3f us  (%d of %lld)\n", op.c_str(),
                    tr.geom.str().c_str(), wname.c_str(), tr_out.us, n_var, (long long)space);
    }

    if (dry) {
        std::printf("\n--dry-run: nothing measured, nothing written.\n");
        return 0;
    }

    if (!skip_why.empty()) {
        std::printf("\nskipped\n");
        for (const auto& [what, n] : skip_why) std::printf("  %-70s x%d\n", what.c_str(), n);
        std::printf("  %d request(s) skipped in total.\n", skipped);
    }
    if (single)
        std::printf("\n%d request(s) had one point or none: nothing to choose, and the kernel's\n"
                    "own axis defaults are already what it runs.\n", single);

    rad_stream_destroy(stream);

    cache.set_machine(machine);
    if (cache.save(save_home) < 0) return 1;

    std::printf("\n%d shape(s) measured, %zu row(s) in %s\n", measured, cache.rows().size(),
                TuneCache::path_for(save_home, machine).c_str());
    return 0;
}
