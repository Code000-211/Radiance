/* oracle.cpp -- run the host kernel beside the device one, on the same bytes, at every issue.
 *
 * ================================================================= why this exists
 *
 * rad-kbench checks one op at a time against libref with operands it invents: dense, rank-2,
 * freshly filled. The engine's operands are row slices of a max_tok buffer, column slices of a
 * fused projection, rank-3 views and weights read from a file, so an op can be right at every
 * shape rad-kbench knows and wrong at the one the model issues. This checks the issued shape. For
 * a matching op it
 *
 *   1. copies every operand (inputs, outputs, weights) to the host before the launch;
 *   2. lets the device kernel run and synchronises;
 *   3. copies every operand back into a second host buffer;
 *   4. passes every weight the device row stores re-laid through that row's RadUnrelayoutFn, so
 *      the dense reference reads canonical planes (see oracle_unrelayout; without this it would
 *      compare a permuted plane against itself);
 *   5. runs the host kernel for the same (op, band) on the step-1 copy, with the same shapes,
 *      strides and parameters;
 *   6. compares each operand's host result against the device result.
 *
 * The host row is the reference library's (the plugin flagged `reference`, as rad-kbench resolves
 * it): the band's own host row when the reference library won it, otherwise the reference row the
 * registry selects at the band's geometry (oracle_reference). A band's host winner may be another
 * host library under test (libavx), and on the host backend it is the kernel being checked.
 *
 * ================================================================= cost
 *
 * Two device-to-host copies of every operand (a fused gate_up weight is 170 MiB), a host sync per
 * issue and a slow reference kernel. A step takes minutes. Off unless requested.
 *
 * ================================================================= usage
 *
 *   RADIANCE_OP_ORACLE=gemm_nt_q,rmsnorm   op names, comma separated; `*` for every op
 *   RADIANCE_OP_ORACLE_MAX=32              stop after this many checks (default 32)
 *   RADIANCE_OP_ORACLE_FROM=0              first issue index to consider (default 0)
 *   RADIANCE_OP_ORACLE_TO=100000           last issue index to consider
 *   RADIANCE_OP_ORACLE_TOL=0.02            relative tolerance below which an operand is silent
 *   RADIANCE_OP_ORACLE_MAXBYTES            the largest operand a check will copy (default 4 GiB)
 *
 * The issue index is Ctx's internal counter, in the order --debug-graph prints, so a failure names
 * a position in the graph.
 *
 * ================================================================= repeat mode
 *
 * RADIANCE_OP_REPEAT asks whether a kernel agrees with itself: it snapshots the operands, runs the
 * device kernel, restores the operands, runs the same device kernel again and byte-compares. This
 * catches unordered reductions, cross-workgroup races and unzeroed workspaces, which instruments
 * that run an op once per side pass whenever the kernel happens to be right. No tolerance: one
 * mantissa bit becomes a different token a few hundred greedy steps later.
 *
 *   RADIANCE_OP_REPEAT=*                   op names, comma separated; `*` for every op
 *   RADIANCE_OP_REPEAT_MAX=1000000         stop after this many checks
 *   RADIANCE_OP_REPEAT_FROM / _TO          issue index window
 *   RADIANCE_OP_REPEAT_MAXBYTES=8388608    operands above this are not copied, restored or
 *                                          compared; valid only because such operands (fused
 *                                          weights) are read-only.
 *
 * ================================================================= blind spots
 *
 * - Ops issued on a lane the snapshot does not synchronise. Lane 1, the only one the ABI has, is
 *   handled; a third lane would need the same line.
 * - Ops the reference library does not serve in the band (skipped, logged once).
 * - Operands whose dtype the core cannot size (plugin-private) and that are not a sized weight.
 * - Device rows that re-lay a weight without publishing an inverse (skipped by name, to avoid
 *   false mismatches).
 * - Ops wrong the same way on both sides (a shared misreading of the schema); rad-kbench's
 *   independent operands cover that.
 */
#include "ctx.h"
#include "format/encoding.h"
#include "plugin/registry.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the state */
struct Ctx::OracleState {
    std::vector<std::string> want;        /* op names; a single "*" means every op */
    int64_t budget = 32;
    int64_t from = 0, to = INT64_MAX;
    double  tol = 2e-2;
    int64_t checked = 0, skipped_nohost = 0;
    bool    said_nohost = false;
    /* Reference host plans resolved for bands another library's host row won, keyed by
     * (op << 16) | band, -1 for a band the reference library does not serve. Deques: an
     * IssuePlan's parameters point into its Resolved's geometry. */
    std::unordered_map<uint64_t, int> ref_at;
    std::deque<Resolved>              ref_res;
    std::deque<IssuePlan>             ref_plans;
    /* ---- repeat mode. Operands over `maxbytes` (read-only weights, e.g. a 170 MiB fused gate_up)
     * are not snapshotted, restored or compared. Decode activations are far below it. */
    bool    repeat = false;
    int64_t maxbytes = 8 << 20;
    /* ---- oracle-mode cap. A snapshot is two host copies of every operand (three for a re-laid
     * weight), so a huge operand would OOM the process. 4 GiB covers a routed layer's whole
     * expert table, the largest operand a check needs. */
    int64_t maxopd = 4LL << 30;
    int64_t differed = 0, unchecked = 0;
    int     skip[MAX_OPERANDS] = {0};

    std::vector<uint8_t>              scratch;    /* the host kernel's, sized per launch */
    /* Re-laid weights inverted to the form a dense reference reads; see oracle_unrelayout. */
    std::vector<std::vector<uint8_t>> logical;
    int                               unlaid[MAX_OPERANDS] = {0};
    std::vector<std::vector<uint8_t>> pre, post;  /* one snapshot per operand */
    int64_t                           span[MAX_OPERANDS] = {0};   /* elements, stacked */
    int64_t                           ent [MAX_OPERANDS] = {0};   /* bytes, one entry */
    /* Entry count for a weight table, 0 for a plane. A table is materialised stacked because the
     * operand is an array of pointers; see oracle_size. */
    int64_t                           tab [MAX_OPERANDS] = {0};
    int                               alias[MAX_OPERANDS] = {0};   /* which operand owns the copy */
    RadTensor                         ht[MAX_OPERANDS]{};
};

namespace {

/* Elements a tensor's strides can reach, which is what must be copied. Not numel: a column slice
 * of a fused projection is not dense. */
int64_t span_of(const RadTensor* t) {
    if (!t->data || t->rank == 0) return 0;
    int64_t s = 1;
    for (uint32_t d = 0; d < t->rank; ++d) {
        if (t->shape[d] <= 0) return 0;
        const int64_t st = t->stride[d];
        if (st > 0) s += (t->shape[d] - 1) * st;
    }
    return s;
}

/* One element as a double. False for a dtype with no scalar value; the caller then byte-compares. */
bool elem(const uint8_t* p, uint32_t dt, int64_t i, double* out) {
    switch (dt) {
    case RAD_F32: case RAD_BF16: case RAD_F16: case RAD_F8E4M3: case RAD_F8E5M2:
    case RAD_I8:  case RAD_U8:   case RAD_BOOL: case RAD_I16:
        *out = rad_load_f32(p, dt, i); return true;
    case RAD_I32:  { int32_t v; std::memcpy(&v, p + i * 4, 4); *out = v; return true; }
    case RAD_U32:  { uint32_t v; std::memcpy(&v, p + i * 4, 4); *out = v; return true; }
    case RAD_I64:  { int64_t v; std::memcpy(&v, p + i * 8, 8); *out = (double)v; return true; }
    default: return false;
    }
}

double row_at(const uint8_t* p, uint32_t dt, int64_t base, int64_t k, int64_t n) {
    double v = 0;
    if (base + k < n) elem(p, dt, base + k, &v);
    return v;
}


}  /* namespace */

/* ------------------------------------------------------------------ open / close */
void Ctx::oracle_open() {
    /* Repeat mode shares arming, filtering and snapshots with the oracle and differs only in the
     * second run and the comparison: the oracle asks whether a kernel is right, repeat whether it
     * is deterministic. */
    const char* spec = std::getenv("RADIANCE_OP_REPEAT");
    const bool  rep  = spec && *spec;
    if (!rep) spec = std::getenv("RADIANCE_OP_ORACLE");
    if (!spec || !*spec) return;

    oracle_ = new OracleState();
    oracle_->repeat = rep;
    for (const char* s = spec; *s; ) {
        const char* e = s;
        while (*e && *e != ',') ++e;
        if (e > s) oracle_->want.emplace_back(s, (size_t)(e - s));
        s = *e ? e + 1 : e;
    }
    auto geti = [](const char* k, int64_t d) {
        const char* v = std::getenv(k);
        return (v && *v) ? (int64_t)std::strtoll(v, nullptr, 10) : d;
    };
    oracle_->budget = geti(rep ? "RADIANCE_OP_REPEAT_MAX" : "RADIANCE_OP_ORACLE_MAX", rep ? 1000000 : 32);
    oracle_->maxbytes = geti("RADIANCE_OP_REPEAT_MAXBYTES", 8 << 20);
    oracle_->maxopd   = geti("RADIANCE_OP_ORACLE_MAXBYTES", 4LL << 30);
    oracle_->from   = geti(rep ? "RADIANCE_OP_REPEAT_FROM" : "RADIANCE_OP_ORACLE_FROM", 0);
    oracle_->to     = geti(rep ? "RADIANCE_OP_REPEAT_TO" : "RADIANCE_OP_ORACLE_TO", INT64_MAX);
    if (const char* t = std::getenv("RADIANCE_OP_ORACLE_TOL")) oracle_->tol = std::strtod(t, nullptr);

    if (oracle_->repeat) {
        RAD_WARN("op REPEAT armed for {%s}: issues [%lld,%lld], at most %lld checks, operands up "
                 "to %lld bytes. Every matching issue runs its DEVICE kernel TWICE on the same "
                 "bytes and byte-compares. A difference is a kernel that is not a function of its "
                 "inputs. This run is not a measurement of anything.",
                 spec, (long long)oracle_->from, (long long)oracle_->to,
                 (long long)oracle_->budget, (long long)oracle_->maxbytes);
        return;
    }
    RAD_WARN("op oracle armed for {%s}: issues [%lld,%lld], at most %lld checks, tolerance %.3g, "
             "operands up to %lld MiB; every matching issue also runs on the host, so this run "
             "measures nothing",
             spec, (long long)oracle_->from, (long long)oracle_->to,
             (long long)oracle_->budget, oracle_->tol, (long long)(oracle_->maxopd >> 20));
}

void Ctx::oracle_close() {
    if (!oracle_) return;
    if (oracle_->repeat)
        RAD_WARN("op repeat: %lld issue(s) run twice, %lld of them answered differently; %lld "
                 "more were not judged because an operand was over the copy cap "
                 "(RADIANCE_OP_REPEAT_MAXBYTES)",
                 (long long)oracle_->checked, (long long)oracle_->differed,
                 (long long)oracle_->unchecked);
    else
        RAD_WARN("op oracle: %lld op(s) checked, %lld skipped for want of a reference kernel",
                 (long long)oracle_->checked, (long long)oracle_->skipped_nohost);
    delete oracle_;
    oracle_ = nullptr;
}

/* ------------------------------------------------------------------ the check */
bool Ctx::oracle_arm(rad_op h) const {
    OracleState& o = *oracle_;
    if (o.budget <= 0) return false;
    if (n_issues_ < o.from || n_issues_ > o.to) return false;
    const std::string& name = plans_[h].info->op;
    for (const std::string& w : o.want)
        if (w == "*" || w == name) return true;
    return false;
}

/* ------------------------------------------------------------------ sizing the snapshot
 *
 * A grouped GEMM's weight operand is a table (Ctx::WeightTable): after placement a layer's experts
 * live in different places, so the operand is a device array of pointers with a zero leading
 * stride. Copied as a plane it would read far past the pointer array and the host kernel would
 * dereference device pointers. Instead every entry is copied into one contiguous host buffer with
 * a non-zero leading stride, the stacked form libref's expert_entry() also accepts.
 *
 * A plugin-private dtype has no byte count in the core by design (spec 19.6). Such an operand is
 * sized from the weight it is bound to (the layout pass records the stored size in WeightInfo);
 * only an operand that is neither sizeable nor a weight is declined. */
void Ctx::oracle_size(rad_op h, int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;

    /* Stored byte count of the weight at each operand position, from the layout pass. All entries
     * of a table map to one position and are proven identical at first issue. */
    int64_t wb[MAX_OPERANDS] = {0};
    int     wlaid[MAX_OPERANDS] = {0};
    if (program_)
        for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j) {
            const rad_weight wh = oi.weights[j];
            const int k = (int)oi.weight_opd[j];
            if (!wh || (size_t)wh >= program_->weights.size()) continue;
            if (k < 0 || k >= n_opd) continue;
            wb[k]    = program_->weights[(size_t)wh].stored_bytes;
            wlaid[k] = program_->weights[(size_t)wh].layout_tag.empty() ? 0 : 1;
        }

    for (int i = 0; i < n_opd; ++i) {
        const RadTensor& t = tbuf_[i];
        /* A zero leading stride marks a table: resolve_weight_table() prepends the expert axis
         * that way to say it is not addressable by arithmetic. */
        o.tab[i] = (t.data && t.rank >= 2 && t.stride[0] == 0 && t.shape[0] > 0) ? t.shape[0] : 0;
        const int64_t es = span_of(&t);           /* one entry, or the whole plane when not a table */
        o.span[i] = o.tab[i] ? es * o.tab[i] : es;
        o.ent[i]  = es > 0 ? rad_dtype_bytes(t.dtype, es) : 0;
        if (o.ent[i] <= 0 && es > 0) o.ent[i] = wb[i];
        /* A re-laid weight's size is the stored size, not the declared plane's. E.g. the fp8
         * lm_head pads each row to 16 bytes and embeds its block scales (2608 bytes per row, not
         * n_embd = 2560), and the inverse rejects a buffer of the wrong size. */
        if (wlaid[i] && wb[i] > 0) o.ent[i] = wb[i];
        o.ht[i] = t;
        /* The stacked reading: the expert axis becomes addressable and its stride is one entry. */
        if (o.tab[i]) o.ht[i].stride[0] = es;

        /* Aliased operands share one host buffer, as on the device. A residual add {x, h} -> {x}
         * with separate copies would report a false mismatch the size of the addend. The first
         * operand with a given pointer owns the copy. */
        o.alias[i] = i;
        for (int j = 0; j < i; ++j)
            if (t.data && tbuf_[j].data == t.data) { o.alias[i] = o.alias[j]; break; }
    }
}

/* Bytes the snapshot of operand `i` occupies; for a table, all entries. */
int64_t Ctx::oracle_opd_bytes(int i) const {
    const OracleState& o = *oracle_;
    if (o.span[i] <= 0 || o.ent[i] <= 0) return 0;
    return o.tab[i] ? o.ent[i] * o.tab[i] : o.ent[i];
}

void Ctx::oracle_snap(int n_opd, bool post) {
    OracleState& o = *oracle_;
    std::vector<std::vector<uint8_t>>& into = post ? o.post : o.pre;
    into.resize(MAX_OPERANDS);
    std::vector<void*> ptr;
    /* Drain lane 1 before the copies. Ops may run on lane 1 (the MoE shared expert does), and the
     * copies below go on `stream_`, which is not ordered against it. Otherwise `post` could read an
     * output before it is written, and `pre` an input before its producer wrote it. */
    if (lane1_) rad_stream_sync(lane1_);
    for (int i = 0; i < n_opd; ++i) {
        const RadTensor& t = tbuf_[i];
        if (o.alias[i] != i) continue;
        const int64_t bytes = oracle_opd_bytes(i);
        /* The maxbytes cap applies to repeat mode only: the oracle's host kernel needs every
         * operand, while repeat re-launches on the device operands in place and just skips
         * restoring and comparing the capped ones. */
        if (!post) o.skip[i] = o.repeat && bytes > o.maxbytes;
        if (o.skip[i]) { into[(size_t)i].clear(); continue; }
        /* Shrink a grossly oversized slot. Otherwise each slot in `pre`, `post` and `logical`
         * keeps the high-water capacity of every operand it ever held, which on a large model
         * can OOM the host. This bounds the footprint by the largest single issue. */
        into[(size_t)i].assign((size_t)bytes, 0);
        if (into[(size_t)i].capacity() > (size_t)bytes * 2 + (1u << 20))
            into[(size_t)i].shrink_to_fit();
        if (bytes <= 0) continue;
        if (!o.tab[i]) { rad_memcpy_async(into[(size_t)i].data(), t.data, bytes, stream_); continue; }
        /* Fetch the entry pointers first, then copy each entry. An entry may be device- or
         * host-resident; the copy infers the direction from the pointer. */
        ptr.assign((size_t)o.tab[i], nullptr);
        rad_memcpy_async(ptr.data(), t.data, o.tab[i] * (int64_t)sizeof(void*), stream_);
        rad_stream_sync(stream_);
        for (int64_t e = 0; e < o.tab[i]; ++e)
            if (ptr[(size_t)e])
                rad_memcpy_async(into[(size_t)i].data() + e * o.ent[i], ptr[(size_t)e],
                                 o.ent[i], stream_);
    }
    rad_stream_sync(stream_);
    if (!post)
        for (int i = 0; i < n_opd; ++i)
            o.ht[i].data = o.span[i] > 0 ? (void*)o.pre[(size_t)o.alias[i]].data() : nullptr;
}

/* ------------------------------------------------------------------ the stored layout
 *
 * A kernel may rearrange a weight's canonical planes at load (spec §4.3); libr4d's fp8a8 GEMMs
 * store WMMA fragment order, the same N*K bytes permuted so a wave's 16 rows arrive in lane order
 * from one coalesced load. libref reads [N, K] row-major, so passing the stored bytes unchanged
 * compares uncorrelated tensors (rel error ~1.35 on a correct kernel).
 *
 * The stored bytes therefore go through the device row's inverse back to canonical planes, then to
 * the form the host row reads: one plane converted element-wise (e.g. int4 expert codes as
 * integers), or a whole encoding decoded by the core (e.g. an lm_head of codes and scales).
 *
 * No inverse means no check: returns false, logs a named skip, and the issue is not counted.
 *
 * The logical shape comes from the host row's `opd_shape` first, the device row's as fallback:
 * `opd_shape` describes the operand the kernel is handed, which for the device row is the stored
 * form being inverted. libref describes every op in docs/OPS.md. Extents come from this issue's
 * parameters, not the weight declaration, so a tensor-parallel shard uses per-rank N and K.
 *
 * Two forms skip the description. A host row whose layout hook joins the planes
 * (abi/rad_planes.h: a quantised lm head, a zero-point grid's scale and zero) is handed its own
 * stored form of the inverted planes, as the engine would hand it, at the weight's declared shape.
 * A single plane the host reads as stored, where no description exists (a zero-point grid's codes),
 * is handed the canonical plane itself. A weight table, or a plane widened at load, takes the
 * description. */
bool Ctx::oracle_unrelayout(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int64_t seq,
                            int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;
    for (int i = 0; i < n_opd; ++i) o.unlaid[i] = 0;

    const RadKernelInfo* ki = dip.row ? dip.row->info : nullptr;
    if (!ki || !ki->layout || !program_) return true;   /* every operand is read as it is */
    const RadKernelInfo* hk = hip.row ? hip.row->info : nullptr;
    const RadShapeFn shape = hk && hk->opd_shape ? hk->opd_shape : ki->opd_shape;

    o.logical.resize(MAX_OPERANDS);
    for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j) {
        const rad_weight wh = oi.weights[j];
        if (!wh || (size_t)wh >= program_->weights.size()) continue;
        const WeightInfo& w = program_->weights[(size_t)wh];
        if (w.layout_tag.empty()) continue;             /* stored as it is */
        const int i = oi.weight_opd[j];
        if (i < 0 || i >= n_opd || o.span[i] <= 0 || o.unlaid[i]) continue;

        if (!ki->unrelayout || !shape) {
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. '%s' is stored as %s's '%s', and no %s "
                     "can be had for it -- so the reference would read a permuted plane row-major "
                     "and disagree with a kernel that is right.",
                     (long long)seq, oi.op.c_str(), w.name.c_str(), ki->name,
                     w.layout_tag.c_str(), ki->unrelayout ? "RadShapeFn" : "RadUnrelayoutFn");
            return false;
        }
        if (!o.tab[i] && hk && !w.widen) {
            RadTensor pt[RAD_ENC_MAX_PLANES];
            weight_planes(w, pt);
            RadLayout hl{};
            const int hst = hk->layout ? hk->layout(hip.args.p, hip.args.n_p, i, &w.enc, w.sel,
                                                    pt, w.n_sel, &hl)
                                       : RAD_E_UNSUPPORTED;
            RadOpdDesc probe{};
            const bool described =
                shape(dip.args.p, dip.args.n_p, i, &probe) == RAD_OK && probe.rank > 0;
            const bool joins = hst == RAD_OK && hk->relayout && hl.bytes > 0;
            const bool as_stored = hst == RAD_E_UNSUPPORTED && w.n_sel == 1 && !described;
            if (joins || as_stored) {
                std::vector<std::vector<uint8_t>> pl((size_t)w.n_sel);
                for (int k = 0; k < w.n_sel; ++k) {
                    pl[(size_t)k].assign((size_t)(w.sel_rows[k] *
                                                  rad_enc_row_bytes(pt[k].dtype, w.sel_cols[k])), 0);
                    pt[k].data = pl[(size_t)k].data();
                }
                const std::vector<uint8_t>& src = o.pre[(size_t)o.alias[i]];
                int rc = ki->unrelayout(dip.args.p, dip.args.n_p, i, &w.enc, w.sel, src.data(),
                                        (int64_t)src.size(), pt, w.n_sel);
                if (rc >= 0 && joins) {
                    o.logical[(size_t)i].assign((size_t)hl.bytes, 0);
                    rc = hk->relayout(hip.args.p, hip.args.n_p, i, &w.enc, w.sel, pt, w.n_sel,
                                      o.logical[(size_t)i].data(), hl.bytes);
                } else if (rc >= 0) {
                    o.logical[(size_t)i] = pl[0];
                }
                if (rc < 0) {
                    RAD_WARN("oracle #%lld op '%s': NOT CHECKED. '%s' could not be brought from "
                             "%s's stored form to %s's (%s).", (long long)seq, oi.op.c_str(),
                             w.name.c_str(), ki->name, hk->name, rad_strerror(rc));
                    return false;
                }
                RadTensor lg{};
                lg.dtype = joins ? hl.dtype : pt[0].dtype;
                lg.rank = w.decl.rank;
                for (uint32_t k = 0; k < w.decl.rank && k < RAD_MAX_RANK; ++k)
                    lg.shape[k] = w.decl.shape[k];
                rad_tensor_pack(&lg);
                lg.data = o.logical[(size_t)i].data();
                o.ht[i] = lg;
                o.unlaid[i] = 1;
                continue;
            }
        }
        RadOpdDesc d{};
        if (shape(dip.args.p, dip.args.n_p, i, &d) != RAD_OK || d.rank == 0) {
            /* RadShapeFn gives no reason when it declines, so log the geometry. */
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. nothing could describe the logical form "
                     "of '%s' at this geometry {%s}, so there is nothing to invert its layout "
                     "into.", (long long)seq, oi.op.c_str(), w.name.c_str(),
                     oi.base.str().c_str());
            return false;
        }
        RadTensor lg{};
        lg.dtype = d.dtype;
        lg.rank  = d.rank;
        int64_t n = 1, acc = 1;
        for (uint32_t k = 0; k < d.rank && k < RAD_MAX_RANK; ++k) { lg.shape[k] = d.shape[k]; n *= d.shape[k]; }
        for (int k = (int)d.rank - 1; k >= 0; --k) { lg.stride[k] = acc; acc *= lg.shape[k]; }
        const int64_t bytes = rad_dtype_bytes(lg.dtype, n);
        if (bytes <= 0) return false;
        /* A table's logical form must be the stacked plane; a single-expert description would
         * make the inverse write past the buffer. */
        if (o.tab[i] && (d.rank < 2 || lg.shape[0] != o.tab[i])) {
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. '%s' is a table of %lld experts and the "
                     "logical form offered for it leads with %lld, so the two are not the same "
                     "tensor.", (long long)seq, oi.op.c_str(), w.name.c_str(),
                     (long long)o.tab[i], (long long)(d.rank ? lg.shape[0] : 0));
            return false;
        }
        o.logical[(size_t)i].assign((size_t)bytes, 0);
        if (o.logical[(size_t)i].capacity() > (size_t)bytes * 2 + (1u << 20))
            o.logical[(size_t)i].shrink_to_fit();
        lg.data = o.logical[(size_t)i].data();

        /* One inverse call per table entry: the hook describes one weight (one expert), and the
         * table is an engine construct. Each entry lands at its offset in the stacked tensor. */
        const std::vector<uint8_t>& src = o.pre[(size_t)o.alias[i]];
        const int64_t n_ent = o.tab[i] ? o.tab[i] : 1;
        const int64_t ent_bytes = o.tab[i] ? o.ent[i] : (int64_t)src.size();
        const int64_t ent_n = n / n_ent;
        std::vector<std::vector<uint8_t>> pl((size_t)w.n_sel);
        RadTensor pt[RAD_ENC_MAX_PLANES];
        weight_planes(w, pt);
        int rc = RAD_OK;
        std::string why;
        for (int64_t e = 0; e < n_ent && rc >= 0; ++e) {
            for (int k = 0; k < w.n_sel; ++k) {
                pl[(size_t)k].assign((size_t)(w.sel_rows[k] *
                                              rad_enc_row_bytes(pt[k].dtype, w.sel_cols[k])), 0);
                pt[k].data = pl[(size_t)k].data();
            }
            rc = ki->unrelayout(dip.args.p, dip.args.n_p, i, &w.enc, w.sel,
                                src.data() + e * ent_bytes, ent_bytes, pt, w.n_sel);
            if (rc < 0) break;
            uint8_t* dst = (uint8_t*)lg.data + rad_dtype_bytes(lg.dtype, ent_n * e);
            const void* pd[RAD_ENC_MAX_PLANES] = {};
            for (int k = 0; k < w.n_sel; ++k) pd[k] = pl[(size_t)k].data();
            rc = enc_selection_logical(w.enc, w.sel, w.n_sel, w.sel_rows, w.sel_cols, pd,
                                       lg.dtype, ent_n, dst, &why);
        }
        if (rc < 0) {
            /* Inverses mostly refuse on size, so log both sizes. */
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. %s could not invert its own layout for "
                     "'%s' (%s%s%s): %lld stored bytes into %lld %s elements, so the reference has "
                     "nothing it can read.",
                     (long long)seq, oi.op.c_str(), ki->name, w.name.c_str(), rad_strerror(rc),
                     why.empty() ? "" : ": ", why.c_str(), (long long)ent_bytes, (long long)ent_n,
                     rad_dtype_name(lg.dtype));
            return false;
        }
        o.ht[i] = lg;
        o.unlaid[i] = 1;
    }
    return true;
}


void Ctx::oracle_run(rad_op h, const IssuePlan& hip, int64_t seq, int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;

    /* The host kernel writes into the `pre` copy; only the pointers differ from the device call. */
    RadArgs a = hip.args;
    a.t   = o.ht;
    a.n_t = n_opd;
    a.phase = phase_;   /* the reference is launched per issue too, so it sees the step's phase */
    o.scratch.assign((size_t)(a.scratch_bytes > 0 ? a.scratch_bytes : 0), 0);
    a.scratch = o.scratch.empty() ? nullptr : o.scratch.data();

    const int rc = hip.launch(&a, nullptr);
    if (rc < 0) {
        /* RAD_E_UNSUPPORTED: the reference does not implement this form (e.g. one of
         * `gdn_recurrent_update`'s two state layouts); a coverage gap, logged as not checked. Any
         * other code is the reference refusing shapes it claims to serve: a finding about the
         * engine's operands. */
        if (rc == RAD_E_UNSUPPORTED)
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. the host kernel (%s) does not implement "
                     "what these parameters ask for {%s}, so there is no comparison to make.",
                     (long long)seq, oi.op.c_str(),
                     hip.row && hip.row->info ? hip.row->info->name : "?", oi.base.str().c_str());
        else {
            /* Log the refused shapes along with the error. */
            std::string sh;
            for (int i = 0; i < n_opd; ++i)
                if (o.ht[i].data)
                    sh += fmt("%s%d:%s", sh.empty() ? "" : " ", i, tensor_shape_str(o.ht[i]).c_str());
            RAD_WARN("oracle #%lld op '%s': the host kernel (%s) refused the engine's own operands "
                     "with %s: {%s} %s",
                     (long long)seq, oi.op.c_str(),
                     hip.row && hip.row->info ? hip.row->info->name : "?", rad_strerror(rc),
                     oi.base.str().c_str(), sh.c_str());
        }
        return;
    }

    /* Per operand: the largest disagreement relative to the device side's max magnitude. A pure
     * input must come out at zero, which validates the snapshot itself. */
    std::string line;
    bool bad = false;
    for (int i = 0; i < n_opd; ++i) {
        if (o.span[i] <= 0) continue;
        /* An inverted weight is read-only and the host read a separate logical buffer, so there is
         * nothing to compare. It is still printed, with the shape the host saw. */
        if (o.unlaid[i]) {
            line += fmt("  opd%d~ %-6s %-22s  read through %s's own inverse; not compared\n",
                        i, rad_dtype_name(o.ht[i].dtype), tensor_shape_str(o.ht[i]).c_str(),
                        ip_kernel_name(h));
            continue;
        }
        const uint32_t dt = o.ht[i].dtype;
        const uint8_t* hp = o.pre[(size_t)o.alias[i]].data();
        const uint8_t* dp = o.post[(size_t)o.alias[i]].data();
        const int64_t  n  = o.span[i];

        double worst = 0, scale = 0, hv = 0, dv = 0, sum = 0, hsum = 0;
        int64_t at = -1, nbad = 0;
        int64_t badix[4] = { -1, -1, -1, -1 };
        double x = 0, y = 0;
        if (!elem(hp, dt, 0, &x)) {
            const bool same = std::memcmp(hp, dp, o.pre[(size_t)o.alias[i]].size()) == 0;
            line += fmt("  opd%d %-6s %s %s\n", i, rad_dtype_name(dt),
                        tensor_shape_str(o.ht[i]).c_str(), same ? "bytes equal" : "bytes differ");
            bad = bad || !same;
            continue;
        }
        /* fp8 planes are compared in code space. E4M3's code gap is 1/16 to 1/8 of the value, well
         * above the tolerance, so a one-ulp bf16 difference before quantisation can move an element
         * to the adjacent code (e.g. host 176, device 192) on a correct kernel. Codes with the same
         * sign and magnitudes one apart are counted and reported but do not fail the check;
         * anything further apart is measured normally. */
        const bool fp8 = (dt == RAD_F8E4M3 || dt == RAD_F8E5M2);
        int64_t ulp1 = 0;
        for (int64_t k = 0; k < n; ++k) {
            if (!elem(hp, dt, k, &x) || !elem(dp, dt, k, &y)) break;
            scale = std::max(scale, std::fabs(y));
            if (std::isfinite(y)) sum += y;
            if (std::isfinite(x)) hsum += x;
            if (fp8 && hp[k] != dp[k]) {
                const int da = (int)(hp[k] & 0x7Fu) - (int)(dp[k] & 0x7Fu);
                if (((hp[k] ^ dp[k]) & 0x80u) == 0 && (da == 1 || da == -1)) { ++ulp1; continue; }
            }
            const double d = std::fabs(x - y);
            if (std::isnan(x) != std::isnan(y) || d > worst) {
                worst = std::isnan(x) != std::isnan(y) ? INFINITY : d;
                at = k; hv = x; dv = y;
            }
            /* Record the first few bad indices too: their positions tell a wrong operand from a
             * wrong lane. */
            if (d > 1e-30 && (std::isnan(x) != std::isnan(y) || d > o.tol * std::fabs(x))) {
                if (nbad < 4) badix[nbad] = k;
                ++nbad;
            }
        }
        const double rel = worst / (scale > 0 ? scale : 1.0);
        if (rel > o.tol) bad = true;
        double f[4] = {0, 0, 0, 0};
        for (int64_t k = 0; k < 4 && k < n; ++k) elem(dp, dt, k, &f[k]);
        /* Per-row sums for up to three rows (dim 0, the token for activations), to localise errors
         * beyond row 0 that the head values and whole-tensor sum cannot. */
        std::string rows;
        if (o.ht[i].rank >= 2 && o.ht[i].shape[0] > 1 && o.ht[i].stride[0] > 0) {
            const int64_t nr = o.ht[i].shape[0] < 3 ? o.ht[i].shape[0] : 3;
            const int64_t w  = o.ht[i].stride[0];
            for (int64_t r = 0; r < nr; ++r) {
                double rs = 0;
                for (int64_t k = r * w; k < (r + 1) * w && k < n; ++k) {
                    double y = 0;
                    if (!elem(dp, dt, k, &y)) break;
                    if (std::isfinite(y)) rs += y;
                }
                double t[3] = {0, 0, 0};
                const int64_t last = (r + 1) * w < n ? (r + 1) * w : n;
                for (int j = 0; j < 3; ++j)
                    if (last - 3 + j >= r * w) elem(dp, dt, last - 3 + j, &t[j]);
                /* Three values at each end of the row, as llama.cpp's eval-callback prints, since
                 * equal sums do not imply equal vectors. */
                rows += fmt("\n      row%lld sum %.6g head [%.5g %.5g %.5g] tail [%.5g %.5g %.5g]",
                            (long long)r, rs,
                            row_at(dp, dt, r * w, 0, n), row_at(dp, dt, r * w, 1, n),
                            row_at(dp, dt, r * w, 2, n), t[0], t[1], t[2]);
            }
        }
        /* The worst element's flat index and both values, to tell a wrong tensor from one wrong
         * padding lane. */
        std::string at_s;
        if (at >= 0 && rel > o.tol) {
            at_s = fmt("\n      worst at [%lld]: host %.9g  device %.9g  (diff %.4g)"
                       "\n      %lld of %lld element(s) differ; host sum %.6g vs device sum %.6g"
                       "\n      first bad at",
                       (long long)at, hv, dv, worst, (long long)nbad, (long long)n, hsum, sum);
            for (int b = 0; b < 4 && badix[b] >= 0; ++b) at_s += fmt(" %lld", (long long)badix[b]);
        }
        /* The one-code-step count is always printed, so absorbed differences stay visible. */
        const std::string u1 = ulp1 ? fmt("  %lld at one fp8 code step", (long long)ulp1)
                                    : std::string();
        line += fmt("  opd%d %-6s %-22s n=%-9lld amax %.4g sum %.6g [%.5g %.5g %.5g %.5g]%s"
                    "  rel %.4g%s%s%s\n",
                    i, rad_dtype_name(dt), tensor_shape_str(o.ht[i]).c_str(), (long long)n, scale,
                    sum, f[0], f[1], f[2], f[3], rows.c_str(), rel, u1.c_str(),
                    rel > o.tol ? "  <-- MISMATCH" : "", at_s.c_str());
    }

    const char* dk = ip_kernel_name(h);
    RAD_WARN("oracle #%lld op '%s' %s [device %s vs host %s] {%s}\n%s",
             (long long)seq, oi.op.c_str(), bad ? "MISMATCH" : "agrees",
             dk, hip.row && hip.row->info ? hip.row->info->name : "?",
             oi.base.str().c_str(), line.c_str());
}

/* The device kernel's name for the report, looked up the same way abort_step does. */
const char* Ctx::ip_kernel_name(rad_op h) const {
    const OpPlan& p = plans_[h];
    const int dom = (int)op_domain_[h].load(std::memory_order_relaxed);
    for (int b = 0; b < p.n_bands; ++b) {
        const IssuePlan& ip = issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS + dom];
        if (ip.row && ip.row->info) return ip.row->info->name;
    }
    return "?";
}

/* ------------------------------------------------------------------ the repeat check */
/* Runs the device kernel a second time on the bytes the first saw and byte-compares (see the file
 * header). Blind spots: under tensor parallelism both ranks re-launch together, so a collective
 * does not deadlock but the pair is measured, not either rank alone; and outputs that depend on an
 * operand over the restore cap. */
void Ctx::oracle_repeat(rad_op h, const IssuePlan& dip, int64_t seq, int n_opd) {
    OracleState& o = *oracle_;

    /* Any uncopied operand makes the issue unjudgeable: it is counted, not guessed at. The cap is
     * meant for read-only weights, but e.g. a GDN recurrent state is in/out and far over it; left
     * unrestored, the second launch would see an advanced state. Raise
     * RADIANCE_OP_REPEAT_MAXBYTES to cover such ops. */
    for (int i = 0; i < n_opd; ++i)
        if (o.skip[i]) { ++o.unchecked; return; }

    /* First result; moved aside because oracle_snap writes `post` again for the second launch. */
    oracle_snap(n_opd, true);
    std::vector<std::vector<uint8_t>> first;
    first.swap(o.post);

    /* Restore only operands whose bytes actually changed (decided by comparison, not the schema).
     * In-place ops such as {x, h} -> {x} must be restored or they report as non-deterministic,
     * while restoring everything would segfault writing a Tier::Mapped weight (a read-only mmap).
     *
     * Never restore a weight table: its snapshot is the stacked plane, and writing it back would
     * overwrite the pointer array with expert weights. */
    bool wrote[MAX_OPERANDS] = {false};
    for (int i = 0; i < n_opd; ++i) {
        if (o.alias[i] != i || o.skip[i] || o.span[i] <= 0 || o.tab[i]) continue;
        const std::vector<uint8_t>& A = o.pre[(size_t)i];
        const std::vector<uint8_t>& B = first[(size_t)i];
        if (A.empty() || A.size() != B.size()) continue;
        if (std::memcmp(A.data(), B.data(), A.size()) == 0) continue;
        wrote[i] = true;
        if (tbuf_[i].data) rad_memcpy_async(tbuf_[i].data, A.data(), A.size(), stream_);
    }
    rad_stream_sync(stream_);

    /* Same plan, args and stream; dip.args.t still points at the device tbuf_ (the oracle only
     * rebinds its own `ht`). */
    RadArgs a = dip.args;
    a.n_t = n_opd;
    const int rc = dip.launch(&a, cur_stream());
    if (rc < 0) {
        RAD_WARN("repeat #%lld op '%s': the second launch was refused with %s, so this issue says "
                 "nothing about determinism", (long long)seq, plans_[h].info->op.c_str(),
                 rad_strerror(rc));
        return;
    }
    oracle_snap(n_opd, true);

    std::string line;
    for (int i = 0; i < n_opd; ++i) {
        /* Only operands the kernel wrote; read-only ones are equal by construction. */
        if (!wrote[i]) continue;
        const std::vector<uint8_t>& A = first[(size_t)i];
        const std::vector<uint8_t>& B = o.post[(size_t)i];
        if (A.size() != B.size() || A.empty()) continue;
        if (std::memcmp(A.data(), B.data(), A.size()) == 0) continue;

        /* Count and locate differing elements: a last-bit difference points to reduction order,
         * widespread garbage to a cross-workgroup read. */
        const uint32_t dt = o.ht[i].dtype;
        const int64_t  n  = o.span[i];
        int64_t nbad = 0, at = -1;
        double  worst = 0, av = 0, bv = 0, scale = 0;
        for (int64_t k = 0; k < n; ++k) {
            double x = 0, y = 0;
            if (!elem(A.data(), dt, k, &x) || !elem(B.data(), dt, k, &y)) { nbad = -1; break; }
            if (x == y) continue;
            ++nbad;
            const double d = std::fabs(x - y);
            if (d > worst) { worst = d; at = k; av = x; bv = y; }
            scale = std::max(scale, std::fabs(y));
        }
        if (nbad < 0)
            line += fmt("  opd%d %-6s %s bytes differ (dtype not scalar)\n", i,
                        rad_dtype_name(dt), tensor_shape_str(o.ht[i]).c_str());
        else
            line += fmt("  opd%d %-6s %s  %lld of %lld element(s) differ; worst at [%lld] "
                        "run1 %.9g run2 %.9g (diff %.4g, rel %.3g)\n",
                        i, rad_dtype_name(dt), tensor_shape_str(o.ht[i]).c_str(),
                        (long long)nbad, (long long)n, (long long)at, av, bv, worst,
                        worst / (scale > 0 ? scale : 1.0));
    }
    if (line.empty()) return;
    ++o.differed;
    RAD_ERR("repeat #%lld op '%s' (kernel %s) is not a function of its inputs: two launches on the "
            "same bytes disagreed.\n%s", (long long)seq, plans_[h].info->op.c_str(),
            ip_kernel_name(h), line.c_str());
}

const IssuePlan* Ctx::oracle_reference(rad_op h, int band) {
    OracleState& o = *oracle_;
    const OpPlan& p = plans_[h];
    const IssuePlan& hip =
        issue_table_[(size_t)p.plan_off + (size_t)band * RAD_N_DOMAINS + RAD_DOMAIN_HOST];
    if (hip.launch && hip.row && hip.row->reference) return &hip;
    if (!registry_) return nullptr;

    const uint64_t key = ((uint64_t)h << 16) | (uint64_t)(uint16_t)band;
    if (const auto it = o.ref_at.find(key); it != o.ref_at.end())
        return it->second >= 0 ? &o.ref_plans[(size_t)it->second] : nullptr;
    o.ref_at[key] = -1;

    /* The question the bucket table asked of this band (Registry::build_bands), with every row
     * but a reference library's refused. */
    const OpInfo& oi = *p.info;
    Geometry g = oi.base;
    if (!oi.ranged_key.empty()) g.set_i(oi.ranged_key, oi.bands[(size_t)band].hi);
    const Registry::RowFilter only_reference = [](const KernelRow& r, const Geometry&) {
        return r.reference ? std::string() : std::string("not a reference library");
    };
    Resolved r = registry_->resolve(oi.op, g, RAD_DOMAIN_HOST, nullptr, &only_reference);
    if (!r || !r.row->info || !r.row->info->launch) return nullptr;
    if (r.row->info->scratch) {
        RadArgs a{};
        a.p          = r.geom.params();
        a.n_p        = r.geom.n_params();
        a.rank       = rank_;
        a.world_size = world_size_;
        const int64_t s = r.row->info->scratch(&a);
        if (s < 0) return nullptr;
        r.scratch_bytes = s;
    }
    if (registry_->init_instance(r, rank_, world_size_) < 0) return nullptr;

    o.ref_res.push_back(std::move(r));
    const Resolved& kept = o.ref_res.back();
    IssuePlan ip{};
    ip.row                = kept.row;
    ip.launch             = kept.row->info->launch;
    ip.args.p             = kept.geom.params();
    ip.args.n_p           = kept.geom.n_params();
    ip.args.scratch_bytes = kept.scratch_bytes;
    ip.args.instance      = kept.instance;
    ip.args.rank          = rank_;
    ip.args.world_size    = world_size_;
    o.ref_plans.push_back(ip);
    o.ref_at[key] = (int)o.ref_plans.size() - 1;
    return &o.ref_plans.back();
}

/* The entry point issue() calls. Returns the host plan to check against, or null. */
const IssuePlan* Ctx::oracle_before(rad_op h, int band, int n_opd) {
    if (!oracle_ || !oracle_arm(h)) return nullptr;
    oracle_size(h, n_opd);
    if (oracle_->repeat) {
        for (int i = 0; i < n_opd; ++i)
            if (tbuf_[i].data && oracle_opd_bytes(i) <= 0) return nullptr;
        oracle_snap(n_opd, false);
        /* Repeat mode needs no host kernel; this pointer only signals "armed". oracle_after
         * re-launches the device plan. */
        static const IssuePlan kRepeatArmed{};
        return &kRepeatArmed;
    }
    /* Not gated on the device domain: on the host backend this checks the host library that
     * serves the op (libavx) against the reference library. */
    const IssuePlan* ref = oracle_reference(h, band);
    if (!ref) {
        ++oracle_->skipped_nohost;
        if (!oracle_->said_nohost) {
            oracle_->said_nohost = true;
            RAD_WARN("oracle: op '%s' has no reference library row in this band, so it cannot be "
                     "checked. A reference kernel library (libref) must be on the kernel search "
                     "path for the oracle to see anything.", plans_[h].info->op.c_str());
        }
        return nullptr;
    }
    const IssuePlan& hip = *ref;
    /* An operand with no byte count (plugin-private dtype, not a sized weight; spec 19.6) is a
     * logged skip, so a gap is not mistaken for nothing to check. */
    for (int i = 0; i < n_opd; ++i) {
        if (!tbuf_[i].data || oracle_opd_bytes(i) > 0) continue;
        RAD_WARN("oracle #%lld op '%s': NOT CHECKED. operand %d is %s and the core cannot size "
                 "it -- a plugin-private dtype is the plugin's to measure, and nothing bound "
                 "this operand to a weight whose stored size the container records.",
                 (long long)n_issues_, plans_[h].info->op.c_str(), i,
                 rad_dtype_name(tbuf_[i].dtype));
        return nullptr;
    }
    /* An operand over maxopd is a logged skip: snapshotting it (e.g. a 47 GiB n-gram table) would
     * get the process OOM-killed with nothing logged. */
    for (int i = 0; i < n_opd; ++i) {
        if (!tbuf_[i].data || oracle_->alias[i] != i) continue;
        const int64_t b = oracle_opd_bytes(i);
        if (b <= oracle_->maxopd) continue;
        const OpInfo& oi = *plans_[h].info;
        const char* wn = nullptr;
        if (program_)
            for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j)
                if ((int)oi.weight_opd[j] == i && oi.weights[j]
                    && (size_t)oi.weights[j] < program_->weights.size())
                    wn = program_->weights[(size_t)oi.weights[j]].name.c_str();
        RAD_WARN("oracle #%lld op '%s': NOT CHECKED. operand %d%s%s%s is %lld MiB against a cap of "
                 "%lld MiB, and two host copies of it would end the run instead of reporting on "
                 "it. RADIANCE_OP_ORACLE_MAXBYTES raises the cap.",
                 (long long)n_issues_, oi.op.c_str(), i, wn ? " ('" : "", wn ? wn : "",
                 wn ? "')" : "", (long long)(b >> 20), (long long)(oracle_->maxopd >> 20));
        return nullptr;
    }
    oracle_snap(n_opd, false);
    return &hip;
}

void Ctx::oracle_after(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int n_opd) {
    if (oracle_->repeat) {
        oracle_repeat(h, dip, n_issues_, n_opd);
        --oracle_->budget;
        ++oracle_->checked;
        return;
    }
    oracle_snap(n_opd, true);
    /* `ht` still points into `pre`, except re-laid weights, which oracle_unrelayout repoints at
     * their inverse. An issue that cannot be inverted is neither checked nor counted. */
    if (!oracle_unrelayout(h, hip, dip, n_issues_, n_opd)) return;
    oracle_run(h, hip, n_issues_, n_opd);
    --oracle_->budget;
    ++oracle_->checked;
}

}  /* namespace rad */
