/* issue.cpp -- THE hot path, and the C surface the architecture plugin actually calls.
 *
 * Per rad_issue, in order: index the op's bucket table with `n`, pick the domain from the resolved
 * instance's current execution site, materialise a RadTensor for each operand out of a template
 * built at declare, resolve every weight handle to a pointer (waiting on the mover's event on the
 * STREAM if the weight is in flight), store the operand count into a RadArgs that was already
 * filled in, and launch.
 *
 * NO ALLOCATION. NO HOST SYNCHRONISATION. NO DEVICE ROUND TRIP. Every one of those is a property
 * you lose by accident, so each of them is checkable here: the only writes are into arrays sized
 * at prepare, the only device call besides the launch is rad_event_wait (which orders a stream and
 * does not block the host), and nothing in this file reads a device value back.
 *
 * See ctx.h for the per-issue cost and where it sits against spec §3.2's budget.
 */
#include <algorithm>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "ctx.h"
#include "device/device.h"
#include "wtab.h"

#include <cstring>

namespace rad {

/* A COLUMN SLICE: `cols` narrows the LAST dimension while every stride is left alone, so the
 * declared width stays the row pitch. That is what makes a fused projection usable without a
 * copy -- q and the attention gate are interleaved per head inside one buffer, and GDN's `a` and
 * `b` are columns of one in-projection. The element offset is what selects the first column, so
 * the two fields compose: offset picks the start, cols picks the width.
 *
 * Narrowing the last dimension rather than dim 0 is the whole point: shortening the OUTERMOST
 * dimension leaves every stride correct because they are products of the trailing ones, and
 * shortening the INNERMOST one leaves them correct because nothing is a product of it. Any other
 * dimension would need the strides rewritten, so the ABI offers only these two. */
static inline void narrow_cols(const RadOperand& o, RadTensor* t) {
    if (o.cols > 0 && t->rank >= 1) t->shape[t->rank - 1] = o.cols;
}

/* DOES THE OPERAND FIT THE BUFFER IT NAMES? The span is the largest element index the descriptor
 * can reach, measured from the buffer's base -- `offset` plus the sum of (extent - 1) * stride
 * over every dimension. It is NOT shape[0] against the declared shape[0]: a row count larger than
 * the declared one is legitimate when `cols` is narrower, because a [T, 2560] buffer read at 160
 * columns IS T*16 rows of 160 and the PLE gather does exactly that. The span is the only reading
 * that accepts the legal case and refuses the overrun.
 *
 * An empty extent reaches nothing and is not an error here; the kernel's own bounds decide. */
static inline bool operand_fits(const RadTensor& t, int64_t offset, const RadTensor& decl) {
    if (offset < 0) return false;
    int64_t span = offset;
    for (uint32_t d = 0; d < t.rank; ++d) {
        if (t.shape[d] <= 0) return true;
        span += (t.shape[d] - 1) * t.stride[d];
    }
    return span < rad_tensor_numel(&decl);
}

/* Extents and strides, for the message that refuses. oracle.cpp has one of these in an anonymous
 * namespace; duplicating four lines beats exporting a debug formatter across translation units. */
static inline std::string opd_shape_str(const RadTensor& t) {
    std::string s = "[";
    for (uint32_t d = 0; d < t.rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t.shape[d]);
    s += "]/[";
    for (uint32_t d = 0; d < t.rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t.stride[d]);
    return s + "]";
}

/* Element offset to byte offset. Split out because the sub-byte cases are the interesting ones:
 * a W4 weight has no addressable odd element, and a plugin-private dtype has no size the core
 * knows at all -- both are refused by name rather than silently truncated into a misaligned
 * pointer. Returns false when the offset cannot be expressed. */
static inline bool byte_delta(int bits, int64_t elems, int64_t* out) {
    if (bits >= 8) { *out = elems * (int64_t)(bits >> 3); return true; }
    if (bits <= 0) return false;                      /* plugin-private: the core has no size */
    const int64_t nbits = (int64_t)bits * elems;
    if (nbits & 7) return false;                      /* mid-byte offset into a packed weight */
    *out = nbits >> 3;
    return true;
}

void* Ctx::resolve_weight(rad_weight w) {
    /* The template already carries WeightInfo::ptr, bound at prepare and refreshed between steps
     * when the mover says it moved something. In the all-in-VRAM case that is the whole answer and
     * there is no table, no virtual call and no branch beyond this one. */
    void* p = w_tmpl_[w].data;
    if (!residency_) return p;

    const WeightSlot* s = residency_->lookup(w);       /* never blocks -- residency.h */
    if (s) {
        p = s->ptr;
        /* Wait once per generation, not once per issue. A stream wait orders everything issued
         * after it, so repeating it buys nothing and at eight experts a layer across 64 layers it
         * would be thousands of runtime calls a step. The mover bumps `generation` when it starts
         * a transfer, which is what makes the memo sound (residency.h). */
        if (s->ready && waited_gen_[w] != s->generation) {
            /* Not part of a recording: once issued it orders everything after it, every later
             * play included, and a move of a weight a pass names directly re-keys the pass. */
            if (rec_) tape_pause(true);
            rad_event_wait(cur_stream(), s->ready);         /* the STREAM waits; the host does not */
            if (rec_) tape_pause(false);
            waited_gen_[w] = s->generation;
            ++n_stream_waits_;
        }
    }
    return p;
}

/* ================================================================== weight tables
 *
 * RAD_OPK_WTAB. See the comment on Ctx::WeightTable and the RAD_WTAB macro in rad_runtime.h: this
 * is where a run of declared weights becomes the device pointer array a grouped GEMM walks.
 *
 * THE ENTRIES MUST AGREE ABOUT SHAPE AND DTYPE, and the check is not pedantry -- the kernel is
 * given one set of trailing extents for the whole table and strides its way through an entry with
 * them, so a run that mixes two shapes is a run the kernel reads off the end of. A MoE layer
 * declares 256 identical experts, so this holds by construction and a violation is a plugin bug
 * that is worth naming at the first issue rather than debugging from wrong logits. */
/* THE TABLE'S MEMORY, ALLOCATED ONCE AND NOT ON THE STEP PATH.
 *
 * Allocating it inside resolve_weight_table, which runs at ISSUE, would have the first step of the
 * first request allocate one pinned array and one device array per grouped-GEMM operand -- an
 * allocator on the step path, and spec §3.1 says there is not one. It would also be invisible to
 * the VRAM budget: the pools are sized before a step has ever run, so those bytes would come out
 * of --gpu-headroom-mib without being named in it.
 *
 * Binding every declared table at prepare() avoids both. The run is a property of the DECLARATION
 * (OpInfo::weights with OpInfo::weight_opd naming the operand position), not of the issue, so
 * there is nothing about it that has to wait for a step. resolve_weight_table still handles a
 * miss, because a plugin may legally issue a table the declaration did not describe -- it just
 * does not happen on any model this engine ships.
 *
 * Returns the slot in wtabs_. */
int Ctx::ensure_weight_table(uint64_t key, rad_weight first, int64_t n, size_t* slot_out) {
    auto it = wtab_of_.find(key);
    if (it != wtab_of_.end()) { *slot_out = it->second; return RAD_OK; }

    WeightTable t;
    t.key   = key;
    t.first = first;
    t.n     = n;
    t.host  = (void**)rad_dev_alloc(n * (int64_t)sizeof(void*), RAD_MEM_HOST_PINNED);
    t.dev   = rad_dev_alloc(n * (int64_t)sizeof(void*), RAD_MEM_DEVICE);
    if (!t.host || !t.dev) {
        if (t.host) rad_dev_free(t.host, RAD_MEM_HOST_PINNED);
        if (t.dev)  rad_dev_free(t.dev,  RAD_MEM_DEVICE);
        return RAD_E_NOMEM;
    }
    std::memset(t.host, 0, (size_t)n * sizeof(void*));
    t.pend.reserve((size_t)n);
    *slot_out = wtabs_.size();
    if (wmemb_head_.size() < program_->weights.size())
        wmemb_head_.resize(program_->weights.size(), -1);
    for (int64_t e = 0; e < n; ++e) {
        const rad_weight w = (rad_weight)(first + e);
        if ((size_t)w >= wmemb_head_.size()) break;
        wmemb_.push_back(WMemb{ (int32_t)*slot_out, (int32_t)e, wmemb_head_[w] });
        wmemb_head_[w] = (int32_t)wmemb_.size() - 1;
    }
    wtabs_.push_back(std::move(t));
    wtab_of_.emplace(key, *slot_out);
    return RAD_OK;
}

/* Every RAD_OPD_WTAB operand of every declared op, bound before the first step. The run is
 * contiguous by construction -- rad_decl_weight hands out consecutive indices and a declaration
 * loop over experts produces exactly such a run -- so it is the smallest declared handle at that
 * operand position and the count of them. A gap would be a plugin bug, and it is named rather
 * than papered over: a table assembled from a non-contiguous run would read the wrong expert. */
int Ctx::bind_weight_tables() {
    const Program& P = *program_;
    int64_t tables = 0, bytes = 0;
    for (size_t oi = 1; oi < P.ops.size(); ++oi) {
        const OpInfo& op = P.ops[oi];
        if (!op.schema || op.weights.size() != op.weight_opd.size()) continue;
        for (int p = 0; p < op.schema->n_operands; ++p) {
            if (op.schema->operands[p].role != RAD_OPD_WTAB) continue;
            rad_weight lo = 0, hi = 0;
            int64_t    cnt = 0;
            for (size_t k = 0; k < op.weights.size(); ++k) {
                if (op.weight_opd[k] != p) continue;
                const rad_weight w = op.weights[k];
                if (cnt == 0) { lo = hi = w; }
                else          { lo = std::min(lo, w); hi = std::max(hi, w); }
                ++cnt;
            }
            if (cnt <= 0) continue;
            if ((int64_t)(hi - lo) + 1 != cnt) {
                RAD_ERR("op '%s' operand %d is a weight table of %lld entries whose declared "
                        "handles run [%u, %u] -- a table must be a CONTIGUOUS run, because the "
                        "kernel is handed one base and strides through it",
                        op.op.c_str(), p, (long long)cnt, lo, hi);
                return RAD_E_INVAL;
            }
            size_t slot = 0;
            const uint64_t key = ((uint64_t)(rad_op)oi << 32) | (uint32_t)p;
            const int rc = ensure_weight_table(key, lo, cnt, &slot);
            if (rc < 0) {
                RAD_ERR("op '%s' operand %d: a table of %lld weights needs %s of pointers and it "
                        "could not be allocated", op.op.c_str(), p, (long long)cnt,
                        humanb(cnt * 2 * (int64_t)sizeof(void*)).c_str());
                return rc;
            }
            ++tables;
            bytes += cnt * 2 * (int64_t)sizeof(void*);
        }
    }
    if (tables > 0)
        RAD_DEBUG("weight tables: %lld bound at prepare, %s of pointer arrays -- none of it "
                  "allocated on the step path", (long long)tables, humanb(bytes).c_str());
    return RAD_OK;
}

static const bool g_dbg_issue = std::getenv("RADIANCE_DEBUG_ISSUE") != nullptr;
/* THE WEIGHT-TABLE WALK, charged separately from the rest of `own`. A grouped-GEMM operand names a
 * run of experts; the first issue walks every entry and later ones the entries the mover changed
 * (WeightTable::pend). Two counters -- entries walked, and the wall time of the walks -- so that
 * cost is read directly rather than inferred from the difference between two builds. */
static double g_wtab_us = 0.0, g_wtab_us_p = 0.0;
static long   g_wtab_e  = 0,   g_wtab_e_p  = 0;

int Ctx::resolve_weight_table(rad_op h, int operand, rad_weight first, int64_t n,
                              RadTensor* out) {
    const uint64_t key = ((uint64_t)h << 32) | (uint32_t)operand;
    size_t slot;
    auto it = wtab_of_.find(key);
    if (it == wtab_of_.end()) {
        /* A MISS IS NOT THE NORMAL PATH ANY MORE -- bind_weight_tables() bound every table the
         * declaration describes. Reaching here means a plugin issued a table at an operand
         * position its declaration did not name as one, which is legal and is served, but it is
         * an allocation on the step path and the log says so once. */
        RAD_DEBUG("op %u operand %d issued a weight table of %lld entries that the declaration did "
                  "not describe; allocating it now", (unsigned)h, operand, (long long)n);
        const int rc = ensure_weight_table(key, first, n, &slot);
        if (rc < 0)
            return abort_step_msg(h, RAD_E_NOMEM,
                                  fmt("operand %d is a table of %lld weights and the %lld bytes it "
                                      "needs could not be allocated", operand, (long long)n,
                                      (long long)(n * 2 * (int64_t)sizeof(void*))));
    } else {
        slot = it->second;
        if (wtabs_[slot].first != first || wtabs_[slot].n != n)
            return abort_step_msg(h, RAD_E_INVAL,
                                  fmt("operand %d named weights [%u, +%lld) and now names "
                                      "[%u, +%lld). A table's run is fixed by the declaration",
                                      operand, wtabs_[slot].first, (long long)wtabs_[slot].n,
                                      first, (long long)n));
    }
    WeightTable& t = wtabs_[slot];
    /* In a pass being recorded the refresh is host work: the recording is cut and the table named,
     * so a play refreshes it here, on this lane, with whatever the mover has changed by then. */
    if (rec_) {
        tape_cut();
        TapeStep st;
        st.kind = kTapeTable;
        st.lane = (uint8_t)lane_;
        st.opd = (int16_t)operand;
        st.a = (int32_t)slot;
        st.b = (int32_t)h;
        rec_->steps.push_back(st);
        tape_pause(true);
    }
    int rc = RAD_OK;
    if (!t.checked || t.full || !t.pend.empty()) rc = table_refresh(h, operand, t);
    if (rec_) tape_pause(false);
    if (rc < 0) return rc;
    return weight_table_tensor(h, operand, t, w_tmpl_[first], out);
}

int Ctx::table_refresh(rad_op h, int operand, WeightTable& t) {
    const RadTensor& e0 = w_tmpl_[t.first];
    const bool wt_time = g_dbg_issue && rank() == 0;   /* same rule: one thread accumulates */
    const auto t_wt0 = wt_time ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    auto resolve = [&](int64_t e, void** out) -> int {
        const rad_weight w = (rad_weight)(t.first + e);
        void* p = resolve_weight(w);
        if (!p)
            return abort_step_msg(h, RAD_E_STATE,
                                  fmt("weight '%s' (entry %lld of operand %d's table) has no "
                                      "pointer at issue: the load phase never bound it and no "
                                      "residency table claims it",
                                      program_->weights[w].name.c_str(), (long long)e, operand));
        *out = p;
        return RAD_OK;
    };

    int64_t walked = 0;
    if (!t.checked) {
        /* THE FIRST ISSUE: agreement, every entry, and the whole array up in one copy. `host` is
         * never written after this, so the copy reads what was resolved here however late the
         * card reaches it. */
        t.cur.assign((size_t)t.n, nullptr);
        for (int64_t e = 0; e < t.n; ++e) {
            const rad_weight w = (rad_weight)(t.first + e);
            const RadTensor& te = w_tmpl_[w];
            if (te.dtype != e0.dtype || te.rank != e0.rank)
                return abort_step_msg(h, RAD_E_SHAPE,
                                      fmt("operand %d is a weight table whose entry %lld ('%s') "
                                          "is %s rank %u while entry 0 ('%s') is %s rank %u; every "
                                          "entry of one table must have the same dtype and shape",
                                          operand, (long long)e, program_->weights[w].name.c_str(),
                                          rad_dtype_name(te.dtype), te.rank,
                                          program_->weights[t.first].name.c_str(),
                                          rad_dtype_name(e0.dtype), e0.rank));
            for (uint32_t d = 0; d < te.rank; ++d)
                if (te.shape[d] != e0.shape[d] || te.stride[d] != e0.stride[d])
                    return abort_step_msg(h, RAD_E_SHAPE,
                                          fmt("operand %d is a weight table whose entry %lld "
                                              "('%s') disagrees with entry 0 about axis %u",
                                              operand, (long long)e,
                                              program_->weights[w].name.c_str(), d));
            RAD_TRY(resolve(e, &t.cur[(size_t)e]));
            t.host[e] = t.cur[(size_t)e];
        }
        walked = t.n;
        const int rc = rad_memcpy_async(t.dev, t.host, t.n * (int64_t)sizeof(void*), cur_stream());
        if (rc < 0)
            return abort_step_msg(h, rc,
                                  fmt("operand %d: the weight table could not be uploaded", operand));
        t.checked = true;
    } else {
        /* THE ENTRIES THAT MAY HAVE MOVED, each resolved -- which is also what orders the stream
         * behind a transfer still in flight -- and written only if its pointer differs. */
        WtabPatch pt;
        auto one = [&](int64_t e) -> int {
            void* p = nullptr;
            RAD_TRY(resolve(e, &p));
            ++walked;
            if (t.cur[(size_t)e] == p) return RAD_OK;
            t.cur[(size_t)e] = p;
            pt.dst[pt.n] = static_cast<void**>(t.dev) + e;
            pt.val[pt.n] = p;
            if (++pt.n == (uint32_t)kWtabPatch) {
                RAD_TRY(wtab_patch(pt, cur_stream()));
                pt.n = 0;
            }
            return RAD_OK;
        };
        if (t.full) {
            for (int64_t e = 0; e < t.n; ++e) RAD_TRY(one(e));
        } else {
            for (int32_t e : t.pend) RAD_TRY(one(e));
        }
        const int rc = wtab_patch(pt, cur_stream());
        if (rc < 0)
            return abort_step_msg(h, rc,
                                  fmt("operand %d: the weight table could not be patched", operand));
    }
    t.full = false;
    t.pend.clear();
    if (wt_time) {
        g_wtab_us += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - t_wt0).count();
        g_wtab_e  += walked;
    }
    return RAD_OK;
}

/* The residency table's changes since the last pass, queued against every table that holds the
 * weight. A weight also resolved straight into an operand moves `dense_epoch_` as well. */
void Ctx::sync_residency() {
    if (!residency_) return;
    const std::vector<rad_weight>& ch = residency_->changes();
    if (ch.empty()) return;
    for (rad_weight w : ch) {
        if ((size_t)w < w_direct_.size() && w_direct_[w]) ++dense_epoch_;
        if ((size_t)w >= wmemb_head_.size()) continue;
        for (int32_t m = wmemb_head_[w]; m >= 0; m = wmemb_[(size_t)m].next) {
            WeightTable& t = wtabs_[(size_t)wmemb_[(size_t)m].slot];
            /* A table not yet issued resolves every entry at its first issue, and one already
             * marked resolves every entry at its next. */
            if (!t.checked || t.full) continue;
            if ((int64_t)t.pend.size() >= t.n) { t.full = true; t.pend.clear(); continue; }
            t.pend.push_back(wmemb_[(size_t)m].entry);
        }
    }
    residency_->clear_changes();
}

/* The operand a kernel is handed for a table: entry 0's dtype and trailing extents, with the
 * expert axis PREPENDED at stride zero -- the marker that says the axis is not addressable by
 * arithmetic and `data` is a pointer array. */
int Ctx::weight_table_tensor(rad_op h, int operand, const WeightTable& t, const RadTensor& e0,
                             RadTensor* out) {
    if (e0.rank + 1 > RAD_MAX_RANK)
        return abort_step_msg(h, RAD_E_SHAPE,
                              fmt("operand %d is a table of rank-%u weights, and one more axis "
                                  "does not fit RAD_MAX_RANK", operand, e0.rank));
    *out = e0;
    for (int d = (int)e0.rank; d >= 1; --d) {
        out->shape[d]  = e0.shape[d - 1];
        out->stride[d] = e0.stride[d - 1];
    }
    out->rank      = e0.rank + 1;
    out->shape[0]  = t.n;
    out->stride[0] = 0;
    out->data      = t.dev;
    return RAD_OK;
}

void Ctx::free_weight_tables() {
    for (WeightTable& t : wtabs_) {
        if (t.host) rad_dev_free(t.host, RAD_MEM_HOST_PINNED);
        if (t.dev)  rad_dev_free(t.dev,  RAD_MEM_DEVICE);
    }
    wtabs_.clear();
    wtab_of_.clear();
}

/* RADIANCE_DEBUG_ISSUE: WHAT THE ISSUE PATH ACTUALLY COSTS, split from the launch it wraps.
 *
 * A draft pass costs several microseconds a dispatch where a bare `hipLaunchKernelGGL` costs
 * around one and a half (`isa/graphrate.hip` prices one). The difference has to be here, and
 * nothing else in the tree can see it: an op's host cost is invisible in a kernel trace, and the
 * per-op profiler times the DEVICE.
 *
 * `own` is everything before the launch -- the band scan, the domain lookup, the operand
 * resolution loop. `launch` is the plugin shim plus HIP. Two clock reads an issue, ~50 ns on a
 * path this says is microseconds, and only when the variable is set.
 *
 * REPORTED PER WINDOW AND NOT AS A RUNNING MEAN. Load and the first prefill put seconds into the
 * accumulator over relatively few issues, so a cumulative average decays through a series of
 * individually plausible numbers, every one of them the startup transient and none of them the
 * steady state. A statistic that is still moving after a million samples is measuring the warm-up,
 * and printing it as "us each" invites exactly the wrong conclusion. */

/* RADIANCE_DEBUG_ARGSHA: ARE A DRAFTER PASS'S KERNEL ARGUMENTS THE SAME EVERY STEP?
 *
 * This is the one question that decides what a captured graph would cost to maintain. A draft pass
 * is the same op sequence at the same shapes every step -- the concurrency does not change inside a
 * request -- so if the ARGUMENTS are identical too, a graph can be captured once and replayed. If
 * they move, every replay needs `hipGraphExecKernelNodeSetParams` over ~50 nodes, which is most of
 * the submit cost back.
 *
 * The staging arena is DOUBLE BUFFERED, so the expectation is not one digest but TWO, alternating.
 * That still replays -- two graphs, picked by parity -- and it is worth knowing before anyone
 * builds either.
 *
 * Digests the operand bytes and `n` of every issue of pass 1 into one FNV-1a and prints it a step. */
static const bool g_dbg_argsha = std::getenv("RADIANCE_DEBUG_ARGSHA") != nullptr;

static int        g_argsha_pass = -99;
static long       g_argsha_seen = 0;
/* ONE DIGEST AN ISSUE, not one a pass: the pass-wide digest only says "something moved", and what
 * decides whether a captured graph is maintainable is HOW MANY nodes move. Kept per position in
 * the pass and compared against the previous step's. */
/* TWO digests, not one, and the op handle beside them. A count of moving nodes is enough to SIZE
 * a graph; it is not enough to DESIGN one. A node whose operand pointer moved can be left alone if
 * the pointer can be made stable -- the same address refilled in place -- and then the captured
 * graph replays untouched. A node whose SHAPE moved cannot: the launch geometry is baked into the
 * node at capture, so that one needs a real update, which needs launch params only the plugin shim
 * has. Which kind the ten are decides whether this is a contained change or an ABI change. */
struct ArgSha { rad_op h; uint64_t ptr; uint64_t shp; };
static std::vector<ArgSha> g_argsha_cur, g_argsha_prev;
static size_t              g_argsha_i = 0;
static ArgSha argsha_of(rad_op h, int64_t n, const RadTensor* t, int n_opd) {
    uint64_t v = 1469598103934665603ull, w = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t k) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < k; ++i) { v ^= b[i]; v *= 1099511628211ull; }
    };
    auto mixw = [&](const void* p, size_t k) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < k; ++i) { w ^= b[i]; w *= 1099511628211ull; }
    };
    mix(&h, sizeof h);
    mixw(&h, sizeof h);
    mixw(&n, sizeof n);   /* the row count IS geometry: it is what the grid is derived from */
    /* ONLY THE LIVE FIELDS. `RadTensor` carries shape[RAD_MAX_RANK] and stride[RAD_MAX_RANK] of
     * which only `rank` are meaningful, and `tbuf_` is REUSED across issues -- so hashing the
     * struct whole picks up whatever the last op left in the tail slots and reports very nearly
     * every issue as having moved, which is the instrument's own scratch and not the model's
     * arguments. */
    for (int i = 0; i < n_opd; ++i) {
        const RadTensor& tt = t[i];
        mix(&tt.data, sizeof tt.data);                 /* WHERE  -- a node-update-free fix exists */
        mixw(&tt.dtype, sizeof tt.dtype);              /* WHAT   -- baked into the node at capture */
        mixw(&tt.rank, sizeof tt.rank);
        for (uint32_t d = 0; d < tt.rank && d < RAD_MAX_RANK; ++d) {
            mixw(&tt.shape[d], sizeof tt.shape[d]);
            mixw(&tt.stride[d], sizeof tt.stride[d]);
        }
    }
    return ArgSha{h, v, w};
}
/* PER OP, because the average is the one number that cannot be acted on. Subtracting the price of
 * a bare fifteen-argument hipLaunchKernelGGL (isa/graphrate.hip has one) from `launch` leaves most
 * of a microsecond an issue in the shim's own prologue -- but that is a subtraction across two
 * different kernels, and an average over a step's ~1850 issues hides whether it is a flat tax or
 * three ops doing something silly. This charges host launch time to the op that spent it. */
struct IssOp { double lau = 0.0, own = 0.0; long n = 0; };
static std::unordered_map<uint32_t, IssOp> g_iss_by_op;
static double g_iss_own = 0.0, g_iss_lau = 0.0;
static double g_iss_own_p = 0.0, g_iss_lau_p = 0.0;
static long   g_iss_n   = 0;

/* THE STAGER'S TWO CALLS, around an op of a pass it armed for. Before the op, because what it
 * stages for this op has to be in the residency table before the op's tables are resolved -- and
 * sync_residency is what carries a table change into the tables that hold the weight, which the
 * pass otherwise does once, at its start. After the op, on the stream the op went to, because that
 * is where "every reader of this weight is behind us" can be recorded. */
int Ctx::issue(rad_op h, const RadOperand* opd, int n_opd, int64_t n) {
    if (!staging_) return issue_body(h, opd, n_opd, n);
    if (step_status_ < 0) return step_status_;
    bool resync = false;
    const int sb = stager_->before_op(h, cur_stream(), &resync);
    if (sb < 0)
        return abort_step_msg(h, sb, "the weight stager could not stage this op's weights");
    if (resync || (residency_ && !residency_->changes().empty())) sync_residency();
    const int rc = issue_body(h, opd, n_opd, n);
    if (rc < 0) return rc;
    const int sa = stager_->after_op(h, cur_stream());
    if (sa < 0)
        return abort_step_msg(h, sa, "the weight stager could not release this op's weights");
    return rc;
}

int Ctx::issue_body(rad_op h, const RadOperand* opd, int n_opd, int64_t n) {
    const auto t_iss0 = g_dbg_issue ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    /* The step is already aborted. rad_arch_step is void and cannot know, so it will keep issuing
     * the rest of the layer; launching those would burn a step whose output is already wrong and
     * bury the first error under fifty more. */
    if (step_status_ < 0) return step_status_;

    if (h == RAD_NULL_HANDLE || (size_t)h >= plans_.size())
        fatal_issue(h, n, "the plugin issued a handle declare never returned. An op that did not "
                          "resolve is RAD_NULL_HANDLE and becomes an error exactly here");

    OpPlan& p = plans_[h];

    /* RAD_N_BATCH is what almost every op wants: the ranged parameter is the step's token count. */
    if (n == RAD_N_BATCH) n = batch_ ? batch_->n_tok : 0;

    /* The bucket table. Bands are sorted by their inclusive upper bound and there are two to four
     * of them -- the union of the LE/GE/EQ values the candidate kernels placed on the ranged
     * parameter (spec §2.2). A linear scan is one to three predictable compares off one cache
     * line, which beats any table or division scheme at this size. */
    const int64_t* hi = band_table_.data() + p.band_off;
    int b = 0;
    while (b < p.n_bands && n > hi[b]) ++b;
    if (b == p.n_bands)
        fatal_issue(h, n, "falls outside every band of the op's bucket table -- the declared "
                          "range does not cover the value the run phase issued at");

    const int dom = (int)op_domain_[h].load(std::memory_order_relaxed);
    IssuePlan& ip = issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS + (size_t)dom];
    if (!ip.launch)
        fatal_issue(h, n, dom == RAD_DOMAIN_HOST
                              ? "has no host kernel for this band, and the placement planner put "
                                "it on the host anyway"
                              : "has no device kernel for this band");

    if (n_opd < 0 || n_opd > MAX_OPERANDS)
        return abort_step_msg(h, RAD_E_INVAL,
                              fmt("issued with %d operands; the runtime carries at most %d",
                                  n_opd, MAX_OPERANDS));
    if (p.n_opd_schema >= 0 && n_opd != p.n_opd_schema)
        return abort_step_msg(h, RAD_E_SCHEMA,
                              fmt("issued with %d operands; the op's schema declares %d. Operands "
                                  "are positional, so a miscount is silent numerical garbage",
                                  n_opd, (int)p.n_opd_schema));

    RadTensor* t = tbuf_;
    bool touches_host = false;   /* a host-domain buffer among the operands; see host_touch_ev_ */

    for (int i = 0; i < n_opd; ++i) {
        const RadOperand& o = opd[i];
        RadTensor&        tt = t[i];

        switch (o.kind) {
        case RAD_OPK_BUF: {
            const uint32_t bh = o.handle;
            if (bh == RAD_NULL_HANDLE || (size_t)bh >= buf_tmpl_.size())
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d names buffer handle %u, which declare never "
                                          "returned", i, bh));
            /* One struct copy: dtype, rank, shape and packed strides came from the declaration and
             * the data pointer is already arena_base + BufferInfo::arena_offset. */
            tt = buf_tmpl_[bh];
            touches_host |= buf_host_[bh] != 0;
            if (o.offset == 0 && o.cols <= 0 && buf_dense_[bh]) {
                if (o.rows > 0) {
                    if (o.rows > buf_tmpl_[bh].shape[0]) {
                        tt.shape[0] = o.rows;
                        return abort_step_msg(h, RAD_E_SHAPE,
                                              fmt("operand %d reaches past buffer '%s': %s at offset "
                                                  "%lld, declared %s", i,
                                                  program_->buffers[bh].name.c_str(),
                                                  opd_shape_str(tt).c_str(), (long long)o.offset,
                                                  opd_shape_str(buf_tmpl_[bh]).c_str()));
                    }
                    tt.shape[0] = o.rows;
                }
            } else {
                if (o.offset) {
                    int64_t d;
                    if (!byte_delta(buf_bits_[bh], o.offset, &d))
                        return abort_step_msg(h, RAD_E_ALIGN,
                                              fmt("operand %d offsets buffer '%s' by %lld elements, "
                                                  "which is not a whole number of bytes in %s", i,
                                                  program_->buffers[bh].name.c_str(),
                                                  (long long)o.offset,
                                                  rad_dtype_name(tt.dtype)));
                    tt.data = static_cast<char*>(tt.data) + d;
                }
                /* rows > 0 narrows dim 0, which is how a prefill chunk of `rows` tokens issues against
                 * a buffer the plan sized at max_tok. The strides do not change and must not: they are
                 * products of the trailing dimensions, so shortening the outermost one leaves every
                 * one of them correct. That is the whole reason the narrowing is dim 0 only. */
                if (o.rows > 0) tt.shape[0] = o.rows;
                narrow_cols(o, &tt);
                /* AND IT HAS TO FIT THE BUFFER IT NAMES. Narrowing dim 0 is how a chunk of `rows`
                 * tokens issues against a plan sized at max_tok, and the arrow points one way only:
                 * an operand may name FEWER rows than the declaration, never more.
                 *
                 * A read past the end is uninitialised memory. A WRITE past the end is another
                 * buffer's bytes, and the arena packs by liveness -- so what it lands on is whatever
                 * the planner put there, which changes with the graph. That is the worst shape of
                 * bug this runtime can have: it is silent, it is not reproducible across a config
                 * change, and the model can stay correct for as long as the victim happens to be
                 * dead. The canonical shape of it is a MoE expert plane declared at max_tok rows while
                 * the gated quantiser writes rows*top_k of them, which nothing notices until some
                 * later op happens to read the buffer the overrun lands on. */
                if (!operand_fits(tt, o.offset, buf_tmpl_[bh]))
                    return abort_step_msg(h, RAD_E_SHAPE,
                                          fmt("operand %d reaches past buffer '%s': %s at offset "
                                              "%lld, declared %s", i,
                                              program_->buffers[bh].name.c_str(),
                                              opd_shape_str(tt).c_str(), (long long)o.offset,
                                              opd_shape_str(buf_tmpl_[bh]).c_str()));
            }
            /* AND THE OP HAS TO BE INSIDE THE BUFFER'S PLANNED LIFETIME.
             *
             * rad_op_reads/rad_op_writes are what the buffer plan is computed from, and they are
             * OPT-IN per op -- a buffer nothing declared gets the whole program and shares nothing,
             * which is the sound default. What is NOT sound is an op that touches a buffer other
             * ops DID declare, outside the range those declarations produced: the arena has laid
             * something else over those bytes, and what comes back is whatever the planner put
             * there. It reads as data, so nothing downstream can tell.
             *
             * This defect class recurs, and its shape is always the same: an op acquires a read
             * its declaration never mentioned. `attn_paged` reading the QSA selection is the
             * pattern -- the block table arrives as a raw batch pointer, so the op's declared read
             * set is just the query, and a page id fetched out of a reused buffer is an address
             * outside every mapping. It survives every step until a wider buffer changes the
             * packing.
             *
             * REFUSED RATHER THAN WARNED, because the failure it prevents is silent and is not
             * reproducible across a graph change -- the victim buffer is whatever the plan chose
             * this time. */
            {
                const BufferInfo& bi = program_->buffers[bh];
                if (bi.first_def >= 0 &&
                    ((int32_t)h < bi.first_def || (int32_t)h > bi.last_use))
                    return abort_step_msg(h, RAD_E_STATE,
                                          fmt("operand %d names buffer '%s', which this op never "
                                              "declared: the plan gives it ops %d..%d and this is "
                                              "op %u, so the arena may have reused those bytes. "
                                              "Call rad_op_reads/rad_op_writes for it at declare",
                                              i,
                                              bi.name.c_str(), (int)bi.first_def,
                                              (int)bi.last_use, (unsigned)h));
            }
            break;
        }

        case RAD_OPK_WEIGHT: {
            const uint32_t wh = o.handle;
            if (wh == RAD_NULL_HANDLE || (size_t)wh >= w_tmpl_.size())
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d names weight handle %u, which declare never "
                                          "returned", i, wh));
            tt      = w_tmpl_[wh];
            tt.data = resolve_weight(wh);
            w_direct_[wh] = 1;
            if (!tt.data)
                return abort_step_msg(h, RAD_E_STATE,
                                      fmt("weight '%s' has no pointer at issue: the load phase "
                                          "never bound it and no residency table claims it",
                                          program_->weights[wh].name.c_str()));
            if (o.offset) {
                int64_t d;
                if (!byte_delta(w_bits_[wh], o.offset, &d))
                    return abort_step_msg(h, RAD_E_ALIGN,
                                          fmt("operand %d offsets weight '%s' by %lld elements, "
                                              "which is not a whole number of bytes in %s", i,
                                              program_->weights[wh].name.c_str(),
                                              (long long)o.offset, rad_dtype_name(tt.dtype)));
                tt.data = static_cast<char*>(tt.data) + d;
            }
            if (o.rows > 0) tt.shape[0] = o.rows;
            narrow_cols(o, &tt);
            /* THE SAME FIT A BUFFER OPERAND HAS TO PASS, for the same reason. A slice that reaches
             * past the declared weight reads whatever the loader or the mover put after it -- the
             * next weight in the slab, or the end of the allocation -- and a GEMM over it produces
             * plausible numbers. Only a sliced operand can overrun, so the whole-weight case every
             * RAD_W issue takes skips the arithmetic. */
            if ((o.offset || o.rows > 0 || o.cols > 0) &&
                !operand_fits(tt, o.offset, w_tmpl_[wh]))
                return abort_step_msg(h, RAD_E_SHAPE,
                                      fmt("operand %d reaches past weight '%s': %s at offset "
                                          "%lld, declared %s", i,
                                          program_->weights[wh].name.c_str(),
                                          opd_shape_str(tt).c_str(), (long long)o.offset,
                                          opd_shape_str(w_tmpl_[wh]).c_str()));
            break;
        }

        case RAD_OPK_WTAB: {
            /* A RUN of declared weights, handed over as a device array of pointers. `handle` is
             * the first and `rows` is the count -- rad_decl_weight gives out consecutive indices,
             * so a declaration loop over experts produces exactly such a run. */
            const uint32_t w0  = o.handle;
            const int64_t  cnt = o.rows;
            if (w0 == RAD_NULL_HANDLE || cnt <= 0 ||
                (size_t)((int64_t)w0 + cnt) > w_tmpl_.size())
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d names a weight table of %lld entries from "
                                          "handle %u; declare handed out %zu weights",
                                          i, (long long)cnt, w0, w_tmpl_.size() - 1));
            if (o.offset || o.cols)
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d is a weight table and carries an offset or "
                                          "a column slice; neither has a meaning for a table -- "
                                          "slice the entries at declare instead", i));
            const int rc = resolve_weight_table(h, i, (rad_weight)w0, cnt, &tt);
            if (rc < 0) return rc;
            break;
        }

        case RAD_OPK_RAW:
            /* A pointer the core did not allocate -- a RadBatch field, almost always. It has no
             * BufferInfo, so there is no dtype and no shape to derive: the kernel knows both from
             * the op's schema, which is where that knowledge belongs. The core carries the pointer
             * and, if the caller gave one, the row count. */
            if (o.offset && o.dtype == RAD_DT_INVALID)
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d is a raw pointer with an element offset of "
                                          "%lld and no dtype. Give the operand a dtype with "
                                          "RAD_P_T() or do the arithmetic on the caller's side; "
                                          "the core cannot size an element it was never told about",
                                          i, (long long)o.offset));
            std::memset(&tt, 0, sizeof tt);
            tt.data  = o.raw;
            tt.dtype = o.dtype;   /* RAD_DT_INVALID unless the caller said, which is the point of
                                   * the field: libref reads positions, cu_seqlens and
                                   * num_accepted through the tensor's dtype and cannot guess. */
            /* `cols` on a RAW operand is the SECOND EXTENT, not a column slice: there is no
             * declared width to slice out of. RAD_P_T2 is how a batch field that is genuinely
             * two-dimensional says so, and the block table is why it exists -- [n_seq, max_blocks]
             * handed over as rank 1 loses its width, and every kernel that reads it then guesses. */
            if (o.rows > 0 && o.cols > 0) {
                tt.rank = 2;
                tt.shape[0] = o.rows;   tt.shape[1] = o.cols;
                tt.stride[0] = o.cols;  tt.stride[1] = 1;
            } else if (o.rows > 0) {
                tt.rank = 1; tt.shape[0] = o.rows; tt.stride[0] = 1;
            }
            break;

        case RAD_OPK_KV: {
            /* A KV group's cache for one layer. `handle` is the rad_kvgroup and `offset` is the
             * ABSOLUTE layer index, which the binding rebases -- a group holds one cache per bound
             * layer, not one per model layer, and 48 GDN layers interleaved with 16 attention ones
             * would otherwise index a 64-deep pool that was never allocated. */
            const uint32_t g = o.handle;
            if ((size_t)g >= kv_.size() || !kv_[g].base)
                return abort_step_msg(h, RAD_E_STATE,
                                      fmt("operand %d names KV group %u, which has no pool bound. "
                                          "The block manager allocates it at load and binds it "
                                          "into the Ctx; a group with none is a startup bug, not "
                                          "a runtime condition", i, g));
            const KVPoolBinding& b = kv_[g];
            /* A POSITION LOOKUP, not `o.offset - first_layer`. The subtraction is right only for a
             * group whose layers are contiguous, and they need not be: with 15 attention layers
             * interleaved through 64, absolute layer 19 -- the fifth of them -- subtracts to slot
             * 16 of a 15-slot pool, which the bounds check below refuses. */
            const int64_t layer = (o.offset >= 0 && (size_t)o.offset < b.layer_slot.size())
                                      ? b.layer_slot[(size_t)o.offset]
                                      : -1;
            const int64_t off   = layer * b.layer_stride;
            if (layer < 0 || (b.bytes && off + b.layer_stride > b.bytes) ||
                (size_t)layer >= b.layer_tensor.size())
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d asks KV group %u for layer %lld, which is "
                                          "outside the %lld bytes it was allocated", i, g,
                                          (long long)o.offset, (long long)b.bytes));
            tt = b.layer_tensor[(size_t)layer];
            if (o.rows > 0) tt.shape[0] = o.rows;
            break;
        }

        default:
            /* RAD_OPK_NONE: an absent optional operand is a null RadTensor.data, per rad_abi.h. */
            std::memset(&tt, 0, sizeof tt);
            break;
        }
    }

    /* Everything else in RadArgs -- the geometry view, the scratch pointer and its size, the
     * instance init() returned, the rank and the world size -- was filled in at prepare and has
     * not changed. args.t already points at tbuf_. */
    ip.args.n_t = n_opd;
    if (!p.issued) {
        p.issued = true;
        first_issued_.push_back(h);
    }
    ++n_issues_;

    /* RADIANCE_DEBUG_ARGSHA -- see the note by the digest. Pass 1 only, so the sequence compared
     * step to step is the same sequence. */

    /* RANK 0 ONLY. These accumulators are file-scope statics and `Ctx::issue` runs on EVERY rank
     * thread, so without this the two ranks interleave into one sequence: the issue COUNT then
     * varies run to run and every digest differs, which reads exactly like a data-dependent op
     * sequence and is not one. */
    if (g_dbg_argsha && batch_ && rank() == 0) {
        const int dp = batch_->draft_pass;
        if (dp != g_argsha_pass) {
            if (g_argsha_pass == 1 && !g_argsha_cur.empty() && g_argsha_seen < 12) {
                size_t mptr = 0, mshp = 0;
                const size_t k = g_argsha_cur.size() < g_argsha_prev.size()
                                     ? g_argsha_cur.size() : g_argsha_prev.size();
                for (size_t i = 0; i < k; ++i) {
                    if (g_argsha_cur[i].ptr != g_argsha_prev[i].ptr) ++mptr;
                    if (g_argsha_cur[i].shp != g_argsha_prev[i].shp) ++mshp;
                }
                if (!g_argsha_prev.empty()) {
                    std::fprintf(stderr,
                                 "D argsha: draft pass 1 -- %zu issues, %zu moved a POINTER, "
                                 "%zu moved a SHAPE\n",
                                 g_argsha_cur.size(), mptr, mshp);
                    if (g_argsha_seen < 2) {
                        auto nm = [&](rad_op oh) -> const char* {
                            return (oh != RAD_NULL_HANDLE && (size_t)oh < plans_.size() &&
                                    plans_[oh].info)
                                       ? plans_[oh].info->op.c_str() : "<?>";
                        };
                        for (size_t i = 0; i < k; ++i) {
                            const bool p = g_argsha_cur[i].ptr != g_argsha_prev[i].ptr;
                            const bool q = g_argsha_cur[i].shp != g_argsha_prev[i].shp;
                            if (p || q)
                                std::fprintf(stderr, "D argsha:   %2zu %-24s %s%s\n", i,
                                             nm(g_argsha_cur[i].h), p ? "ptr " : "",
                                             q ? "shape" : "");
                        }
                    }
                }
                ++g_argsha_seen;
                g_argsha_prev = g_argsha_cur;
            } else if (g_argsha_pass == 1) {
                g_argsha_prev = g_argsha_cur;
            }
            if (dp == 1) { g_argsha_cur.clear(); g_argsha_i = 0; }
            g_argsha_pass = dp;
        }
        if (dp == 1) { g_argsha_cur.push_back(argsha_of(h, n, t, n_opd)); ++g_argsha_i; }
    }

    /* THE LINK BILL, charged where the message size is known and nothing has to be guessed: `n`
     * IS the collective's `numel` (every all_reduce declares it as the ranged parameter) and
     * operand 0 carries the dtype it is sent in, which is the ACTIVATION's dtype and not the op's
     * `dtype` key -- that key is the architecture's operand-combination spelling ("w4a8"), which
     * is not an element width. Two relaxed adds on seventeen issues a step. */
    if (p.collective && n_opd > 0) {
        const int bits = rad_dtype_bits(t[0].dtype);
        ar_calls_.fetch_add(1, std::memory_order_relaxed);
        if (bits > 0) ar_bytes_.fetch_add(n * bits / 8, std::memory_order_relaxed);
    }

    /* The oracle, if this issue is one it was asked about. It snapshots every operand to the host
     * before the launch so the host kernel can be given the same bytes -- an in-place op would
     * otherwise be compared against its own output. Null in every run that did not arm it. */
    const IssuePlan* horacle = oracle_ ? oracle_before(h, b, n_opd) : nullptr;

    /* ============================== A HOST LAUNCH IS A JOIN ==============================
     *
     * THE HOST THREAD RUNS AHEAD OF THE DEVICE. Every device launch is asynchronous, so by the
     * time this function is called for op N the GPU may still be somewhere back at op N-200 --
     * that gap is the whole reason the issue path is allowed to cost a microsecond. A HOST-SITE op
     * is the one place it stops being harmless: it runs NOW, on this thread, and reads and writes
     * the host arena NOW.
     *
     * A PLE gather on the host is exactly that shape. `ngram_ids` writes the hashed row ids into a
     * host-domain buffer on the device, and `embed_lookup_q` reads them on the host. With no join
     * the host reads whatever the arena holds from an earlier step and gathers the rows of the
     * wrong n-gram -- valid ids, valid rows, every kernel agreeing with its oracle, and fluent text
     * that is a stable wrong answer rather than an intermittent one.
     *
     * WHAT THE JOIN WAITS FOR is the last device op on each lane that touched the host arena
     * (host_touch_ev_), not the whole stream. Every hazard a host op has is with such an op -- the
     * producer of an input, or a device reader of bytes its output reuses -- and on one lane they
     * all complete no later than the last of them. Draining the stream instead waits for
     * everything issued before the op, which on a decode step is every layer in front of it, and
     * then the device runs dry while the host works, because nothing after the host op has been
     * issued yet. Waiting on the producer alone lets an architecture issue it early and keep the
     * device busy with the layers issued in between while the host gathers.
     *
     * AN OPERAND THE ARENA DOES NOT DESCRIBE IS NOT TRACKED -- a raw batch pointer, a KV cache, a
     * device-domain buffer -- so a host op that names one drains its stream (host_join).
     *
     * AND THE HOST'S WRITES HAVE TO REACH THE DEVICE. Host-arena memory is cached on the device like
     * any other memory, and with no host wait between the host op and the kernel that reads its
     * output, nothing would make that kernel look past a stale line: rad_dev_host_wrote, after the
     * launch below, is what does. */
    if (dom == RAD_DOMAIN_HOST) {
        /* TIMED APART FROM THE OP, because it is not the op's cost -- it is the cost of the
         * PIPELINE this op stops. */
        if (profiling_) {
            const auto j0 = std::chrono::steady_clock::now();
            const int js = host_join(opd, n_opd);
            const auto j1 = std::chrono::steady_clock::now();
            prof_join_ms_[(size_t)h] += std::chrono::duration<double, std::milli>(j1 - j0).count();
            if (js < 0) return abort_step(h, b, dom, ip, n, js);
        } else {
            const int js = host_join(opd, n_opd);
            if (js < 0) return abort_step(h, b, dom, ip, n, js);
        }
    }

    /* A SCRATCH REGION IS ONE STREAM'S. The kernels on a lane are handed consecutive slices of it
     * (scratch_take), and two that could run at once on the lane are kept apart by the slices
     * being disjoint -- the pass's recording orders any pair whose slices meet (haz_mark). A
     * scratch user on each lane with no join between them overlaps with nothing recorded to order
     * it, and the two would overwrite each other's partials with nothing downstream able to tell.
     * A split-K router GEMM on lane 0 and the shared arm's split-K GEMM on lane 1 meet exactly
     * like that on every MoE layer.
     *
     * So a program that declares its second lane gets a region per lane, and each launch is
     * pointed at its own lane's. A lane used WITHOUT that declaration shares region 0, and there
     * the second of such a pair is ordered behind the first instead -- the overlap is given up for
     * that launch, never the numbers. That fallback is kept on lane 0 before lane 1 exists as well:
     * a lane built later starts with nothing ordering it behind what lane 0 already issued. */
    if (ip.args.scratch_bytes > 0 && dom != RAD_DOMAIN_HOST && arena_.scratch_regions() > 1) {
        /* Each lane has its own region, so two scratch users on two lanes never share bytes and
         * nothing has to be ordered. The plan entry serves whichever lane issues it. */
        ip.args.scratch = scratch_take(lane_, ip.args.scratch_bytes);
    } else if (ip.args.scratch_bytes > 0 && dom != RAD_DOMAIN_HOST) {
        ip.args.scratch = scratch_take(0, ip.args.scratch_bytes);
        const int other = lane_ ^ 1;
        if (scratch_open_[other]) {
            if (!scratch_join_warned_) {
                scratch_join_warned_ = true;
                RAD_WARN("op '%s' (%s, n=%lld) uses scratch on lane %d while '%s' (n=%lld) on lane "
                         "%d, also a scratch user, is not ordered before it; lane %d now waits for "
                         "lane %d there, which costs the overlap at that point",
                         p.info->op.c_str(),
                         ip.row && ip.row->info ? ip.row->info->name : "?", (long long)n, lane_,
                         scratch_kernel_[other] ? scratch_kernel_[other] : "?",
                         (long long)scratch_n_[other], other, lane_, other);
            }
            const int js = lane_join(other, lane_);
            if (js < 0) return abort_step(h, b, dom, ip, n, js);
        }
        scratch_open_[lane_] = true;
        scratch_kernel_[lane_] = ip.row && ip.row->info ? ip.row->info->name : nullptr;
        scratch_n_[lane_] = n;
    }

    /* A HOST-SITE OP IN A PASS BEING RECORDED is host work a tape cannot hold: the recording is
     * cut here and the op kept, to be joined and run again on these operands at every play. Its
     * launch is not recorded either way -- anything it submitted would be submitted twice. */
    if (rec_ && dom == RAD_DOMAIN_HOST) {
        tape_cut();
        TapeHost th;
        th.h = h;
        th.launch = ip.launch;
        th.args = ip.args;
        th.n = n;
        th.drain = host_join_drains(opd, n_opd);
        for (int i = 0; i < n_opd; ++i) th.t[i] = t[i];
        TapeStep st;
        st.kind = kTapeHost;
        st.lane = (uint8_t)lane_;
        st.a = (int32_t)rec_->host.size();
        rec_->host.push_back(th);
        rec_->steps.push_back(st);
    }
    const bool host_paused = rec_ && dom == RAD_DOMAIN_HOST;
    if (host_paused) tape_pause(true);
    const int rec0 = rec_ && !host_paused ? tape_mark() : 0;

    int s;
    if (profiling_ && dom == RAD_DOMAIN_HOST) {

        /* A HOST-SITE OP IS TIMED ON THE HOST CLOCK, because device events cannot see it: they
         * record when the STREAM reaches them and the stream does not stall for a host call, so an
         * event pair around this launch measures nothing and the op vanishes from the table.
         *
         * Without this the most expensive thing in the step vanishes from the table. The PLE
         * gather reads sixteen 160-byte rows out of a multi-gigabyte mapping per token, so its
         * layer costs milliseconds a step against tens of microseconds of device kernels -- and a
         * device-event profile shows only the tens of microseconds. A host kernel's cost is a
         * STALL in the middle of the step, which is the one thing an operator most needs named.
         *
         * Wall clock and not a device event, so it includes whatever the host call waits on:
         * page faults, the link read of its input, everything. That is the number that matters. */
        const auto t0 = std::chrono::steady_clock::now();
        s = ip.launch(&ip.args, cur_stream());
        const auto t1 = std::chrono::steady_clock::now();
        prof_ms_[(size_t)h] +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        prof_calls_[(size_t)h] += 1;
        if ((size_t)h < prof_host_.size()) prof_host_[(size_t)h] = 1;
    } else if (profiling_) {
        /* One predictable branch in exchange for §16's per-op timing. The events go on the compute
         * stream around the launch; the synchronisation that reads them is at the end of the step,
         * and it is what makes a profiled run incomparable to an unprofiled one. */
        if (prof_n_ < prof_ring_.size()) {
            ProfSlot& ps = prof_ring_[prof_n_++];
            ps.op = (int32_t)h;
            rad_event_record(ps.a, cur_stream());
            s = ip.launch(&ip.args, cur_stream());
            rad_event_record(ps.b, cur_stream());
        } else {
            prof_overflowed_ = true;
            s = ip.launch(&ip.args, cur_stream());
        }
    /* RANK 0 ONLY, FOR THE SECOND TIME IN THIS FILE. `Ctx::issue` runs on every rank thread, so a
     * file-scope accumulator is written by both. Unguarded, the digest above gives wrong NUMBERS
     * -- two ranks interleaving into one sequence -- and the per-op std::unordered_map here gives
     * a SEGFAULT within a window. Doubles race benignly enough to mislead, a hash map does not
     * race at all. Anything this instrument accumulates belongs to one thread. */
    } else if (g_dbg_issue && rank() == 0) {
        const auto ta = std::chrono::steady_clock::now();
        s = ip.launch(&ip.args, cur_stream());
        const auto tb = std::chrono::steady_clock::now();
        using us = std::chrono::duration<double, std::micro>;
        g_iss_own += us(ta - t_iss0).count();
        g_iss_lau += us(tb - ta).count();
        IssOp& io = g_iss_by_op[(uint32_t)h];
        io.own += us(ta - t_iss0).count();
        io.lau += us(tb - ta).count();
        ++io.n;
        if (++g_iss_n % 200000 == 0) {
            const double w = 200000.0;
            std::fprintf(stderr, "D issue: %ld issues -- this window: own %.3f us, launch %.3f us "
                                 "an issue (%.3f total)\n",
                         g_iss_n, (g_iss_own - g_iss_own_p) / w, (g_iss_lau - g_iss_lau_p) / w,
                         (g_iss_own - g_iss_own_p + g_iss_lau - g_iss_lau_p) / w);
            /* The walk is inside `own`, so this says how much of `own` it IS. */
            std::fprintf(stderr, "D issue:   of which weight tables: %.3f us an issue over "
                                 "%.1f entries (%.1f ns an entry)\n",
                         (g_wtab_us - g_wtab_us_p) / w,
                         (double)(g_wtab_e - g_wtab_e_p) / w,
                         (g_wtab_e - g_wtab_e_p) > 0
                             ? (g_wtab_us - g_wtab_us_p) * 1000.0 / (double)(g_wtab_e - g_wtab_e_p)
                             : 0.0);
            /* Top by TOTAL host launch time, not by the per-issue mean: an op issued 48 times a
             * step at 2 us costs more than one issued twice at 20. Both columns are printed so a
             * fat prologue and a fat call count can be told apart. */
            std::vector<std::pair<double, uint32_t>> ord;
            ord.reserve(g_iss_by_op.size());
            for (const auto& kv : g_iss_by_op) ord.push_back({kv.second.lau, kv.first});
            std::sort(ord.begin(), ord.end(), std::greater<>());
            for (size_t i = 0; i < ord.size() && i < 12; ++i) {
                const IssOp& io = g_iss_by_op[ord[i].second];
                const OpInfo* oi = (size_t)ord[i].second < plans_.size()
                                       ? plans_[ord[i].second].info : nullptr;
                std::fprintf(stderr,
                             "D issue:   %-26s %7ld issues  launch %6.3f us  own %6.3f us  "
                             "(%8.1f us of launch in all)\n",
                             oi ? oi->op.c_str() : "<?>", io.n, io.lau / (double)io.n,
                             io.own / (double)io.n, io.lau);
            }
            g_iss_by_op.clear();
            g_wtab_us_p = g_wtab_us;
            g_wtab_e_p  = g_wtab_e;
            g_iss_own_p = g_iss_own;
            g_iss_lau_p = g_iss_lau;
        }
    } else {
        s = ip.launch(&ip.args, cur_stream());
    }
    if (host_paused) tape_pause(false);

    if (s < 0) return abort_step(h, b, dom, ip, n, s);
    if (rec_ && !host_paused) haz_note(p, ip, t, n_opd, rec0, tape_mark());

    if (dom == RAD_DOMAIN_HOST) {
        rad_dev_host_wrote();
    } else if (touches_host) {
        const int l = lane_;
        if (!host_touch_ev_[l]) {
            const int ec = rad_event_create(&host_touch_ev_[l]);
            if (ec < 0) return abort_step(h, b, dom, ip, n, ec);
        }
        const int er = rad_event_record(host_touch_ev_[l], cur_stream());
        if (er < 0) return abort_step(h, b, dom, ip, n, er);
        host_touch_pending_[l] = true;
    }
    if (horacle) oracle_after(h, *horacle, ip, n_opd);
    return RAD_OK;
}

/* The next slice of a scratch region; see scratch_cur_ in ctx.h. */
void* Ctx::scratch_take(int region, int64_t bytes) {
    constexpr int64_t kAlign = 256;
    const int64_t need = (bytes + kAlign - 1) / kAlign * kAlign;
    int64_t off = scratch_cur_[region];
    if (off + need > arena_.scratch_bytes()) off = 0;
    scratch_cur_[region] = off + need;
    return (char*)arena_.scratch(region) + off;
}

/* What one device op of a pass being recorded reads and writes; see HazOp in ctx.h. */
void Ctx::haz_note(const OpPlan& p, const IssuePlan& ip, const RadTensor* t, int n_opd, int rec0,
                   int rec1) {
    if (rec1 <= rec0) return;
    HazOp o;
    o.rec0 = rec0;
    o.rec1 = rec1;
    o.op = (uint32_t)(&p - plans_.data());
    const RadOpSchema* sc = p.info ? p.info->schema : nullptr;
    o.opaque = p.collective || !sc || sc->n_operands != n_opd || !ip.row || !ip.row->concurrent;
    std::pair<uintptr_t, uintptr_t> rd[MAX_OPERANDS + 1], wr[MAX_OPERANDS + 1];
    int n_rd = 0, n_wr = 0;
    for (int i = 0; i < n_opd && !o.opaque; ++i) {
        const RadTensor& tt = t[i];
        if (!tt.data) continue;
        const int role = sc->operands[i].role;
        if (role == RAD_OPD_WEIGHT) continue;
        const uintptr_t lo = (uintptr_t)tt.data;
        if (role == RAD_OPD_WTAB) {
            /* The pointer array itself; the experts it names are weights. */
            if (tt.rank < 1 || tt.shape[0] <= 0) { o.opaque = true; break; }
            rd[n_rd++] = { lo, lo + (uintptr_t)tt.shape[0] * sizeof(void*) };
            continue;
        }
        const int bits = rad_dtype_bits(tt.dtype);
        if (tt.rank == 0 || bits <= 0) { o.opaque = true; break; }
        int64_t span = 1;
        bool empty = false;
        for (uint32_t d = 0; d < tt.rank; ++d) {
            if (tt.shape[d] <= 0) { empty = true; break; }
            if (tt.stride[d] < 0) { o.opaque = true; break; }
            span += (tt.shape[d] - 1) * tt.stride[d];
        }
        if (o.opaque) break;
        if (empty) continue;
        const std::pair<uintptr_t, uintptr_t> r{ lo, lo + (uintptr_t)((span * bits + 7) / 8) };
        if (role == RAD_OPD_IN || role == RAD_OPD_INOUT) rd[n_rd++] = r;
        if (role == RAD_OPD_OUT || role == RAD_OPD_INOUT) wr[n_wr++] = r;
    }
    if (!o.opaque && ip.args.scratch_bytes > 0 && ip.args.scratch) {
        const uintptr_t lo = (uintptr_t)ip.args.scratch;
        rd[n_rd++] = wr[n_wr++] = { lo, lo + (uintptr_t)ip.args.scratch_bytes };
    }
    o.rd = (uint32_t)haz_rng_.size();
    o.n_rd = (uint32_t)n_rd;
    for (int i = 0; i < n_rd; ++i) haz_rng_.push_back(rd[i]);
    o.wr = (uint32_t)haz_rng_.size();
    o.n_wr = (uint32_t)n_wr;
    for (int i = 0; i < n_wr; ++i) haz_rng_.push_back(wr[i]);
    haz_ops_.push_back(o);
}

/* THE HOST OP'S JOIN; see "A HOST LAUNCH IS A JOIN" in Ctx::issue. */
int Ctx::host_join(const RadOperand* opd, int n_opd) {
    return host_join_wait(host_join_drains(opd, n_opd));
}

/* Whether an operand is one the arena does not track, which the join can only cover by draining
 * the stream. */
bool Ctx::host_join_drains(const RadOperand* opd, int n_opd) const {
    for (int i = 0; i < n_opd; ++i) {
        const RadOperand& o = opd[i];
        if (o.kind == RAD_OPK_WEIGHT || o.kind == RAD_OPK_WTAB || o.kind == RAD_OPK_NONE) continue;
        if (o.kind == RAD_OPK_BUF && buf_host_[o.handle]) continue;
        return true;
    }
    return false;
}

int Ctx::host_join_wait(bool drain) {
    ++n_host_syncs_;
    if (drain) return rad_stream_sync(cur_stream());
    for (int l = 0; l < 2; ++l) {
        if (!host_touch_pending_[l]) continue;
        RAD_TRY(rad_event_sync(host_touch_ev_[l]));
        host_touch_pending_[l] = false;
    }
    return RAD_OK;
}

/* ================================================================== the cold paths */

int Ctx::abort_step(rad_op h, int band, int dom, const IssuePlan& ip, int64_t n, int status) {
    const OpInfo&   o     = *plans_[h].info;
    const char*     kname = ip.row && ip.row->info ? ip.row->info->name : "<unnamed>";
    const char*     pname = ip.row ? ip.row->plugin.c_str() : "<unknown plugin>";
    const Resolved& r     = o.bands[(size_t)band].dom[dom];
    const char*     key   = o.ranged_key.empty() ? "n" : o.ranged_key.c_str();

    /* std::string here, on a path that has already lost the step. The hot path above allocates
     * nothing; this one is allowed to, because the alternative is a truncated diagnostic on the
     * one occasion anybody needs a complete one. */
    step_fail_ = fmt("kernel %s (%s, %s, %s domain) refused op '%s' at %s=%lld "
                     "[band %s<=%lld] geometry {%s}: %s",
                     kname, pname, r.choice.empty() ? "no tunables" : r.choice.c_str(),
                     dom == RAD_DOMAIN_HOST ? "host" : "device",
                     o.op.c_str(), key, (long long)n, key,
                     (long long)o.bands[(size_t)band].hi, r.geom.str().c_str(),
                     rad_strerror(status));

    RAD_ERR("%s", step_fail_.c_str());
    RAD_ERR("a kernel returning negative at issue is a bug in SELECTION, not a runtime condition. "
            "The step is aborted and the requests in it fail. There is no fallback here, silent "
            "or otherwise -- a silent fallback is how you ship a slow path nobody knows about "
            "(spec §17).");

    step_status_ = status;
    return status;
}

int Ctx::step_fail(const char* what) {
    step_fail_ = what ? what : "the architecture block refused this batch";
    RAD_ERR("%s", step_fail_.c_str());
    step_status_ = RAD_E_SHAPE;
    return RAD_E_SHAPE;
}

int Ctx::abort_step_msg(rad_op h, int status, std::string what) {
    const OpInfo* o = (h != RAD_NULL_HANDLE && (size_t)h < plans_.size()) ? plans_[h].info : nullptr;
    step_fail_ = o ? fmt("op '%s' {%s}: %s", o->op.c_str(), o->base.str().c_str(), what.c_str())
                   : fmt("op handle %u: %s", h, what.c_str());
    RAD_ERR("%s", step_fail_.c_str());
    step_status_ = status;
    return status;
}

void Ctx::fatal_issue(rad_op h, int64_t n, const char* why) const {
    if (h == RAD_NULL_HANDLE || (size_t)h >= plans_.size() || !plans_[h].info)
        fatal("rad_issue: op handle %u at n=%lld %s", (unsigned)h, (long long)n, why);
    const OpInfo& o = *plans_[h].info;
    fatal("rad_issue: op '%s' {%s} at %s=%lld %s",
          o.op.c_str(), o.base.str().c_str(),
          o.ranged_key.empty() ? "n" : o.ranged_key.c_str(), (long long)n, why);
}

/* ================================================================== pointer queries */

void* Ctx::weight_ptr(rad_weight w) {
    if (w == RAD_NULL_HANDLE || (size_t)w >= w_tmpl_.size()) {
        RAD_WARN("rad_weight_ptr: handle %u was never declared", (unsigned)w);
        return nullptr;
    }
    w_direct_[w] = 1;
    return resolve_weight(w);
}

void* Ctx::buf_ptr(rad_buf b) {
    if (b == RAD_NULL_HANDLE || (size_t)b >= buf_tmpl_.size()) {
        RAD_WARN("rad_buf_ptr: handle %u was never declared", (unsigned)b);
        return nullptr;
    }
    return buf_tmpl_[b].data;
}

}  /* namespace rad */

/* ================================================================== the C surface */
/* rad_runtime.h is C, and this is the whole of its implementation. Ctx derives from RadCtx, so
 * every one of these is a static_cast and a forwarded call -- the compiler sees straight through
 * it, which is the point of the empty base rather than a reinterpret_cast. */
extern "C" {

int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    if (!c) rad::fatal("rad_issue: null RadCtx");
    return static_cast<rad::Ctx*>(c)->issue(op, opd, n_opd, n);
}

int rad_step_fail(RadCtx* c, const char* what) {
    if (!c) rad::fatal("rad_step_fail: null RadCtx");
    return static_cast<rad::Ctx*>(c)->step_fail(what ? what : "the architecture block refused this batch");
}

const RadBatch* rad_batch(RadCtx* c)      { return c ? static_cast<rad::Ctx*>(c)->batch() : nullptr; }
int             rad_rank(RadCtx* c)       { return c ? static_cast<rad::Ctx*>(c)->rank() : 0; }
int             rad_world_size(RadCtx* c) { return c ? static_cast<rad::Ctx*>(c)->world_size() : 1; }
RadStream       rad_stream(RadCtx* c)     { return c ? static_cast<rad::Ctx*>(c)->stream() : nullptr; }

/* The second lane. `rad_lane` says which one subsequent issues launch on; `rad_lane_join` orders
 * one behind the other with an event and blocks neither the host nor either stream. The runtime
 * joins lane 1 back into lane 0 at the end of every step whatever the plugin did, so forgetting
 * the join is a performance bug and not a correctness one. Read Ctx::lane1_ before using this:
 * every collective must ride ONE lane, a second lane may not carry a split-K GEMM, and the buffer
 * planner packs from a single linear order. */
int rad_lane(RadCtx* c, int lane) {
    if (!c) return RAD_E_INVAL;
    return static_cast<rad::Ctx*>(c)->set_lane(lane);
}
int rad_lane_join(RadCtx* c, int from, int to) {
    if (!c) return RAD_E_INVAL;
    return static_cast<rad::Ctx*>(c)->lane_join(from, to);
}

int rad_route_report(RadCtx* c, int layer, const RadRouting* r) {
    if (!c) return RAD_E_INVAL;
    return static_cast<rad::Ctx*>(c)->route_report(layer, r);
}

int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert) {
    return c ? static_cast<rad::Ctx*>(c)->route_counts(layer, n_expert) : nullptr;
}

void* rad_weight_ptr(RadCtx* c, rad_weight w) {
    return c ? static_cast<rad::Ctx*>(c)->weight_ptr(w) : nullptr;
}

void* rad_buf_ptr(RadCtx* c, rad_buf b) {
    return c ? static_cast<rad::Ctx*>(c)->buf_ptr(b) : nullptr;
}

}  /* extern "C" */
