/* issue.cpp -- the hot path, and the C surface the architecture plugin calls.
 *
 * Per rad_issue: index the op's bucket table with `n`, pick the domain from the op's current
 * execution site, build a RadTensor per operand from its declare-time template, resolve weight
 * handles (a stream wait on the mover's event if in flight), set the operand count in the
 * prefilled RadArgs, and launch.
 *
 * No allocation, no host synchronisation, no device read-back: writes go only to arrays sized at
 * prepare, and the only device call besides the launch is rad_event_wait, which does not block
 * the host. See ctx.h for the cost budget (spec §3.2).
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

/* Column slice: `cols` narrows the last dimension and leaves every stride alone, so the declared
 * width stays the row pitch. This reads a fused projection without a copy (q and the attention
 * gate interleaved per head; GDN's `a` and `b` in one in-projection). The element offset picks the
 * first column, `cols` the width.
 *
 * Only the outermost and innermost dimensions can be shortened without rewriting strides, so the
 * ABI offers only those two. */
static inline void narrow_cols(const RadOperand& o, RadTensor* t) {
    if (o.cols > 0 && t->rank >= 1) t->shape[t->rank - 1] = o.cols;
}

/* Whether the operand fits the buffer it names. The span is the largest reachable element index,
 * `offset` + sum((extent - 1) * stride). Comparing shape[0] instead would be wrong: a [T, 2560]
 * buffer read at 160 columns is legitimately T*16 rows of 160 (the PLE gather does this).
 *
 * An empty extent reaches nothing; the kernel's own bounds decide. */
static inline bool operand_fits(const RadTensor& t, int64_t offset, const RadTensor& decl) {
    if (offset < 0) return false;
    int64_t span = offset;
    for (uint32_t d = 0; d < t.rank; ++d) {
        if (t.shape[d] <= 0) return true;
        span += (t.shape[d] - 1) * t.stride[d];
    }
    return span < rad_tensor_numel(&decl);
}

std::string tensor_shape_str(const RadTensor& t) {
    std::string s = "[";
    for (uint32_t d = 0; d < t.rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t.shape[d]);
    s += "]/[";
    for (uint32_t d = 0; d < t.rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t.stride[d]);
    return s + "]";
}

/* Element offset to byte offset. False when it cannot be expressed: a mid-byte offset into a
 * packed sub-byte weight (e.g. an odd W4 element), or a plugin-private dtype with no core-known
 * size. */
static inline bool byte_delta(int bits, int64_t elems, int64_t* out) {
    if (bits >= 8) { *out = elems * (int64_t)(bits >> 3); return true; }
    if (bits <= 0) return false;                      /* plugin-private: the core has no size */
    const int64_t nbits = (int64_t)bits * elems;
    if (nbits & 7) return false;                      /* mid-byte offset into a packed weight */
    *out = nbits >> 3;
    return true;
}

void* Ctx::resolve_weight(rad_weight w) {
    /* The template carries WeightInfo::ptr, refreshed between steps after a move. In the
     * all-in-VRAM case that is the answer. */
    void* p = w_tmpl_[w].data;
    if (!residency_) return p;

    const WeightSlot* s = residency_->lookup(w);       /* never blocks -- residency.h */
    if (s) {
        p = s->ptr;
        /* Wait once per generation and lane, not per issue; a stream wait orders everything
         * issued after it on that stream, and nothing on the other lane. Sound because the mover
         * bumps `generation` per transfer (residency.h). */
        std::vector<uint32_t>& waited = waited_gen_[lane_];
        if (s->ready && waited[w] != s->generation) {
            /* Kept out of a recording: it orders every later play too, and moving a weight a
             * pass names directly re-keys the pass. */
            if (rec_) tape_pause(true);
            rad_event_wait(cur_stream(), s->ready);         /* the stream waits, not the host */
            if (rec_) tape_pause(false);
            waited[w] = s->generation;
            ++n_stream_waits_;
        }
    }
    return p;
}

/* ================================================================== weight tables
 *
 * RAD_OPK_WTAB (Ctx::WeightTable, RAD_WTAB in rad_runtime.h): a run of declared weights becomes
 * the device pointer array a grouped GEMM walks.
 *
 * Entries must agree on shape and dtype: the kernel strides every entry with one set of trailing
 * extents, so a mixed run reads out of bounds. A violation is a plugin bug, named at first issue. */
/* A table's pinned and device arrays, allocated once. Normally called from prepare
 * (bind_weight_tables), so no allocation happens on the step path (spec §3.1) and the bytes are
 * not taken unbudgeted from --gpu-headroom-mib. resolve_weight_table calls it on a miss, which is
 * legal for a table the declaration did not describe. Returns the slot in wtabs_. */
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

/* Binds every RAD_OPD_WTAB operand of every declared op before the first step. A run is the
 * smallest declared handle at that operand position plus the count; rad_decl_weight hands out
 * consecutive indices, so a gap is a plugin bug and is refused (it would read the wrong expert). */
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
                        "handles run [%u, %u]; a table must be a contiguous run, because the "
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
        RAD_DEBUG("weight tables: %lld bound at prepare, %s of pointer arrays",
                  (long long)tables, humanb(bytes).c_str());
    return RAD_OK;
}

static const bool g_dbg_issue = std::getenv("RADIANCE_DEBUG_ISSUE") != nullptr;
/* RADIANCE_DEBUG_ISSUE: entries walked and wall time spent in weight-table refreshes, reported
 * separately from the rest of `own`. */
static double g_wtab_us = 0.0, g_wtab_us_p = 0.0;
static long   g_wtab_e  = 0,   g_wtab_e_p  = 0;

int Ctx::resolve_weight_table(rad_op h, int operand, rad_weight first, int64_t n,
                              RadTensor* out) {
    const uint64_t key = ((uint64_t)h << 32) | (uint32_t)operand;
    size_t slot;
    auto it = wtab_of_.find(key);
    if (it == wtab_of_.end()) {
        /* A plugin issued a table at an operand position its declaration did not name. Legal, but
         * it allocates on the step path. */
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
    /* While recording, the refresh is host work: cut the tape and name the table, so a play
     * refreshes it on this lane with whatever the mover has changed by then. */
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
    const bool wt_time = g_dbg_issue && rank() == 0;   /* one thread accumulates */
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
        /* First issue: check agreement, resolve every entry, upload the whole array. `host` is
         * never written again, so the async copy is safe however late it runs. */
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
        /* Entries that may have moved: resolve each (which also orders the stream behind an
         * in-flight transfer) and patch only those whose pointer differs. */
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

/* Queues the residency table's changes against every table holding each weight. A weight also
 * resolved directly into an operand bumps `dense_epoch_`. */
void Ctx::sync_residency() {
    if (!residency_) return;
    const std::vector<rad_weight>& ch = residency_->changes();
    if (ch.empty()) return;
    for (rad_weight w : ch) {
        if ((size_t)w < w_direct_.size() && w_direct_[w]) ++dense_epoch_;
        if ((size_t)w >= wmemb_head_.size()) continue;
        for (int32_t m = wmemb_head_[w]; m >= 0; m = wmemb_[(size_t)m].next) {
            WeightTable& t = wtabs_[(size_t)wmemb_[(size_t)m].slot];
            /* Unissued or already-full tables resolve every entry at their next issue. */
            if (!t.checked || t.full) continue;
            if ((int64_t)t.pend.size() >= t.n) { t.full = true; t.pend.clear(); continue; }
            t.pend.push_back(wmemb_[(size_t)m].entry);
        }
    }
    residency_->clear_changes();
}

/* The operand a kernel sees for a table: entry 0's dtype and extents with the expert axis
 * prepended at stride 0, which marks `data` as a pointer array. */
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

/* RADIANCE_DEBUG_ISSUE: host cost of the issue path, split from the launch it wraps. Kernel
 * traces and the per-op profiler only see device time.
 *
 * `own` is everything before the launch (band scan, domain lookup, operand resolution); `launch`
 * is the plugin shim plus HIP. Costs two clock reads (~50 ns) per issue when enabled.
 *
 * Reported per window, not as a running mean, so load and the first prefill do not dominate, and
 * per op, so host launch time is charged to the op that spent it. */
struct IssOp { double lau = 0.0, own = 0.0; long n = 0; };
static std::unordered_map<uint32_t, IssOp> g_iss_by_op;
static double g_iss_own = 0.0, g_iss_lau = 0.0;
static double g_iss_own_p = 0.0, g_iss_lau_p = 0.0;
static long   g_iss_n   = 0;

/* Wraps issue_body with the stager's calls on a pass it armed for. before_op runs before the
 * op's tables are resolved, followed by sync_residency if the stager changed the residency table.
 * after_op runs on the op's stream, where "every reader of this weight is behind us" can be
 * recorded. */
int Ctx::issue(rad_op h, const RadOperand* opd, int n_opd, int64_t n) {
    if (!staging_) {
        const int rc = issue_body(h, opd, n_opd, n);
        if (rc < 0 || (size_t)h >= plans_.size() || plans_[h].lora_site < 0) return rc;
        return issue_lora(h, opd, n_opd, n);
    }
    if (step_status_ < 0) return step_status_;
    bool resync = false;
    const int sb = stager_->before_op(h, cur_stream(), &resync);
    if (sb < 0)
        return abort_step_msg(h, sb, "the weight stager could not stage this op's weights");
    if (resync) sync_residency();
    const int rc = issue_body(h, opd, n_opd, n);
    if (rc < 0) return rc;
    const int sa = stager_->after_op(h, cur_stream());
    if (sa < 0)
        return abort_step_msg(h, sa, "the weight stager could not release this op's weights");
    if ((size_t)h < plans_.size() && plans_[h].lora_site >= 0) return issue_lora(h, opd, n_opd, n);
    return rc;
}

/* Runtime LoRA: the two ops the core appended for a targeted gemm, issued right behind it on the
 * same stream, over the same rows, through the ordinary issue path (bands, tape, hazards).
 *
 *   t  = A[slot] . x          lora_shrink, x = the gemm's own activation operand(s)
 *   y += scale . B[slot] . t  lora_expand, y = the gemm's own output operand
 *
 * A row-parallel gemm's y is this rank's partial sum and the plugin reduces it after this returns,
 * so the delta rides the same reduction (B.A.x is linear in x's shards). Nothing is issued on a
 * step with no adapter row (RadBatch::lora_slot null), which is then the pass an engine without
 * adapters runs, nor on a draft pass, whose proposals the trunk verifies. An encoder pass never
 * reaches a site: such a weight is outside the targets' scope rule (core/lora/lora.h), so reaching
 * one here is refused rather than run without its delta.
 *
 * A generator's pass runs one adapter in all its rows (gen_lora_slot_, -1 for none), whatever the
 * rows are -- a stream's tokens, a text encoder's taps, a video's patches -- so its slot operand
 * is row `slot` of Program::lora_rows, the gemm's first n rows of it. */
int Ctx::issue_lora(rad_op h, const RadOperand* opd, int n_opd, int64_t n) {
    if (!batch_) return RAD_OK;
    const bool gen = batch_->gen != 0;
    if (batch_->enc != 0)
        return abort_step_msg(h, RAD_E_UNSUPPORTED,
                              "a LoRA-targeted gemm was issued by an encoder pass, which runtime "
                              "LoRA does not reach; leave its module out of --lora-target-modules");
    if (gen ? gen_lora_slot_ < 0 : (!batch_->lora_slot || batch_->draft_pass != 0)) return RAD_OK;
    const OpPlan& p = plans_[h];
    const LoraIssueSite& s = program_->lora_sites[(size_t)p.lora_site];
    if (n == RAD_N_BATCH) n = batch_->n_tok;
    if (n <= 0) return RAD_OK;
    /* The adapter ops run in the gemm's domain and read the pools, which are device buffers: on a
     * card that is the device domain, and on the host backend every domain is host memory. */
    const int dom = (int)op_domain_[h].load(std::memory_order_relaxed);
    if (dom != RAD_DOMAIN_DEVICE && !device_is_host())
        return abort_step_msg(h, RAD_E_UNSUPPORTED,
                              "a LoRA-targeted gemm was placed on the host, and the adapter pools "
                              "are on the card");
    if (s.opd_a < 0 || s.opd_a >= n_opd || s.opd_y < 0 || s.opd_y >= n_opd)
        return abort_step_msg(h, RAD_E_STATE, "LoRA site: the gemm was issued with fewer operands "
                                              "than its schema names");
    const RadOperand& ya = opd[s.opd_y];
    const RadOperand& xa = opd[s.opd_a];
    if (ya.kind != RAD_OPK_BUF || xa.kind != RAD_OPK_BUF ||
        (size_t)ya.handle >= program_->buffers.size() || (size_t)xa.handle >= program_->buffers.size())
        return abort_step_msg(h, RAD_E_UNSUPPORTED, "LoRA site: the gemm's activation or output is "
                                                    "not a declared buffer");
    /* The rows the gemm wrote are the step's token rows from r0 on, so the slot array is read from
     * r0 too. A row slice offsets by whole rows of the buffer's declared width and a column slice
     * by less than one row; any other view cannot be placed. */
    const RadBufDecl& yd = program_->buffers[ya.handle].decl;
    const RadBufDecl& xd = program_->buffers[xa.handle].decl;
    if (yd.dtype != s.y_dtype || xd.dtype != s.a_dtype)
        return abort_step_msg(h, RAD_E_STATE,
                              fmt("LoRA site: declared for activation %s / output %s, issued with "
                                  "%s / %s", rad_dtype_name(s.a_dtype), rad_dtype_name(s.y_dtype),
                                  rad_dtype_name(xd.dtype), rad_dtype_name(yd.dtype)));
    const int64_t yw = yd.rank >= 2 ? yd.shape[yd.rank - 1] : 0;
    if (yw <= 0) return abort_step_msg(h, RAD_E_STATE, "LoRA site: the gemm's output has no width");
    /* A token step reads each row's slot at the row the gemm wrote, so the views must name token
     * rows; a generator pass's one slot holds for any rows. */
    if (!gen && ya.cols == 0 && ya.offset % yw != 0)
        return abort_step_msg(h, RAD_E_STATE, "LoRA site: the gemm's output view starts inside a "
                                              "row, so its token rows cannot be told");
    const int64_t r0 = gen ? 0 : ya.offset / yw;
    /* The activation must start at the same row: shrink reads x's row m and expand writes y's row
     * m with one slot. */
    if (!gen) {
        const int64_t xw = xd.rank >= 2 ? xd.shape[xd.rank - 1] : 0;
        if (xw <= 0 || (xa.cols == 0 && xa.offset % xw != 0) || xa.offset / xw != r0)
            return abort_step_msg(h, RAD_E_STATE, "LoRA site: the gemm's activation and output "
                                                  "views do not start at the same token row");
    }
    if ((int)op_domain_[s.shrink].load(std::memory_order_relaxed) != dom ||
        (int)op_domain_[s.expand].load(std::memory_order_relaxed) != dom)
        return abort_step_msg(h, RAD_E_UNSUPPORTED, "LoRA site: the adapter ops were placed in "
                                                    "another domain than the gemm they follow");
    if (!gen && (r0 < 0 || r0 + n > batch_->n_tok))
        return abort_step_msg(h, RAD_E_STATE,
                              fmt("LoRA site: the gemm wrote rows [%lld, %lld) of a %lld-token step",
                                  (long long)r0, (long long)(r0 + n), (long long)batch_->n_tok));
    if (gen && (!program_->lora_rows || gen_lora_slot_ >= program_->lora_pools[(size_t)s.pool].S))
        return abort_step_msg(h, RAD_E_STATE,
                              fmt("LoRA site: a generator pass in slot %d of %lld", gen_lora_slot_,
                                  (long long)program_->lora_pools[(size_t)s.pool].S));
    /* Each row's adapter is the step batch's (RadBatch::lora_slot), rewritten in stream order
     * before each step's kernels, so the pointer this launch records is the one a replay of it
     * reads. A generator pass is never recorded, so its run of the slot table is named directly,
     * and it runs the pair over chunks of at most lora_rows_n rows: a video's passes run to
     * hundreds of thousands of rows, and `t` and the slot run are sized for one chunk. */
    const LoraPool&    pool = program_->lora_pools[(size_t)s.pool];
    const rad_buf      t    = program_->lora_t[(size_t)s.t].lane[lane_ ? 1 : 0];
    /* A generator's slot holds its ranks from 0 (Program::lora_rused): no pair where it holds
     * none, the narrowest declared pair that holds them otherwise. */
    uint32_t sh_op = s.shrink, ex_op = s.expand;
    int64_t  rv = s.R;
    if (gen) {
        const size_t ui = (size_t)s.pool * (size_t)pool.S + (size_t)gen_lora_slot_;
        const int64_t used = ui < program_->lora_rused.size() ? program_->lora_rused[ui] : s.R;
        if (used <= 0) return RAD_OK;
        for (int v = 0; v < s.n_v; ++v)
            if (s.R_v[v] >= used) { sh_op = s.shrink_v[v]; ex_op = s.expand_v[v]; rv = s.R_v[v]; }
    }
    RadOperand         xs   = s.opd_a_scale >= 0 && s.opd_a_scale < n_opd ? opd[s.opd_a_scale]
                                                                          : RAD_NONE;
    const int64_t      step = gen ? program_->lora_rows_n : n;
    if (step <= 0) return abort_step_msg(h, RAD_E_STATE, "LoRA site: no slot rows");
    int64_t sw = 0;   /* a_scale's row stride, for a chunk past the first */
    if (n > step && xs.kind != RAD_OPK_NONE) {
        if (xs.kind != RAD_OPK_BUF || (size_t)xs.handle >= program_->buffers.size())
            return abort_step_msg(h, RAD_E_UNSUPPORTED, "LoRA site: a generator pass's activation "
                                                        "scale is not a declared buffer, so its "
                                                        "rows cannot be chunked");
        const RadBufDecl& sd = program_->buffers[xs.handle].decl;
        sw = sd.rank >= 2 ? sd.shape[sd.rank - 1] : 1;
    }
    const int64_t xw = xd.rank >= 2 ? xd.shape[xd.rank - 1] : 0;
    for (int64_t c = 0; c < n; c += step) {
        const int64_t nc = std::min(step, n - c);
        RadOperand x = xa, y = ya, a_s = xs;
        if (c > 0) {
            x.offset += c * xw;
            y.offset += c * yw;
            if (a_s.kind == RAD_OPK_BUF) a_s.offset += c * sw;
        }
        if (n > step) {
            x.rows = nc;
            y.rows = nc;
            if (a_s.kind == RAD_OPK_BUF) a_s.rows = nc;
        }
        RadOperand slot = gen ? RAD_B(program_->lora_rows) : RAD_P_T(batch_->lora_slot + r0, RAD_I32);
        if (gen) slot.offset = (int64_t)gen_lora_slot_ * program_->lora_rows_n;
        slot.rows = nc;
        RadOperand tv = RAD_B_N(t, 0, nc);
        if (rv < s.R) tv.cols = rv;
        const RadOperand sh[5] = { x, a_s, slot, RAD_B(pool.a), tv };
        int rc = issue_body((rad_op)sh_op, sh, 5, nc);
        if (rc < 0) return rc;
        const RadOperand ex[5] = { tv, slot, RAD_B(pool.b), RAD_B(pool.scale), y };
        rc = issue_body((rad_op)ex_op, ex, 5, nc);
        if (rc < 0) return rc;
    }
    return RAD_OK;
}

int Ctx::issue_body(rad_op h, const RadOperand* opd, int n_opd, int64_t n) {
    const auto t_iss0 = g_dbg_issue ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    /* Step already aborted. rad_arch_step is void and keeps issuing; skip the rest so the first
     * error stays the reported one. */
    if (step_status_ < 0) return step_status_;

    if (h == RAD_NULL_HANDLE || (size_t)h >= plans_.size())
        fatal_issue(h, n, "the plugin issued a handle declare never returned. An op that did not "
                          "resolve is RAD_NULL_HANDLE and becomes an error exactly here");

    OpPlan& p = plans_[h];

    /* RAD_N_BATCH: the ranged parameter is the step's token count. */
    if (n == RAD_N_BATCH) n = batch_ ? batch_->n_tok : 0;

    /* The bucket table: two to four bands sorted by inclusive upper bound, from the candidate
     * kernels' LE/GE/EQ constraints on the ranged parameter (spec §2.2). A linear scan is the
     * fastest option at this size. */
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
    /* The row's DIV on the ranged parameter held at the band's upper bound, maybe not here. */
    if (ip.ranged_div > 1 && n % ip.ranged_div != 0)
        return abort_step_msg(h, RAD_E_SHAPE,
                              fmt("issued at %lld, which kernel %s does not serve: it requires a "
                                  "multiple of %lld and was selected at the band's upper bound",
                                  (long long)n, ip.row->info->name, (long long)ip.ranged_div));

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
            /* Template: declared dtype, rank, shape and packed strides; data is already
             * arena_base + BufferInfo::arena_offset. */
            tt = buf_tmpl_[bh];
            touches_host |= buf_host_[bh] != 0;
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
            /* rows > 0 narrows dim 0, so a prefill chunk issues against a buffer sized at
             * max_tok. Strides stay valid since they are products of the trailing dimensions. */
            if (o.rows > 0) tt.shape[0] = o.rows;
            narrow_cols(o, &tt);
            /* An operand may name fewer rows than the declaration, never more. A write past the
             * end lands on whatever buffer the liveness packing placed there, which changes with
             * the graph, so the corruption is silent and config-dependent (e.g. a MoE expert plane
             * declared at max_tok rows but written at rows*top_k). */
            if (!operand_fits(tt, o.offset, buf_tmpl_[bh]))
                return abort_step_msg(h, RAD_E_SHAPE,
                                      fmt("operand %d reaches past buffer '%s': %s at offset "
                                          "%lld, declared %s", i,
                                          program_->buffers[bh].name.c_str(),
                                          tensor_shape_str(tt).c_str(), (long long)o.offset,
                                          tensor_shape_str(buf_tmpl_[bh]).c_str()));
            /* The op must lie inside the buffer's planned lifetime. The plan is computed from the
             * opt-in rad_op_reads/rad_op_writes; a buffer nothing declared lives for the whole
             * program. An op touching a declared buffer outside [first_def, last_use] may see
             * bytes the arena reused for another buffer, typically because the op gained a read
             * its declaration never mentioned. Refused, not warned: the failure is silent and
             * depends on the plan's packing. */
            {
                const BufferInfo& bi = program_->buffers[bh];
                /* A core-appended op (runtime LoRA) is judged at the op it rides on: it touches
                 * that op's activation and output inside that op's issue, never after it. */
                const int32_t at = p.info && p.info->anchor >= 0 ? p.info->anchor : (int32_t)h;
                if (bi.first_def >= 0 && (at < bi.first_def || at > bi.last_use))
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
            /* Same fit check as a buffer: an overrunning slice reads the next weight in the slab
             * and yields plausible numbers. Only a sliced operand can overrun, so whole-weight
             * issues skip it. */
            if ((o.offset || o.rows > 0 || o.cols > 0) &&
                !operand_fits(tt, o.offset, w_tmpl_[wh]))
                return abort_step_msg(h, RAD_E_SHAPE,
                                      fmt("operand %d reaches past weight '%s': %s at offset "
                                          "%lld, declared %s", i,
                                          program_->weights[wh].name.c_str(),
                                          tensor_shape_str(tt).c_str(), (long long)o.offset,
                                          tensor_shape_str(w_tmpl_[wh]).c_str()));
            break;
        }

        case RAD_OPK_WTAB: {
            /* A run of declared weights, passed as a device array of pointers. `handle` is the
             * first, `rows` the count. */
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
            /* A pointer the core did not allocate, usually a RadBatch field. No BufferInfo, so
             * dtype and shape come from the op's schema; the core carries the pointer and the
             * optional row count. */
            if (o.offset && o.dtype == RAD_DT_INVALID)
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d is a raw pointer with an element offset of "
                                          "%lld and no dtype. Give the operand a dtype with "
                                          "RAD_P_T() or do the arithmetic on the caller's side; "
                                          "the core cannot size an element it was never told about",
                                          i, (long long)o.offset));
            std::memset(&tt, 0, sizeof tt);
            tt.data  = o.raw;
            tt.dtype = o.dtype;   /* RAD_DT_INVALID unless given; libref reads positions,
                                   * cu_seqlens and num_accepted through the tensor's dtype. */
            /* On a raw operand `cols` is the second extent, not a column slice (RAD_P_T2), e.g.
             * the [n_seq, max_blocks] block table. */
            if (o.rows > 0 && o.cols > 0) {
                tt.rank = 2;
                tt.shape[0] = o.rows;   tt.shape[1] = o.cols;
                tt.stride[0] = o.cols;  tt.stride[1] = 1;
            } else if (o.rows > 0) {
                tt.rank = 1; tt.shape[0] = o.rows; tt.stride[0] = 1;
            }
            break;

        case RAD_OPK_KV: {
            /* A KV group's cache for one layer. `handle` is the rad_kvgroup, `offset` the absolute
             * layer index, rebased through the binding because a group holds one cache per bound
             * layer, not per model layer. */
            const uint32_t g = o.handle;
            if ((size_t)g >= kv_.size() || !kv_[g].base)
                return abort_step_msg(h, RAD_E_STATE,
                                      fmt("operand %d names KV group %u, which has no pool bound on "
                                          "this rank. The block manager allocates it at load and "
                                          "binds it into the Ctx; a group with none is a startup "
                                          "bug -- or a group this rank declared with "
                                          "rad_kv_group_zero and then issued against, which the "
                                          "plugin must not do", i, g));
            const KVPoolBinding& b = kv_[g];
            /* A lookup, not a subtraction: a group's layers need not be contiguous
             * (KVPoolBinding::layer_slot). */
            const int64_t layer = (o.offset >= 0 && (size_t)o.offset < b.layer_slot.size())
                                      ? b.layer_slot[(size_t)o.offset]
                                      : -1;
            const int64_t off   = layer * b.layer_stride;
            if (layer < 0 || (b.bytes && off + b.layer_stride > b.bytes))
                return abort_step_msg(h, RAD_E_INVAL,
                                      fmt("operand %d asks KV group %u for layer %lld, which is "
                                          "outside the %lld bytes it was allocated", i, g,
                                          (long long)o.offset, (long long)b.bytes));
            std::memset(&tt, 0, sizeof tt);
            tt.data  = static_cast<char*>(b.base) + off;
            tt.dtype = b.dtype;
            /* The operand carries the layer's shape, composed at load (see KVPoolBinding in
             * ctx.h); libr4d shims refuse a flat span. */
            if (b.rank) {
                tt.rank = b.rank;
                for (uint32_t d = 0; d < b.rank; ++d) tt.shape[d] = b.shape[d];
                rad_tensor_pack(&tt);
            } else {
                tt.rank      = 1;
                tt.shape[0]  = b.layer_stride
                                   ? b.layer_stride * 8 /
                                         (rad_dtype_bits(b.dtype) ? rad_dtype_bits(b.dtype) : 8)
                                   : 0;
                tt.stride[0] = 1;
            }
            if (o.rows > 0) tt.shape[0] = o.rows;
            break;
        }

        default:
            /* RAD_OPK_NONE: an absent optional operand is a null RadTensor.data, per rad_abi.h. */
            std::memset(&tt, 0, sizeof tt);
            break;
        }
    }

    /* The rest of RadArgs was filled in at prepare; args.t already points at tbuf_. */
    ip.args.n_t   = n_opd;
    /* The step's phase, so a kernel measures per phase instead of inferring one (RadArgs::phase). */
    ip.args.phase = phase_;
    p.issued    = true;
    ++n_issues_;

    /* All-reduce link accounting: `n` is the collective's numel (its ranged parameter) and operand
     * 0's dtype is the wire element type. The op's `dtype` key ("w4a8") is not an element
     * width. */
    if (p.billed && n_opd > 0) {
        const int bits = rad_dtype_bits(t[0].dtype);
        ar_calls_.fetch_add(1, std::memory_order_relaxed);
        if (bits > 0) ar_bytes_.fetch_add(n * bits / 8, std::memory_order_relaxed);
    }

    /* The oracle snapshots every operand before the launch, so the host kernel sees the same
     * input bytes even for an in-place op. */
    const IssuePlan* horacle = oracle_ ? oracle_before(h, b, n_opd) : nullptr;

    /* ============================== a host launch is a join ==============================
     *
     * The host runs ahead of the device, but a host-site op reads and writes the host arena now,
     * on this thread. Without a join it reads stale data, e.g. a host PLE gather
     * (`embed_lookup_q`) reading row ids a device `ngram_ids` has not yet written: valid ids,
     * wrong rows, fluent wrong output.
     *
     * The join waits on the last device op per lane that touched the host arena (host_touch_ev_),
     * not the whole stream: every hazard is with such an op, and draining would idle the device
     * while the host works. An architecture can issue the producer early and keep the device busy
     * meanwhile. A host op naming an untracked operand (raw pointer, KV cache, device-domain
     * buffer) drains its stream instead (host_join).
     *
     * The device caches host-arena memory, so rad_dev_host_wrote after the launch makes the
     * host's writes visible to later kernels. */
    if (dom == RAD_DOMAIN_HOST) {
        /* Timed apart from the op: it is the cost of stopping the pipeline. */
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

    /* A scratch region belongs to one stream. Kernels on a lane get consecutive slices
     * (scratch_take); a recorded pass orders any pair whose slices meet (haz_mark). Scratch users
     * on two unjoined lanes would overwrite each other's partials (e.g. split-K router GEMM on
     * lane 0 and the shared arm's split-K GEMM on lane 1, every MoE layer).
     *
     * A program that declares its second lane gets a region per lane. Otherwise both lanes share
     * region 0 and the second user is ordered behind the first, giving up the overlap but not
     * correctness. That also covers lane 0 before lane 1 exists, since a lane built later has
     * nothing ordering it behind lane 0's earlier work. */
    if (ip.args.scratch_bytes > 0 && dom != RAD_DOMAIN_HOST && arena_.scratch_regions() > 1) {
        /* Per-lane regions: nothing to order. */
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

    /* A host-site op while recording: cut the tape and keep the op, to be joined and rerun on
     * these operands at every play. Its launch is not recorded, or anything it submitted would be
     * submitted twice. */
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

    /* In a staged pass a collective holds the card's link; the stager's copy lists yield meanwhile
     * (see rad_link_hold). Taken after every wait this op's operands put on the stream, so nothing
     * between hold and release can wait on a weight stuck behind a yielding list. */
    const bool hold = staging_ && p.collective && dom == RAD_DOMAIN_DEVICE;
    const RadArgs* la = &ip.args;
    RadArgs gate_args;
    if (hold) {
        const int hs = rad_link_hold(cur_stream(), 1);
        if (hs < 0) return abort_step(h, b, dom, ip, n, hs);
        if (ip.gate_p) {
            gate_args = ip.args;
            gate_args.p = ip.gate_p;
            gate_args.n_p = ip.gate_n_p;
            la = &gate_args;
        }
    }

    int s;
    if (profiling_ && dom == RAD_DOMAIN_HOST) {

        /* A host-site op is timed on the host wall clock; a device event pair around it would
         * measure nothing. It includes page faults and link reads, which can make a host op (e.g.
         * the PLE gather) the most expensive thing in the step. */
        const auto t0 = std::chrono::steady_clock::now();
        s = ip.launch(la, cur_stream());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        prof_ms_[(size_t)h] += ms;
        prof_calls_[(size_t)h] += 1;
        if ((size_t)h < prof_host_.size() && phase_ >= 0 && phase_ < 3) {
            const size_t k = (size_t)phase_ * prof_calls_.size() + (size_t)h;
            if (k < prof_ms_ph_.size()) {
                prof_ms_ph_[k] += ms;
                ++prof_calls_ph_[k];
            }
        }
        if ((size_t)h < prof_host_.size()) prof_host_[(size_t)h] = 1;
    } else if (profiling_) {
        /* Spec §16 per-op timing: events around the launch, read by a sync at the end of the
         * step (which is why profiled runs are not comparable). */
        if (prof_n_ < prof_ring_.size()) {
            ProfSlot& ps = prof_ring_[prof_n_++];
            ps.op = (int32_t)h;
            ps.lane = (uint8_t)lane_;
            ps.phase = (uint8_t)(phase_ >= 0 && phase_ < 3 ? phase_ : 2);
            rad_event_record(ps.a, cur_stream());
            s = ip.launch(la, cur_stream());
            rad_event_record(ps.b, cur_stream());
        } else {
            prof_overflowed_ = true;
            s = ip.launch(la, cur_stream());
        }
    /* Rank 0 only: these file-scope accumulators (including an unordered_map) are not
     * thread-safe, and every rank thread runs this. */
    } else if (g_dbg_issue && rank() == 0) {
        const auto ta = std::chrono::steady_clock::now();
        s = ip.launch(la, cur_stream());
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
            /* The table walk is part of `own`. */
            std::fprintf(stderr, "D issue:   of which weight tables: %.3f us an issue over "
                                 "%.1f entries (%.1f ns an entry)\n",
                         (g_wtab_us - g_wtab_us_p) / w,
                         (double)(g_wtab_e - g_wtab_e_p) / w,
                         (g_wtab_e - g_wtab_e_p) > 0
                             ? (g_wtab_us - g_wtab_us_p) * 1000.0 / (double)(g_wtab_e - g_wtab_e_p)
                             : 0.0);
            /* Top ops by total host launch time, with the per-issue mean beside it. */
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
        s = ip.launch(la, cur_stream());
    }
    if (hold) {
        /* Always released, by a barrier unless the launch ended the hold itself
         * (RAD_LINK_ENDED); otherwise a list waiting on the hold never ends. */
        const int hr = rad_link_hold(cur_stream(), la != &ip.args && s == RAD_LINK_ENDED ? 2 : 0);
        if (hr < 0 && s >= 0) s = hr;
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

/* See "a host launch is a join" in Ctx::issue_body. */
int Ctx::host_join(const RadOperand* opd, int n_opd) {
    return host_join_wait(host_join_drains(opd, n_opd));
}

/* Whether any operand is untracked by the host arena, so the join must drain the stream. */
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

    /* Allocating is fine here: the step is already lost, and the diagnostic must be complete. */
    step_fail_ = fmt("kernel %s (%s, %s, %s domain) refused op '%s' at %s=%lld "
                     "[band %s<=%lld] geometry {%s}: %s",
                     kname, pname, r.choice.empty() ? "no tunables" : r.choice.c_str(),
                     dom == RAD_DOMAIN_HOST ? "host" : "device",
                     o.op.c_str(), key, (long long)n, key,
                     (long long)o.bands[(size_t)band].hi, r.geom.str().c_str(),
                     rad_strerror(status));

    /* A kernel refusing at issue is a selection bug, not a runtime condition: the step fails
     * and there is no fallback (spec §17). */
    RAD_ERR("%s; the step is aborted (a kernel that refuses at issue was selected wrongly)",
            step_fail_.c_str());

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
/* rad_runtime.h's implementation. Ctx derives from RadCtx, so each entry is a static_cast and a
 * forwarded call. */
extern "C" {

int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    if (!c) rad::fatal("rad_issue: null RadCtx");
    return static_cast<rad::Ctx*>(c)->issue(op, opd, n_opd, n);
}

int rad_step_fail(RadCtx* c, const char* what) {
    if (!c) rad::fatal("rad_step_fail: null RadCtx");
    return static_cast<rad::Ctx*>(c)->step_fail(what);
}

const RadBatch* rad_batch(RadCtx* c)      { return c ? static_cast<rad::Ctx*>(c)->batch() : nullptr; }
int             rad_rank(RadCtx* c)       { return c ? static_cast<rad::Ctx*>(c)->rank() : 0; }
int             rad_world_size(RadCtx* c) { return c ? static_cast<rad::Ctx*>(c)->world_size() : 1; }
RadStream       rad_stream(RadCtx* c)     { return c ? static_cast<rad::Ctx*>(c)->stream() : nullptr; }

/* The second lane. `rad_lane` selects the lane subsequent issues launch on; `rad_lane_join`
 * orders one behind the other with an event, blocking neither the host nor a stream. The runtime
 * joins lane 1 back into lane 0 at the end of every step, so a missing join costs performance,
 * not correctness. See Ctx::lane1_ for the constraints before using it. */
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

int rad_expert_prefetch(RadCtx* c, const RadPrefetchNext* next, rad_buf h, int64_t r0, int64_t rows,
                        int64_t n_embd) {
    return c ? static_cast<rad::Ctx*>(c)->xp_predict(next, h, r0, rows, n_embd) : RAD_E_INVAL;
}
int rad_expert_prefetch_mark(RadCtx* c) {
    return c ? static_cast<rad::Ctx*>(c)->xp_mark() : RAD_E_INVAL;
}
int rad_expert_prefetch_step(RadCtx* c, int layer) {
    return c ? static_cast<rad::Ctx*>(c)->xp_step(layer) : RAD_E_INVAL;
}

void* rad_weight_ptr(RadCtx* c, rad_weight w) {
    return c ? static_cast<rad::Ctx*>(c)->weight_ptr(w) : nullptr;
}

void* rad_buf_ptr(RadCtx* c, rad_buf b) {
    return c ? static_cast<rad::Ctx*>(c)->buf_ptr(b) : nullptr;
}

}  /* extern "C" */
