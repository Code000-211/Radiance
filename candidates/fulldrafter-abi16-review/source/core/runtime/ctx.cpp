/* ctx.cpp -- everything the run phase does that is not the hot path: bringing the arena and the
 * plans up, driving a step, the routing handoff, and the opt-in per-op timing. The hot path itself
 * is issue.cpp, kept in its own translation unit so it stays readable at the size it has to be.
 */
#include <vector>
#include "ctx.h"
#include "../device/device.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>

namespace rad {

/* ================================================================== lifecycle */

Ctx::~Ctx() { shutdown(); }

int Ctx::init(const CtxDesc& d) {
    if (!d.program) return RAD_E_INVAL;
    program_   = d.program;
    rank_      = d.rank;
    world_size_= d.world_size < 1 ? 1 : d.world_size;
    device_    = d.device;
    record_passes_ = d.record_passes;
    residency_ = d.residency;
    kv_        = d.kv;
    for (auto& b : kv_) {
        if (!b.base) continue;
        int32_t max_slot = -1;
        for (int32_t s : b.layer_slot) {
            if (s > max_slot) max_slot = s;
        }
        if (max_slot >= 0) {
            b.layer_tensor.resize((size_t)(max_slot + 1));
            for (int32_t s = 0; s <= max_slot; ++s) {
                RadTensor tt{};
                std::memset(&tt, 0, sizeof tt);
                const int64_t off = (int64_t)s * b.layer_stride;
                tt.data = static_cast<char*>(b.base) + off;
                tt.dtype = b.dtype;
                if (b.rank) {
                    tt.rank = b.rank;
                    for (uint32_t d = 0; d < b.rank; ++d) tt.shape[d] = b.shape[d];
                    rad_tensor_pack(&tt);
                } else {
                    tt.rank = 1;
                    tt.shape[0] = b.layer_stride
                                      ? b.layer_stride * 8 /
                                            (rad_dtype_bits(b.dtype) ? rad_dtype_bits(b.dtype) : 8)
                                      : 0;
                    tt.stride[0] = 1;
                }
                b.layer_tensor[(size_t)s] = tt;
            }
        }
    }
    arch_step_ = d.arch_step;
    profiling_ = d.profile_ops;

    /* The rank owns one device. Set it here so allocation lands on the right card, and again from
     * the rank thread in bind_thread() because the backend's current device is thread-local. */
    RAD_TRY(rad_dev_set(device_));

    if (d.stream) {
        stream_     = d.stream;
        own_stream_ = false;
    } else {
        /* Ordinary priority. The mover runs on its own stream and the compute stream waits on its
         * events (spec §5.4); making compute high-priority would let it preempt the transfers it
         * is about to wait for, which is exactly backwards. */
        RAD_TRY(rad_stream_create(&stream_, 0));
        own_stream_ = true;
    }

    /* The second lane is NOT created here. See lane_init(). */

    int s = prepare();
    if (s < 0) { shutdown(); return s; }
    return RAD_OK;
}

void Ctx::shutdown() {
    oracle_close();
    /* The one host synchronisation in the whole component, and it is at teardown rather than on
     * the step path. Everything below frees memory the stream may still be reading or writing --
     * the routing staging is the destination of a copy nobody waited on by design (spec §5.5), and
     * the arena is every kernel's operands. Freeing either out from under an in-flight stream is a
     * use-after-free that reproduces once a week and never under a debugger. */
    if (stream_) rad_stream_sync(stream_);
    if (lane1_) rad_stream_sync(lane1_);

    for (PassTape& t : tapes_) tape_drop(t);
    tapes_.clear();

    for (auto& p : prof_ring_) {
        if (p.a) rad_event_destroy(p.a);
        if (p.b) rad_event_destroy(p.b);
    }
    prof_ring_.clear();

    for (RadEvent& e : route_done_)
        if (e) { rad_event_destroy(e); e = nullptr; }
    route_.clear();

    if (route_pool_) { rad_dev_free(route_pool_, RAD_MEM_HOST_PINNED); route_pool_ = nullptr; }
    if (route_dev_)  { rad_dev_free(route_dev_, RAD_MEM_DEVICE);        route_dev_  = nullptr; }

    /* The expert pointer tables. Freed after the stream syncs above for exactly the reason those
     * syncs exist: the device half is the source operand of a grouped GEMM and the host half is
     * the source of a copy on that stream. */
    free_weight_tables();

    arena_.release();
    host_arena_.release();

    for (int i = 0; i < 2; ++i) {
        if (lane_ev_[i]) { rad_event_destroy(lane_ev_[i]); lane_ev_[i] = nullptr; }
        if (host_touch_ev_[i]) { rad_event_destroy(host_touch_ev_[i]); host_touch_ev_[i] = nullptr; }
        host_touch_pending_[i] = false;
    }
    if (own_lane1_ && lane1_) rad_stream_destroy(lane1_);
    lane1_      = nullptr;
    own_lane1_  = false;
    lane_       = 0;
    scratch_open_[0] = scratch_open_[1] = false;

    if (own_stream_ && stream_) rad_stream_destroy(stream_);
    stream_     = nullptr;
    own_stream_ = false;
    program_    = nullptr;
}

int Ctx::bind_thread() { return rad_dev_set(device_); }

/* ================================================================== prepare */

int Ctx::prepare() {
    oracle_open();
    RAD_TRY(bind_arena());
    RAD_TRY(bind_buffers());
    RAD_TRY(bind_weights(/*refresh_only=*/false));
    RAD_TRY(bind_weight_tables());
    RAD_TRY(bind_ops());
    RAD_TRY(bind_routing());
    RAD_TRY(bind_profiling());
    /* Recorded passes need a backend that records, and are off for everything that has to see
     * each issue: the per-op profile, the oracle, and the argument digest. */
    tape_on_ = record_passes_ && tape_supported() && !profiling_ && !oracle_ &&
               std::getenv("RADIANCE_DEBUG_ARGSHA") == nullptr;
    return RAD_OK;
}

int Ctx::bind_arena() {
    Program& P = *program_;

    /* The buffer plan hands us one offset per buffer, and arena_plan_bytes() turns those into the
     * two numbers Arena::init wants. THE WALK LIVES THERE AND NOT HERE because the VRAM budget
     * resolver has to subtract this allocation from the card before any of it exists -- see
     * core/mem/vram_budget.cpp. What stays here is the part that can only be done at bind time:
     * refusing a plan that did not run, and filling in a size the plan left at zero.
     *
     * Where a program has RAD_DOMAIN_HOST buffers -- host-site execution, spec §5.1 -- those
     * offsets cannot index the same block, so there are two arenas. */
    int64_t host_hw = 0;
    for (size_t i = 1; i < P.buffers.size(); ++i) {
        BufferInfo& bi = P.buffers[i];
        if (bi.arena_offset < 0) {
            RAD_ERR("buffer '%s' has no arena offset: the buffer plan did not run",
                    bi.name.c_str());
            return RAD_E_STATE;
        }
        if (bi.bytes <= 0) {
            int64_t n = 1;
            for (uint32_t d = 0; d < bi.decl.rank; ++d) n *= bi.decl.shape[d];
            bi.bytes = rad_dtype_bytes(bi.decl.dtype, n);
            RAD_DEBUG("buffer '%s' had no size from the plan; %s from its declaration",
                      bi.name.c_str(), humanb(bi.bytes).c_str());
        }
        if (bi.decl.domain == RAD_DOMAIN_HOST)
            host_hw = std::max(host_hw, bi.arena_offset + bi.bytes);
    }

    int64_t dev_bytes = 0, total = 0;
    arena_plan_bytes(P, &dev_bytes, &total);

    RAD_TRY(arena_.init(dev_bytes, P.scratch_bytes, RAD_MEM_DEVICE, std::max(P.scratch_regions, 1)));
    /* SAY WHAT THIS COSTS, BECAUSE NOTHING ELSE DOES. The allocation is in no pool, and it is not
     * something --gpu-headroom-mib can cover: it scales with max_tok, so any round number an
     * operator picks either under-reserves or wastes VRAM. On a part whose experts do not fit,
     * waste is not free -- a GiB of headroom nothing uses is about 1.4 ms a decode step, because
     * that GiB is experts that would otherwise be resident. So it is computed and charged by name:
     * core/mem/vram_budget.cpp does the charging, this line is the receipt, and the two numbers
     * must agree. */
    RAD_INFO("arena: %s device + %d x %s scratch = %s, charged to the VRAM budget before it was "
             "split", humanb(dev_bytes).c_str(), arena_.scratch_regions(),
             humanb(P.scratch_bytes).c_str(), humanb(total).c_str());

    if (host_hw > 0) {
        /* RAD_MEM_HOST_MAPPED AND NOT RAD_MEM_HOST, AND THAT IS THE WHOLE OF HOW A HOST RUN
         * REACHES THE DEVICE GRAPH.
         *
         * A host-site op writes a host-domain buffer and something downstream reads it on the
         * device. NOTHING IN THIS RUNTIME COPIES BETWEEN THE TWO ARENAS -- there is no transfer op
         * and no implicit staging in rad_issue, and the planner's `link_crossings` is an objective
         * it minimises, not a mechanism it provides. With plain RAD_MEM_HOST the host arena is
         * "ordinary malloc'd pages the GPU has never been shown" (core/device/hip.cpp), so the
         * device side of that hand-off had no way to read it and host-site execution could not
         * complete a step. Mapped, the same allocation is addressable from a kernel over the link
         * and the crossing IS the read -- which is row three of spec 5.1 applied to an activation
         * rather than to a weight.
         *
         * The cost is that every device access to a host-domain buffer is a link read, so this is
         * for the narrow hand-off at a host run's edge and not for a hot activation. A buffer's
         * domain is declared by the ARCHITECTURE, so that judgement stays where it belongs. */
        RAD_TRY(host_arena_.init(host_hw, 0, RAD_MEM_HOST_MAPPED));
        /* ONE POINTER SERVES BOTH SIDES, AND THAT IS CHECKED RATHER THAN ASSUMED. bind_buffers
         * stores a single base per buffer, which is correct exactly while the mapped allocation's
         * host and device views share an address -- true under HIP's unified addressing and true
         * by construction on the host backend. A backend where they differ gets a refusal here
         * instead of handing one of the two sides a pointer into nothing. */
        void* hb = host_arena_.base();
        void* db = rad_dev_device_ptr(hb);
        if (!db || db != hb) {
            RAD_ERR("the host activation arena at %p has device view %p: this runtime binds one "
                    "pointer per buffer, so a mapped allocation whose two views differ cannot be "
                    "handed to both a host kernel and a device kernel", hb, db);
            return RAD_E_UNSUPPORTED;
        }
        RAD_DEBUG("arena: %s host, mapped (host-site activations)", humanb(host_hw).c_str());
    }
    return RAD_OK;
}

int Ctx::bind_buffers() {
    Program& P = *program_;
    std::vector<RadTensor> t0(P.buffers.size(), RadTensor{});
    buf_bits_.assign(P.buffers.size(), 0);
    buf_host_.assign(P.buffers.size(), 0);
    buf_dense_.assign(P.buffers.size(), 0);

    for (size_t i = 1; i < P.buffers.size(); ++i) {
        BufferInfo& bi = P.buffers[i];
        RadTensor&  t  = t0[i];

        t.dtype = bi.decl.dtype;
        t.rank  = bi.decl.rank;
        for (uint32_t d = 0; d < bi.decl.rank && d < RAD_MAX_RANK; ++d) t.shape[d] = bi.decl.shape[d];
        rad_tensor_pack(&t);

        Arena& a = (bi.decl.domain == RAD_DOMAIN_HOST) ? host_arena_ : arena_;
        if (!a.live()) {
            RAD_ERR("buffer '%s' wants the %s arena, which was not allocated",
                    bi.name.c_str(), bi.decl.domain == RAD_DOMAIN_HOST ? "host" : "device");
            return RAD_E_STATE;
        }
        if (bi.arena_offset + bi.bytes > a.bytes()) {
            RAD_ERR("buffer '%s' at offset %lld + %s runs past the %s arena (%s)",
                    bi.name.c_str(), (long long)bi.arena_offset, humanb(bi.bytes).c_str(),
                    bi.decl.domain == RAD_DOMAIN_HOST ? "host" : "device",
                    humanb(a.bytes()).c_str());
            return RAD_E_FULL;
        }

        /* rad_buf_ptr is arena_base + BufferInfo::arena_offset, and nothing else -- there is no
         * table walk and no allocator between a handle and its bytes. */
        t.data  = a.base() + bi.arena_offset;
        bi.ptr  = t.data;
        buf_bits_[i] = rad_dtype_bits(bi.decl.dtype);
        buf_host_[i] = bi.decl.domain == RAD_DOMAIN_HOST ? 1 : 0;
        buf_dense_[i] = (t.rank >= 1 && rad_tensor_is_contiguous(&t)) ? 1 : 0;
    }
    level_tmpl_.clear();
    level_tmpl_.push_back(std::move(t0));
    level_ = 0;
    buf_tmpl_ = level_tmpl_[0];
    return RAD_OK;
}

int Ctx::add_level(int64_t rows, const std::vector<int64_t>& offset,
                   const std::vector<RadBufDecl>& decl) {
    Program& P = *program_;
    if (level_tmpl_.empty() || offset.size() != P.buffers.size() || decl.size() != P.buffers.size())
        return RAD_E_INVAL;
    std::vector<RadTensor> t = level_tmpl_[0];
    for (size_t i = 1; i < P.buffers.size(); ++i) {
        const BufferInfo& bi = P.buffers[i];
        /* THE HOST ARENA AND THE FIXED REGION ARE NOT LEVELLED: their template is level 0's. */
        if (bi.decl.domain == RAD_DOMAIN_HOST || bi.decl.kind != RAD_BUF_TRANSIENT) {
            if (offset[i] != bi.arena_offset) return RAD_E_STATE;
            continue;
        }
        const int64_t bytes = rad_dtype_bytes(decl[i].dtype, [&] {
            int64_t n = 1;
            for (uint32_t d = 0; d < decl[i].rank; ++d) n *= decl[i].shape[d];
            return n;
        }());
        if (offset[i] < P.arena_fixed_bytes || offset[i] + bytes > arena_.bytes()) {
            RAD_ERR("arena level for %lld rows puts buffer '%s' at %lld + %s, outside the arena's "
                    "transient region", (long long)rows, bi.name.c_str(), (long long)offset[i],
                    humanb(bytes).c_str());
            return RAD_E_STATE;
        }
        RadTensor& r = t[i];
        for (uint32_t d = 0; d < decl[i].rank && d < RAD_MAX_RANK; ++d) r.shape[d] = decl[i].shape[d];
        rad_tensor_pack(&r);
        r.data = arena_.base() + offset[i];
    }
    level_tmpl_.push_back(std::move(t));
    buf_tmpl_ = level_tmpl_[(size_t)level_];
    return (int)level_tmpl_.size() - 1;
}

int Ctx::set_level(int level) {
    if (level < 0 || level >= (int)level_tmpl_.size()) return RAD_E_INVAL;
    if (level == level_) return RAD_OK;
    buf_tmpl_ = level_tmpl_[(size_t)level];
    level_ = level;
    return RAD_OK;
}

int Ctx::bind_weights(bool refresh_only) {
    Program& P = *program_;
    ++w_tmpl_epoch_;
    if (!refresh_only) {
        w_tmpl_.assign(P.weights.size(), RadTensor{});
        w_bits_.assign(P.weights.size(), 0);
        w_gen_.assign(P.weights.size(), 0);
        waited_gen_.assign(P.weights.size(), UINT32_MAX);
        w_direct_.assign(P.weights.size(), 0);
    }

    for (size_t i = 1; i < P.weights.size(); ++i) {
        WeightInfo& wi = P.weights[i];
        const uint32_t gen = wi.generation.load(std::memory_order_acquire);
        if (refresh_only && w_gen_[i] == gen) continue;

        RadTensor& t = w_tmpl_[i];
        /* The stored dtype is what the resolved kernel's layout hook asked for; the logical dtype
         * is what the plugin declared. The kernel reads the layout it chose, so the stored one
         * wins where the layout pass filled it in.
         *
         * THE SHAPE IS THE DECLARED ONE and the dtype is the stored one, which is a seam worth
         * naming rather than a symmetry. WeightInfo DOES carry stored_shape -- the layout pass
         * fills it beside stored_dtype -- and this deliberately does not use it, because a kernel
         * whose layout reshapes has to keep working out its own stride from the geometry either
         * way: a row-sharded weight's stored shape is the whole tensor's, not this rank's.
         *
         * So a layout that changes the WIDTH (libr4d's fp8 lm_head appends each row's scales to
         * its own row) is invisible here, and its kernel derives the stride from the same rule its
         * layout hook used. What keeps the two honest is the layout TAG the engine checks at load,
         * not a shape it could compare. The declared shape is already divided by world_size,
         * because the plugin declares dimensions per rank (spec §9). */
        t.dtype = wi.stored_dtype != RAD_DT_INVALID ? wi.stored_dtype : wi.decl.dtype;
        t.rank  = wi.decl.rank;
        for (uint32_t d = 0; d < wi.decl.rank && d < RAD_MAX_RANK; ++d) t.shape[d] = wi.decl.shape[d];
        rad_tensor_pack(&t);
        t.data  = wi.ptr;
        w_bits_[i] = rad_dtype_bits(t.dtype);
        w_gen_[i]  = gen;
    }
    /* A rebinding can move any template, and a table entry the residency table has no slot for
     * reads its template: every table resolves every entry at its next issue. */
    if (refresh_only)
        for (WeightTable& t : wtabs_) if (t.checked) { t.full = true; t.pend.clear(); }
    return RAD_OK;
}

int Ctx::bind_ops() {
    Program& P = *program_;

    plans_.assign(P.ops.size(), OpPlan{});
    first_issued_.clear();
    first_issued_.reserve(P.ops.size());
    op_domain_ = std::vector<std::atomic<uint8_t>>(P.ops.size());
    band_table_.clear();
    issue_table_.clear();

    int64_t n_bands_total = 0, n_plans = 0;
    for (auto& o : P.ops) { n_bands_total += (int64_t)o.bands.size(); }
    n_plans = n_bands_total * RAD_N_DOMAINS;
    band_table_.reserve((size_t)n_bands_total);
    issue_table_.reserve((size_t)n_plans);

    for (size_t i = 1; i < P.ops.size(); ++i) {
        OpInfo& oi = P.ops[i];
        OpPlan& p  = plans_[i];
        p.info     = &oi;
        p.band_off = (int32_t)band_table_.size();
        p.n_bands  = (int32_t)oi.bands.size();
        p.plan_off = (int32_t)issue_table_.size();
        p.n_opd_schema = oi.schema ? (int32_t)oi.schema->n_operands : -1;
        /* The one op whose issue costs interconnect bytes. Matched by name once, here, rather
         * than on the issue path -- see OpPlan::collective. */
        p.collective = (oi.op == "all_reduce");

        for (const Band& b : oi.bands) band_table_.push_back(b.hi);

        for (const Band& b : oi.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                IssuePlan ip{};
                const Resolved& r = b.dom[d];
                if (r && r.row && r.row->info && r.row->info->launch) {
                    ip.row     = r.row;
                    ip.launch  = r.row->info->launch;

                    /* Bound once. args.p views the frozen geometry inside Program::ops, which is
                     * why declare must not touch a Resolved after this point: the vector would
                     * reallocate and every one of these would dangle. */
                    ip.args.t             = tbuf_;
                    ip.args.n_t           = 0;
                    ip.args.p             = r.geom.params();
                    ip.args.n_p           = r.geom.n_params();
                    ip.args.scratch       = arena_.scratch();
                    ip.args.scratch_bytes = r.scratch_bytes;
                    ip.args.instance      = r.instance;
                    ip.args.rank          = rank_;
                    ip.args.world_size    = world_size_;

                    if (r.scratch_bytes > arena_.scratch_bytes()) {
                        RAD_ERR("op '%s' band %lld kernel %s wants %s of scratch but the plan "
                                "reserved %s -- the arena was sized before this kernel was "
                                "resolved",
                                oi.op.c_str(), (long long)b.hi, r.row->info->name,
                                humanb(r.scratch_bytes).c_str(),
                                humanb(arena_.scratch_bytes()).c_str());
                        return RAD_E_SCRATCH;
                    }
                }
                issue_table_.push_back(ip);
            }
        }

        /* Execution site. Site is a property of the weights (spec §5.1), so an op whose every
         * weight operand was placed on the host runs its host kernel and everything else runs on
         * the device. An op with no weights stays on the device; the planner overrides through
         * set_op_domain, which is where the "contiguous host runs" objective actually lands. */
        int dom = RAD_DOMAIN_DEVICE;
        if (!oi.weights.empty()) {
            bool all_host = true;
            for (rad_weight w : oi.weights) {
                const WeightInfo* wi = P.weight(w);
                if (!wi || wi->site != Site::Host) { all_host = false; break; }
            }
            /* ...and whether that host kernel can READ the weights, which is a separate question
             * from whether one exists. The planner asked for the host; the device is always a
             * correct answer, so this is a fallback and a warning rather than a refusal. */
            std::string why;
            if (all_host && check_domain_layout((rad_op)i, RAD_DOMAIN_HOST, &why) < 0) {
                RAD_WARN("op '%s' was placed on the host and runs on the device instead: %s",
                         oi.op.c_str(), why.c_str());
                all_host = false;
            }
            if (all_host) dom = RAD_DOMAIN_HOST;
        }
        op_domain_[i].store((uint8_t)dom, std::memory_order_relaxed);

        if (dom == RAD_DOMAIN_HOST) {
            int s = check_domain((rad_op)i, RAD_DOMAIN_HOST);
            if (s < 0) return s;
        }
    }
    return RAD_OK;
}

/* CAN THE KERNEL THAT WOULD SERVE THIS OP ON `domain` READ THE BYTES THE CONTAINER HOLDS?
 *
 * A weight is stored ONCE, in whatever arrangement the kernel that claimed it at declare asked
 * for (spec 4.2), and that claimant is always a DEVICE row: rad_builder.cpp exempts the host
 * domain from the layout claim deliberately, because libref declares no layout for anything and
 * every re-laying device kernel would otherwise collide with its own reference fallback and no
 * model would declare at all.
 *
 * So moving an op to a domain whose kernel reads its weights AS DECLARED hands that kernel a
 * plane it cannot read. libr4d stores an fp8a8 weight in WMMA fragment order -- the same N*K
 * bytes in a different arrangement -- so the byte count matches, every shape check passes, and
 * the op computes a plausible wrong answer for the life of the process. Two readings of one
 * permuted plane are uncorrelated tensors, which is a model that serves and lies.
 *
 * The load-time tag check in engine_bringup cannot see this: it compares the container against
 * the tag the claimant recorded, and both of those are the device row's. This asks the same
 * question of the kernel that will actually run.
 *
 * `why` gets the sentence; the caller decides whether it is a warning or a refusal, because the
 * two callers differ. The planner ASKED for the host and can be given the device instead, which
 * is always a correct answer; an explicit set_op_domain has no such fallback. */
int Ctx::check_domain_layout(rad_op h, int domain, std::string* why) const {
    if (h == RAD_NULL_HANDLE || h >= plans_.size()) return RAD_E_INVAL;
    if (domain < 0 || domain >= RAD_N_DOMAINS)      return RAD_E_INVAL;
    const OpPlan& p = plans_[h];
    if (!p.info || !program_) return RAD_OK;
    const OpInfo& oi = *p.info;

    for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j) {
        const rad_weight wh = oi.weights[j];
        if (!wh || (size_t)wh >= program_->weights.size()) continue;
        const WeightInfo& w = program_->weights[(size_t)wh];
        if (w.layout_tag.empty()) continue;      /* stored as it is: any reader will do */
        const int opd = oi.weight_opd[j];

        for (int b = 0; b < p.n_bands; ++b) {
            const IssuePlan& ip =
                issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS + (size_t)domain];
            const RadKernelInfo* ki = ip.row ? ip.row->info : nullptr;
            if (!ki) continue;                   /* no kernel here at all: check_domain's answer */
            RadLayout L{};
            RadTensor t[RAD_ENC_MAX_PLANES];
            weight_planes(w, t);
            /* A row with no hook, or one that declines this operand, is asserting that it reads
             * the selected plane AS IT IS -- the same reading rad_builder.cpp gives the same code.
             * One that does not read the encoding at all cannot take the op either. */
            const int st = ki->layout ? ki->layout(ip.args.p, ip.args.n_p, opd, &w.enc, w.sel, t,
                                                   w.n_sel, &L)
                                      : RAD_E_UNSUPPORTED;
            const std::string tag = st == RAD_OK ? (L.tag && *L.tag ? L.tag : ki->name)
                                                 : std::string();
            if (st >= 0 || st == RAD_E_UNSUPPORTED) {
                if (tag == w.layout_tag) continue;
            }
            if (why)
                *why = "'" + w.name + "' is stored as '" + w.layout_tag + "' -- what the kernel "
                       "that claimed it asked for -- and " + ki->name + " reads it " +
                       (st < 0 && st != RAD_E_UNSUPPORTED
                            ? std::string("not at all (") + rad_strerror(st) + ")"
                            : tag.empty() ? std::string("as it is") : "as '" + tag + "'") +
                       ". A weight is stored once, so this is a different reading of the same "
                       "bytes and not a fallback";
            return RAD_E_FORMAT;
        }
    }
    return RAD_OK;
}


int Ctx::check_domain(rad_op h, int domain) const {
    if (h == RAD_NULL_HANDLE || h >= plans_.size()) return RAD_E_INVAL;
    if (domain < 0 || domain >= RAD_N_DOMAINS)      return RAD_E_INVAL;
    const OpPlan& p = plans_[h];
    if (!p.info) return RAD_E_INVAL;

    for (int b = 0; b < p.n_bands; ++b) {
        const IssuePlan& ip = issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS + domain];
        if (ip.launch) continue;
        const Band& bd = p.info->bands[(size_t)b];
        RAD_ERR("op '%s' was placed on the %s domain, but band %s<=%lld has no %s kernel: %s",
                p.info->op.c_str(), domain == RAD_DOMAIN_HOST ? "host" : "device",
                p.info->ranged_key.empty() ? "n" : p.info->ranged_key.c_str(),
                (long long)bd.hi, domain == RAD_DOMAIN_HOST ? "host" : "device",
                bd.miss[domain].empty() ? "nothing resolved" : bd.miss[domain].c_str());
        return RAD_E_NOKERNEL;
    }

    /* A kernel existing for the band is not the same as a kernel that can READ THE WEIGHT. */
    std::string why;
    if (check_domain_layout(h, domain, &why) < 0) {
        RAD_ERR("op '%s' cannot run on the %s domain: %s", p.info->op.c_str(),
                domain == RAD_DOMAIN_HOST ? "host" : "device", why.c_str());
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}


int Ctx::set_op_domain(rad_op h, int domain) {
    int s = check_domain(h, domain);
    if (s < 0) return s;
    if (op_domain_[h].exchange((uint8_t)domain, std::memory_order_relaxed) != (uint8_t)domain)
        domain_epoch_.fetch_add(1, std::memory_order_relaxed);
    return RAD_OK;
}

int Ctx::op_domain(rad_op h) const {
    if (h == RAD_NULL_HANDLE || h >= op_domain_.size()) return RAD_E_INVAL;
    return (int)op_domain_[h].load(std::memory_order_relaxed);
}

/* ================================================================== routing (spec §5.5) */

int Ctx::bind_routing() {
    const Program& P = *program_;
    /* THE ROUTED LAYER COUNT IS NOT meta.n_layers, AND A DRAFTER IS WHY.
     *
     * `meta.n_layers` counts TRUNK layers. An MTP head is a further block with its own expert
     * plane, and it reports itself as the layer one past that count. Sized from `meta.n_layers`
     * this vector is one short, so every one of that head's route reports is refused with an error
     * and the heat engine never sees a single count for it. The placement plan still knows about
     * the units but has no measurement to promote on, so they stay wherever the warm start left
     * them -- and the head's experts stream over the link on EVERY draft round of every step,
     * which is worth several percent of the decode step on a model that has one.
     *
     * The weights know the answer and the header does not, so ask them. A drafter, a shared
     * expert plane, anything a plugin declares past the trunk lands here by construction rather
     * than by the core being told about it.
     */
    int64_t n_layer = P.meta.n_layers;
    for (const WeightInfo& w : P.weights)
        if (w.decl.group.expert >= 0)
            n_layer = std::max(n_layer, (int64_t)w.decl.group.layer + 1);
    const int64_t n_expert = P.meta.n_expert;
    if (n_layer <= 0 || n_expert <= 0) return RAD_OK;   /* dense: nothing routes */

    route_n_expert_ = n_expert;
    route_.assign((size_t)n_layer, RouteSlot{});

    /* One pinned block for every bank, and ONE ROW A LAYER on the card. Pinned because the copy
     * has to be a real asynchronous DMA: a pageable destination makes the driver stage it through
     * its own buffer and, on some paths, synchronise -- which would put a host sync on the compute
     * stream, which is the one thing §5.5 says routing must never do.
     *
     * The card's rows are not banked: a pass's reports land in them and the pass's flush carries
     * them to the step's bank on the host, both on the compute stream, so the next pass's reports
     * queue behind the flush that reads them. A report's copy then names the same address every
     * step, which is what lets a recorded pass be replayed without rewriting it. */
    const int64_t elems = n_layer * kRouteBanks * n_expert;
    const int64_t bytes = elems * (int64_t)sizeof(int32_t);
    const int64_t dbytes = n_layer * n_expert * (int64_t)sizeof(int32_t);
    void* p = rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
    void* d = rad_dev_alloc(dbytes, RAD_MEM_DEVICE);
    if (!p || !d) {
        RAD_ERR("routing: %s of %s staging failed: %s", humanb(p ? dbytes : bytes).c_str(),
                p ? "device" : "pinned", rad_dev_last_error());
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
        if (d) rad_dev_free(d, RAD_MEM_DEVICE);
        return RAD_E_NOMEM;
    }
    route_pool_ = static_cast<int32_t*>(p);
    route_dev_  = static_cast<int32_t*>(d);
    std::memset(route_pool_, 0, (size_t)elems * sizeof(int32_t));

    for (int64_t l = 0; l < n_layer; ++l) {
        RouteSlot& r = route_[(size_t)l];
        for (int i = 0; i < kRouteBanks; ++i) {
            r.staging[i] = route_pool_ + ((int64_t)i * n_layer + l) * n_expert;
            r.step[i]    = -1;          /* never written, as against written at step 0 */
        }
    }
    /* Local: the host reads the bank on the strength of this event, but the bytes it reads were
     * put in host memory by the copy in front of it, whose own packets carry the system scope. */
    for (RadEvent& e : route_done_) RAD_TRY(rad_event_create_local(&e));
    return RAD_OK;
}

/* THE PASS'S HISTOGRAMS GO TO THE HOST IN ONE TRANSFER: the layers it reported, which are
 * contiguous rows of the bank, and then the bank's event. Called at the end of every pass. */
int Ctx::route_flush() {
    if (route_bank_ < 0 || route_lo_ > route_hi_) return RAD_OK;
    if (route_d2h_ || std::getenv("RADIANCE_DEBUG_ROUTING") != nullptr) {
        const int64_t n_layer = (int64_t)route_.size();
        const int64_t off = ((int64_t)route_bank_ * n_layer + route_lo_) * route_n_expert_;
        const int64_t n = (int64_t)(route_hi_ - route_lo_ + 1) * route_n_expert_;
        RAD_TRY(rad_memcpy_async(route_pool_ + off, route_dev_ + (int64_t)route_lo_ * route_n_expert_,
                                 n * (int64_t)sizeof(int32_t), stream_));
        RAD_TRY(rad_event_record(route_done_[route_bank_], stream_));
    }
    route_lo_ = 1;
    route_hi_ = 0;
    return RAD_OK;
}

int Ctx::route_report(int layer, const RadRouting* r) {
    if (rec_ && r) {
        tape_cut();
        TapeStep st;
        st.kind = kTapeRoute;
        st.a = (int32_t)rec_->route.size();
        rec_->route.emplace_back(layer, *r);
        rec_->steps.push_back(st);
    }
    return route_report_impl(layer, r, true);
}

int Ctx::route_report_impl(int layer, const RadRouting* r, bool copy) {
    if (!r) return RAD_E_INVAL;
    if (layer < 0 || (size_t)layer >= route_.size()) {
        RAD_ERR("rad_route_report: layer %d is outside the %zu layers the model declared "
                "(or the model has no experts)", layer, route_.size());
        return RAD_E_INVAL;
    }

    RouteSlot& s = route_[(size_t)layer];
    const int step = batch_ ? batch_->step : 0;
    const int bank = ((step % kRouteBanks) + kRouteBanks) % kRouteBanks;

    /* The plugin computed these with ordinary declared ops. This is the handoff, not a
     * computation: we record the device buffers so the mover can be told what was wanted, and
     * nothing here reads them. */
    s.routing  = *r;
    s.reported = step;

    if (r->expert_count) {
        int64_t n = r->n_expert;
        if (n > route_n_expert_) {
            RAD_WARN("rad_route_report: layer %d reported %lld experts, staging holds %lld; "
                     "the histogram is truncated", layer, (long long)n, (long long)route_n_expert_);
            n = route_n_expert_;
        }
        if (n > 0) {
            /* Asynchronous, on the compute stream, and nothing waits on it. The heat engine reads
             * it on the NEXT step, so placement decisions are always one step stale -- which for
             * heat-based promotion is irrelevant, because a hot expert is hot across many steps
             * (spec §5.5).
             *
             * "NOTHING WAITS ON IT" IS TRUE OF THE HOST AND NOT OF THE STREAM. A copy is a blit
             * kernel, so it sits in the compute stream's order and every later kernel queues
             * behind it -- and a copy INTO HOST MEMORY must also make its writes visible to the
             * host before it completes, which waits out every write still crossing the link. With
             * the link carrying streamed experts, that release and the event record after it would
             * cost a routed layer several microseconds of an otherwise idle compute stream.
             *
             * So the layer's counts are copied WITHIN THE CARD, into its row of the step's bank,
             * and the rows this pass wrote cross to the host once, at the end of the pass
             * (route_flush). One card-local blit a routed layer is the whole cost -- and none when
             * the router wrote the row itself (rad_route_counts), which is the report naming it.
             *
             * MOVING IT OFF THE COMPUTE STREAM IS WORTH LESS THAN IT LOOKS. The second lane would
             * have to wait on this stream first, or the copy races the routing kernel that writes
             * `expert_count`. That is a lane_join a routed layer and an event record back on the
             * compute stream, which costs more than the blit it moves. */
            /* RECYCLING A BANK NOBODY READ IS THE RING BEING TOO SHALLOW FOR THIS RUN-AHEAD, and
             * it is the only way evidence is lost, since the drain takes every landed bank. It
             * is counted here, at the one place it can happen, rather than inferred from a total
             * that came up short. A second report of the SAME step is not a recycle: the drain
             * does not take a bank before its step is over, and `reports` carries the pass. */
            const bool again = s.step[bank] == step;
            if (s.step[bank] >= 0 && !again && !s.credited[bank]) ++route_dropped_;
            s.credited[bank] = false;
            if (route_bank_ != bank) {
                /* Empty: every pass flushes at its end. Not recorded if it were not. */
                if (rec_) tape_pause(true);
                const int rf = route_flush();
                if (rec_) tape_pause(false);
                RAD_TRY(rf);
                route_bank_ = bank;
            }
            int32_t* const row = route_dev_ + (int64_t)layer * route_n_expert_;
            if (copy && r->expert_count != row)
                RAD_TRY(rad_memcpy_async(row, r->expert_count, n * (int64_t)sizeof(int32_t),
                                         stream_));
            if (route_lo_ > route_hi_) {
                route_lo_ = route_hi_ = layer;
            } else {
                if (layer < route_lo_) route_lo_ = layer;
                if (layer > route_hi_) route_hi_ = layer;
            }
            s.reports[bank] = again ? s.reports[bank] + 1 : 1;
            s.step[bank] = step;

            /* ==================== WHAT A FAULT LOOKS LIKE IN THIS HISTOGRAM ====================
             *
             * The sum of a layer's counts is the number of placements, which cannot exceed
             * n_tok * top_k. Negative counts, or totals orders of magnitude above that bound, say
             * something OUTSIDE this path wrote over the buffer rather than that the routing
             * kernel is wrong.
             *
             * The shape such an overrun takes is worth knowing, because it hides. A routed expert
             * plane is one row per (token, slot), so a buffer declared at max_tok rows is written
             * `rows * top_k` deep and runs past its declaration -- and the overrun only EXISTS
             * once rows * top_k exceeds the declared extent. A short prefill chunk and a decode
             * step are therefore clean while a full batch is not, and the discriminator is not
             * `n_tok == max_tok` but how far past the end the write reaches and what the arena
             * put there. core/runtime/issue.cpp carries the guard that refuses an operand reaching
             * past its declaration.
             *
             * RADIANCE_DEBUG_ROUTING: a composition fault in a routed model shows up here first.
             * It SYNCHRONISES THE STREAM, which is exactly what the copy above exists to avoid --
             * so it is an environment variable, off, and costs nothing when it is not set. The
             * value is the layer to print. */
            static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
            if (copy && dbg && layer == (int)std::strtol(dbg, nullptr, 10) && step < 3) {
                RAD_TRY(route_flush());
                rad_stream_sync(stream_);
                /* READ THE DEVICE BUFFER AGAIN, SYNCHRONOUSLY, and print both totals. The staging
                 * copy is asynchronous and this print is the only reader of it, so a disagreement
                 * here says the copy raced and an agreement says the histogram itself is what it
                 * is. Debug-only, so the extra transfer costs nothing when the variable is off. */
                std::vector<int32_t> direct((size_t)n, 0);
                int64_t dtot = 0;
                if (rad_memcpy_async(direct.data(), r->expert_count,
                                     n * (int64_t)sizeof(int32_t), stream_) >= 0 &&
                    rad_stream_sync(stream_) >= 0)
                    for (int64_t i = 0; i < n; ++i) dtot += direct[(size_t)i];
                const int32_t* h = s.staging[bank];
                int64_t tot = 0, used = 0, mx = 0; int arg = 0;
                for (int64_t i = 0; i < n; ++i) {
                    tot += h[i];
                    if (h[i]) ++used;
                    if (h[i] > mx) { mx = h[i]; arg = (int)i; }
                }
                std::string first;
                for (int64_t i = 0; i < n && i < 24; ++i)
                    first += (i ? "," : "") + std::to_string(h[i]);
                RAD_WARN("routing layer %d step %d: %lld experts of %lld used, %lld placements "
                         "(direct read %lld), max %lld at e%d; first 24 = [%s]",
                         layer, step, (long long)used, (long long)n, (long long)tot,
                         (long long)dtot, (long long)mx, arg, first.c_str());
            }
        }
    }
    return RAD_OK;
}

/* THE OLDEST LANDED BANK THIS LAYER HAS NOT BEEN CREDITED FOR, or -1 when there is none.
 *
 * OLDEST AND NOT NEWEST, because the caller drains in a loop and the heat engine decays a layer's
 * plane on every observation. Fed newest-first, a burst would apply the most recent decay to the
 * oldest evidence and rank the plane by an order the device never produced.
 *
 * A BANK IS SKIPPED UNTIL ITS COPY HAS COMPLETED. `step[b]` is stamped when the copy is ISSUED,
 * which on a long prefill chunk is many steps before the device reaches it; reading the staging
 * then would hand back whatever the bank's previous occupant left in it, which is a histogram from
 * kRouteBanks steps ago wearing this step's number. The event is the only thing that separates
 * the two, so it is queried rather than assumed -- and that also excludes a half-written bank,
 * which a mid-step reader would otherwise see.
 *
 * A BANK OF A STEP STILL BEING ISSUED IS SKIPPED TOO, landed or not: a later pass of that step can
 * still report into it. See `before_step` in ctx.h.
 */
void Ctx::mark_credited(int layer, int bank) {
    if (layer < 0 || (size_t)layer >= route_.size()) return;
    if (bank < 0 || bank >= kRouteBanks) return;
    route_[(size_t)layer].credited[bank] = true;
}

int Ctx::next_histogram(int layer, int after_step, int before_step, int8_t* landed) const {
    if (layer < 0 || (size_t)layer >= route_.size()) return -1;
    const RouteSlot& s = route_[(size_t)layer];
    int best = -1;
    for (int b = 0; b < kRouteBanks; ++b) {
        if (s.step[b] < 0 || s.step[b] <= after_step || s.step[b] >= before_step) continue;
        if (best >= 0 && s.step[b] >= s.step[best]) continue;
        int8_t ok = landed ? landed[b] : (int8_t)-1;
        if (ok < 0) {
            ok = (route_done_[b] && rad_event_query(route_done_[b]) == 1) ? 1 : 0;
            if (landed) landed[b] = ok;
        }
        if (!ok) continue;
        best = b;
    }
    return best;
}

const int32_t* Ctx::last_histogram(int layer, int bank) const {
    if (layer < 0 || (size_t)layer >= route_.size()) return nullptr;
    if (bank < 0 || bank >= kRouteBanks) return nullptr;
    const RouteSlot& s = route_[(size_t)layer];
    return s.step[bank] >= 0 ? s.staging[bank] : nullptr;
}

int Ctx::last_histogram_step(int layer, int bank) const {
    if (layer < 0 || (size_t)layer >= route_.size()) return -1;
    if (bank < 0 || bank >= kRouteBanks) return -1;
    return route_[(size_t)layer].step[bank];
}

int Ctx::last_histogram_reports(int layer, int bank) const {
    if (layer < 0 || (size_t)layer >= route_.size()) return 0;
    if (bank < 0 || bank >= kRouteBanks) return 0;
    return route_[(size_t)layer].reports[bank];
}

const RadRouting* Ctx::last_routing(int layer) const {
    if (layer < 0 || (size_t)layer >= route_.size()) return nullptr;
    return route_[(size_t)layer].reported >= 0 ? &route_[(size_t)layer].routing : nullptr;
}

/* ================================================================== profiling (spec §16) */

int Ctx::bind_profiling() {
    prof_ms_.assign(program_->ops.size(), 0.0);
    prof_calls_.assign(program_->ops.size(), 0);
    prof_host_.assign(program_->ops.size(), 0);
    prof_join_ms_.assign(program_->ops.size(), 0.0);
    if (!profiling_) return RAD_OK;

    /* Printed, not buried in a doc. Anyone reading a profiled run's numbers has to see this on the
     * same terminal that produced them. */
    RAD_WARN("per-op timing is ON. It restores synchronisations that change the mover's slot "
             "supply, so THIS RUN IS NOT COMPARABLE against an unprofiled one (spec §16). "
             "Anything being measured across builds runs without it.");

    /* A fixed ring, allocated here, because the step path does not allocate. 8192 pairs covers a
     * 64-layer model at ~20 ops a layer with room for speculation; past that we stop timing and
     * say so, rather than growing a vector inside a step.
     *
     * CARD-LOCAL EVENTS, NOT HOST-WAIT ONES. A host-wait record takes an interrupt signal, of which
     * a process has a few thousand, and every pair recorded in a step holds two until the pair is
     * recorded again a step later: a prefill step on a large model records more pairs than that
     * across the ranks, and the engine's own staging event then has no signal left to take. The
     * host never waits on these one at a time -- profile_collect waits for the stream once -- and
     * a card-scope release keeps the records from waiting out writes still on their way to the
     * host, which would be charged to the op they bracket. */
    const size_t cap = 8192;
    prof_ring_.resize(cap);
    for (size_t i = 0; i < cap; ++i) {
        if (rad_event_create_local(&prof_ring_[i].a) < 0 ||
            rad_event_create_local(&prof_ring_[i].b) < 0) {
            RAD_ERR("per-op timing: could not create %zu event pairs", cap);
            return RAD_E_DEVICE;
        }
    }
    return RAD_OK;
}

void Ctx::profile_collect() {
    /* This is the synchronisation the caveat above is about. It is here, at the end of the step,
     * rather than per launch, so at least the ordering inside a step is undisturbed -- but the
     * mover's slot supply still sees a step boundary it would not otherwise see. */
    if (prof_n_ > 0) {
        /* Lane 1 is already joined into the compute stream (end of step), so that stream finishing
         * is every bracketed launch finishing. */
        ++n_host_syncs_;                     /* counted, so the caveat is a number and not a claim */
        if (rad_stream_sync(stream_) < 0) prof_n_ = 0;
    }
    for (size_t i = 0; i < prof_n_; ++i) {
        ProfSlot& s = prof_ring_[i];
        float ms = 0.0f;
        if (rad_event_elapsed_ms(s.a, s.b, &ms) < 0) continue;
        prof_ms_[(size_t)s.op]    += (double)ms;
        prof_calls_[(size_t)s.op] += 1;
    }
    if (prof_overflowed_) {
        RAD_WARN("per-op timing: the step issued more than %zu ops; the tail was not timed",
                 prof_ring_.size());
        prof_overflowed_ = false;
    }
    prof_n_ = 0;
}

void Ctx::dump_profile() const {
    if (!profiling_) { RAD_WARN("per-op timing was not enabled for this run"); return; }
    if (!program_)   { RAD_WARN("per-op timing: the context has already been shut down"); return; }

    fprintf(stderr,
            "\nper-op timing, rank %d. NOT COMPARABLE against an unprofiled run:\n"
            "enabling this restores synchronisations that change the mover's slot supply.\n"
            "A row marked `host` was timed on the HOST CLOCK around a host-site launch and\n"
            "includes everything that call waited on -- page faults, link reads. The rest are\n"
            "device event pairs around a kernel.\n\n"
            "  %10s %12s %10s %5s %10s  %s\n", rank_, "calls", "total ms", "us/call", "site",
            "join us", "op / geometry");

    /* Slowest first: the only ordering anybody reads this table in. */
    std::vector<size_t> order;
    for (size_t i = 1; i < prof_calls_.size(); ++i) if (prof_calls_[i]) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return prof_ms_[a] > prof_ms_[b]; });

    double total = 0.0;
    for (size_t i : order) {
        const OpInfo& o = program_->ops[i];
        const bool ho = i < prof_host_.size() && prof_host_[i];
        char jbuf[32] = "         -";
        if (ho && i < prof_join_ms_.size())
            std::snprintf(jbuf, sizeof jbuf, "%10.2f",
                          prof_join_ms_[i] * 1000.0 / (double)prof_calls_[i]);
        /* The op's position and its first weight name the LAYER: every layer's instance of an
         * op declares the same geometry, so without them sixty rows read alike and a per-layer
         * cost -- residency, a pooled expert streamed by one layer and not the next -- cannot be
         * tied to anything. */
        const char* wname = !o.weights.empty() && o.weights[0] < program_->weights.size()
                                ? program_->weights[o.weights[0]].name.c_str() : "-";
        fprintf(stderr, "  %10lld %12.3f %10.2f %5s %s  %s  %s  #%d %s\n",
                (long long)prof_calls_[i], prof_ms_[i],
                prof_ms_[i] * 1000.0 / (double)prof_calls_[i], ho ? "host" : "dev", jbuf,
                o.op.c_str(), o.base.str().c_str(), (int)o.index, wname);
        total += prof_ms_[i];
    }
    fprintf(stderr, "  %10s %12.3f\n\n", "total", total);
}

/* ================================================================== the step driver */

/* LAZY, AND THAT IS THE POINT: an architecture that never asks for a second lane must take the
 * same code path an architecture with one lane would take if this feature did not exist, so that
 * "the lane costs nothing when unused" is a property of the source rather than a claim about a
 * measurement. Building the stream eagerly would make it one -- and one that is hard to establish,
 * because a second stream's cost is small enough to sit inside the run-to-run spread a warming
 * part produces. */
int Ctx::lane_init() {
    if (lane1_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    RAD_TRY(rad_stream_create(&lane1_, 0));
    own_lane1_ = true;
    /* Local: a lane join orders one card's two queues and nothing else reads through it. */
    for (int i = 0; i < 2; ++i)
        if (!lane_ev_[i]) RAD_TRY(rad_event_create_local(&lane_ev_[i]));
    return RAD_OK;
}

int Ctx::set_lane(int lane) {
    if (lane != 0 && lane != 1) return RAD_E_INVAL;
    if (lane == 0 && !lane1_) { lane_ = 0; return RAD_OK; }   /* lane 0 needs nothing built */
    RAD_TRY(lane_init());
    lane_ = lane;
    return RAD_OK;
}

int Ctx::lane_join(int from, int to) {
    if (from == to) return RAD_OK;
    if (from < 0 || from > 1 || to < 0 || to > 1) return RAD_E_INVAL;
    /* Nothing was ever issued on lane 1, so there is nothing to order against. This is the arm
     * every unpipelined step takes at the end of run_step, and it must not build a stream. */
    if (!lane1_) return RAD_OK;
    RadStream fs = from ? lane1_ : stream_, ts = to ? lane1_ : stream_;
    RAD_TRY(rad_event_record(lane_ev_[from], fs));
    RAD_TRY(rad_event_wait(ts, lane_ev_[from]));
    scratch_open_[from] = false;        /* `to` is now behind every scratch user `from` issued */
    return RAD_OK;
}

/* ================================================================== recorded passes */
/* See "recorded passes" in ctx.h. */

uint64_t Ctx::pass_key(const RadBatch* b) const {
    /* An encoder pass is shaped by its media, and a pass that carries encoder rows issues the
     * scatter that puts them in place only when it has some: neither repeats as a tape would. */
    if (b->enc || b->n_mm_rows > 0) return 0;
    uint64_t h = 0x9e3779b97f4a7c15ull;
    auto mix = [&](uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xbf58476d1ce4e5b9ull;
    };
    auto ptr = [&](const void* p) { mix((uint64_t)(uintptr_t)p); };
    mix((uint64_t)b->phase);           mix((uint64_t)b->n_tok);      mix((uint64_t)b->n_seq);
    mix((uint64_t)b->n_out);           mix((uint64_t)b->max_q_len);  mix((uint64_t)b->max_ctx_len);
    mix((uint64_t)b->n_kv_groups);     mix((uint64_t)b->n_spec);     mix((uint64_t)b->draft_pass);
    mix((uint64_t)b->n_checkpoints);   mix((uint64_t)b->n_seq_decode);
    mix((uint64_t)b->n_tok_decode);    mix((uint64_t)b->max_q_len_decode);
    mix((uint64_t)b->max_q_len_prefill); mix((uint64_t)b->n_draft_out); mix((uint64_t)b->n_ahead);
    mix((uint64_t)b->rope_mixed);      ptr(b->rope_pos);
    ptr(b->token_ids);  ptr(b->positions);  ptr(b->cu_seqlens);    ptr(b->seq_ids);
    ptr(b->out_ids);    ptr(b->q_lens);     ptr(b->ctx_lens);      ptr(b->num_accepted);
    ptr(b->spec_parent); ptr(b->checkpoint_tok); ptr(b->checkpoint_slot); ptr(b->draft_out_ids);
    for (int g = 0; g < b->n_kv_groups; ++g) {
        const RadKVGroupBatch& k = b->kv[g];
        mix((uint64_t)k.group);  ptr(k.slot_mapping); ptr(k.block_table);
        mix((uint64_t)k.block_table_pitch); ptr(k.seqused); ptr(k.state_index);
        mix((uint64_t)k.state_index_pitch); mix((uint64_t)k.block_size); mix((uint64_t)k.max_blocks);
    }
    mix((uint64_t)level_);
    mix(w_tmpl_epoch_);
    mix(dense_epoch_);
    mix(domain_epoch_.load(std::memory_order_relaxed));
    ptr(arena_.base());
    mix((lane1_ ? 1u : 0u) | (host_touch_pending_[0] ? 2u : 0u) | (host_touch_pending_[1] ? 4u : 0u) |
        (scratch_open_[0] ? 8u : 0u) | (scratch_open_[1] ? 16u : 0u));
    return h | 1;
}

bool Ctx::tape_seen(uint64_t key) {
    for (uint64_t k : tape_seen_) if (k == key) return true;
    tape_seen_[tape_seen_at_++ % (sizeof tape_seen_ / sizeof tape_seen_[0])] = key;
    return false;
}

void Ctx::tape_cut() {
    const int m = tape_mark();
    if (m > rec_mark_) {
        TapeStep st;
        st.kind = kTapePlay;
        st.a = rec_mark_;
        st.b = m;
        rec_->steps.push_back(st);
    }
    rec_mark_ = m;
}

void Ctx::tape_drop(PassTape& t) {
    if (t.dev) tape_destroy(t.dev);
    t.dev = nullptr;
}

/* Issues the pass for real, through the architecture, recording it. Returns 1 when `out` holds a
 * tape, 0 when the pass ran but cannot be recorded. */
int Ctx::record_pass(const RadBatch* b, uint64_t key, PassTape* out, bool compare_only) {
    *out = PassTape{};
    out->key = key;
    if (tape_begin() < 0) { arch_step_(this, b); return 0; }
    const int64_t i0 = n_issues_, c0 = ar_calls_.load(), y0 = ar_bytes_.load();
    rec_ = out;
    rec_mark_ = 0;
    haz_ops_.clear();
    haz_rng_.clear();
    arch_step_(this, b);
    tape_cut();
    rec_ = nullptr;
    RadTape t = tape_end(step_status_ >= 0, compare_only);
    if (!t) {
        if (step_status_ >= 0) {
            tape_refused_.push_back(key);
            RAD_WARN("rank %d: draft pass %d at %lld tokens is issued in full every step: %s",
                     rank_, b->draft_pass, (long long)b->n_tok, rad_dev_last_error());
        }
        return 0;
    }
    out->dev = t;
    out->issues = n_issues_ - i0;
    out->ar_calls = ar_calls_.load() - c0;
    out->ar_bytes = ar_bytes_.load() - y0;
    out->touch_pending[0] = host_touch_pending_[0];
    out->touch_pending[1] = host_touch_pending_[1];
    out->scratch_open[0] = scratch_open_[0];
    out->scratch_open[1] = scratch_open_[1];
    if (haz_mark(*out) < 0) {
        tape_destroy(t);
        out->dev = nullptr;
        return 0;
    }
    const int bit = b->draft_pass + 8;
    if (!compare_only && rank_ == 0 && bit >= 0 && bit < 64 && !(tape_told_ & (1ull << bit))) {
        tape_told_ |= 1ull << bit;
        RAD_INFO("recorded pass: draft pass %d at %lld token(s) -- %d records, %d of the %d "
                 "dispatches unordered behind the one before; %zu host op(s), "
                 "%zu table refresh point(s) and %zu routing report(s) between the played "
                 "stretches", b->draft_pass, (long long)b->n_tok, tape_len(t), out->unordered,
                 out->dispatches, out->host.size(),
                 (size_t)std::count_if(out->steps.begin(), out->steps.end(),
                                       [](const TapeStep& st) { return st.kind == kTapeTable; }),
                 out->route.size());
    }
    return 1;
}

/* WHICH DISPATCHES OF A RECORDED PASS CAN START BEFORE THE ONES IN FRONT OF THEM FINISH.
 *
 * Walked once, in queue order, per queue: the ops launched since the queue last ordered a packet
 * are in flight together, and the next op may join them when it reads nothing they write and
 * writes nothing they read or write. Anything the walk cannot see into orders its packet: the
 * dispatch after an event, every dispatch after an op's first (an op's own dispatches depend on
 * each other), a dispatch no op owns (a routing copy), and an opaque op on either side. A run is
 * cut at kTapeOverlapRun, the bound the backend reclaims its argument slots on.
 *
 * The host work between played stretches needs nothing here. A host op joins the device work it
 * depends on before it runs, and whatever a stretch of host work puts on a queue -- a table patch,
 * a mover wait -- makes the backend order the next played packet on that queue itself. */
int Ctx::haz_mark(PassTape& t) {
    const int n = tape_len(t.dev);
    if (n < 0) return n;
    std::vector<int32_t> first_of((size_t)n, -1), part_of((size_t)n, -1);
    for (size_t k = 0; k < haz_ops_.size(); ++k) {
        const HazOp& o = haz_ops_[k];
        if (o.rec0 < 0 || o.rec1 > n) continue;
        first_of[(size_t)o.rec0] = (int32_t)k;
        for (int32_t r = o.rec0; r < o.rec1; ++r) part_of[(size_t)r] = (int32_t)k;
    }
    auto overlaps = [&](uint32_t a, uint32_t na, uint32_t b, uint32_t nb) {
        for (uint32_t i = 0; i < na; ++i)
            for (uint32_t j = 0; j < nb; ++j)
                if (haz_rng_[a + i].first < haz_rng_[b + j].second &&
                    haz_rng_[b + j].first < haz_rng_[a + i].second)
                    return true;
        return false;
    };
    struct Q {
        uint32_t id = 0;
        std::vector<int32_t> infl;     /* ops in flight since the last ordered packet */
        bool opaque = true;            /* something unknown is among them */
        int  run = 0;
    };
    std::vector<Q> qs;
    for (int i = 0; i < n; ++i) {
        int kind = 0;
        uint32_t qid = 0;
        const int rc = tape_record(t.dev, i, &kind, &qid);
        if (rc < 0) return rc;
        Q* q = nullptr;
        for (Q& x : qs) if (x.id == qid) { q = &x; break; }
        if (!q) { qs.push_back(Q{}); q = &qs.back(); q->id = qid; }
        /* An event's packet, and the dispatch after it is ordered too: a wait is complete only
         * once the event it names is, and nothing after it may start before that. */
        if (kind != 0) { q->infl.clear(); q->opaque = true; q->run = 0; continue; }
        ++t.dispatches;

        const int32_t op = first_of[(size_t)i];
        bool ok = op >= 0 && !q->opaque && !haz_ops_[(size_t)op].opaque && q->run < kTapeOverlapRun;
        if (ok) {
            const HazOp& h = haz_ops_[(size_t)op];
            for (int32_t x : q->infl) {
                const HazOp& f = haz_ops_[(size_t)x];
                if (overlaps(h.rd, h.n_rd, f.wr, f.n_wr) || overlaps(h.wr, h.n_wr, f.rd, f.n_rd) ||
                    overlaps(h.wr, h.n_wr, f.wr, f.n_wr)) { ok = false; break; }
            }
        }
        if (ok) {
            const int sc = tape_set_overlap(t.dev, i, true);
            if (sc < 0) return sc;
            q->infl.push_back(op);
            ++q->run;
            ++t.unordered;
            continue;
        }
        /* Ordered: everything before it has finished, and it is what is in flight now. */
        const int32_t own = part_of[(size_t)i];
        q->infl.clear();
        if (own >= 0) q->infl.push_back(own);
        q->opaque = own < 0 || haz_ops_[(size_t)own].opaque;
        q->run = 0;
    }
    return RAD_OK;
}

/* Two recordings of one key, compared: the backend's records, and the host work between them. */
bool Ctx::tapes_agree(const PassTape& a, const PassTape& b, std::string* why) {
    if (a.steps.size() != b.steps.size()) {
        *why = fmt("%zu stretches against %zu", a.steps.size(), b.steps.size());
        return false;
    }
    for (size_t i = 0; i < a.steps.size(); ++i) {
        const TapeStep& x = a.steps[i];
        const TapeStep& y = b.steps[i];
        if (x.kind != y.kind || x.lane != y.lane || x.opd != y.opd || x.a != y.a || x.b != y.b) {
            *why = fmt("stretch %zu: kind %d [%d, %d) lane %d against kind %d [%d, %d) lane %d", i,
                       x.kind, x.a, x.b, x.lane, y.kind, y.a, y.b, y.lane);
            return false;
        }
    }
    for (size_t i = 0; i < a.host.size(); ++i) {
        const TapeHost& x = a.host[i];
        const TapeHost& y = b.host[i];
        bool same = x.h == y.h && x.launch == y.launch && x.n == y.n && x.drain == y.drain &&
                    x.args.n_t == y.args.n_t && x.args.scratch == y.args.scratch;
        for (int k = 0; same && k < x.args.n_t; ++k)
            same = std::memcmp(&x.t[k], &y.t[k], sizeof(RadTensor)) == 0;
        if (!same) { *why = fmt("host op %zu (op %u) differs", i, (unsigned)x.h); return false; }
    }
    for (size_t i = 0; i < a.route.size(); ++i)
        if (a.route[i].first != b.route[i].first ||
            std::memcmp(&a.route[i].second, &b.route[i].second, sizeof(RadRouting)) != 0) {
            *why = fmt("routing report %zu (layer %d) differs", i, a.route[i].first);
            return false;
        }
    std::string d;
    if (tape_diff(a.dev, b.dev, &d) >= 0) { *why = d; return false; }
    return true;
}

int Ctx::play_pass(PassTape& t) {
    const int l0 = lane_;
    int play_from = -1;
    int play_to = -1;

    auto flush_play = [&]() -> int {
        if (play_from >= 0 && play_to > play_from) {
            const int rc = tape_play(t.dev, play_from, play_to);
            play_from = play_to = -1;
            if (rc < 0) {
                step_fail_ = fmt("a recorded pass could not be submitted: %s",
                                 rad_dev_last_error());
                return rc;
            }
        }
        return RAD_OK;
    };

    for (const TapeStep& st : t.steps) {
        int rc = RAD_OK;
        switch (st.kind) {
        case kTapePlay:
            if (play_from < 0) {
                play_from = st.a;
                play_to = st.b;
            } else if (st.a == play_to) {
                play_to = st.b;
            } else {
                rc = flush_play();
                if (rc < 0) break;
                play_from = st.a;
                play_to = st.b;
            }
            break;
        case kTapeTable: {
            WeightTable& wt = wtabs_[(size_t)st.a];
            if (wt.full || !wt.pend.empty()) {
                rc = flush_play();
                if (rc < 0) break;
                lane_ = st.lane;
                rc = table_refresh((rad_op)st.b, st.opd, wt);
                lane_ = l0;
            }
            break;
        }
        case kTapeHost: {
            rc = flush_play();
            if (rc < 0) break;
            TapeHost& th = t.host[(size_t)st.a];
            lane_ = st.lane;
            rc = host_join_wait(th.drain);
            if (rc >= 0) {
                th.args.t = th.t;
                rc = th.launch(&th.args, cur_stream());
                rad_dev_host_wrote();
            }
            lane_ = l0;
            if (rc < 0) step_fail_ = fmt("a host op of a recorded pass (op %u) failed: %s",
                                         (unsigned)th.h, rad_strerror(rc));
            break;
        }
        case kTapeRoute:
            rc = route_report_impl(t.route[(size_t)st.a].first, &t.route[(size_t)st.a].second,
                                   false);
            break;
        }
        if (rc < 0) {
            if (step_status_ >= 0) step_status_ = rc;
            return rc;
        }
    }
    const int rc = flush_play();
    if (rc < 0) {
        if (step_status_ >= 0) step_status_ = rc;
        return rc;
    }
    n_issues_ += t.issues;
    ar_calls_.fetch_add(t.ar_calls, std::memory_order_relaxed);
    ar_bytes_.fetch_add(t.ar_bytes, std::memory_order_relaxed);
    host_touch_pending_[0] = t.touch_pending[0];
    host_touch_pending_[1] = t.touch_pending[1];
    scratch_open_[0] = t.scratch_open[0];
    scratch_open_[1] = t.scratch_open[1];
    ++t.plays;
    t.used = ++tape_clock_;
    return RAD_OK;
}

/* THE PASS: played from its tape, or issued -- and recorded, or checked against its tape. */
int Ctx::run_pass(const RadBatch* b) {
    const auto bump = [](std::atomic<int64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); };
    scratch_cur_[0] = scratch_cur_[1] = 0;
    /* Winner replay reads host-mapped path values, not merely the shape-keyed
     * device operands. Its dispatch list must be issued live on every commit. */
    if (!tape_on_ || staging_ || b->draft_pass == -2) {
        bump(pc_unkeyed_); arch_step_(this, b); return RAD_OK;
    }
    const uint64_t key = pass_key(b);
    if (key == 0) { bump(pc_unkeyed_); arch_step_(this, b); return RAD_OK; }
    if (std::find(tape_refused_.begin(), tape_refused_.end(), key) != tape_refused_.end()) {
        bump(pc_refused_);
        arch_step_(this, b);
        return RAD_OK;
    }
    PassTape* t = nullptr;
    for (PassTape& x : tapes_) if (x.key == key) { t = &x; break; }

    if (t && t->plays > 0 && t->plays % kAuditPlays != 0) {
        bump(pc_played_);
        pc_dispatched_.fetch_add(t->dispatches, std::memory_order_relaxed);
        pc_unordered_.fetch_add(t->unordered, std::memory_order_relaxed);
        return play_pass(*t);
    }

    if (t) {
        bump(pc_audited_);
        /* The check: issue the pass for real, recording it, and compare. The first comparison
         * comes before the tape is ever played, and then one every kAuditPlays plays. */
        PassTape fresh;
        if (!record_pass(b, key, &fresh, /*compare_only=*/true)) return RAD_OK;
        std::string why;
        if (!tapes_agree(*t, fresh, &why)) {
            tape_drop(*t);
            *t = std::move(fresh);
            t->used = ++tape_clock_;
            step_fail_ = fmt("rank %d: draft pass %d at %lld tokens issued differently from its "
                             "recording after %lld plays, so the pass depends on something the "
                             "recording's key does not carry: %s", rank_, b->draft_pass,
                             (long long)b->n_tok, (long long)t->plays, why.c_str());
            RAD_ERR("%s", step_fail_.c_str());
            step_status_ = RAD_E_STATE;
            return RAD_E_STATE;
        }
        tape_drop(fresh);
        ++t->plays;
        t->used = ++tape_clock_;
        return RAD_OK;
    }

    /* A KEY IS RECORDED THE SECOND TIME IT IS SEEN, so a pass that never recurs -- a prefill
     * chunk, whose token count and positions are its own -- costs no recording and does not push
     * a tape that will be played out of the least-recently-used set. A DECODE pass recurs by
     * construction: the same shape runs every step until a sequence's context crosses its bound's
     * bucket, and then the new key runs every step after that. Seen-once is a whole live issue
     * spent learning that, at every crossing, so a decode pass is recorded the first time. The
     * audit before its first play is unchanged. */
    if (b->phase != RAD_PHASE_DECODE && !tape_seen(key)) {
        bump(pc_first_);
        arch_step_(this, b);
        return RAD_OK;
    }

    bump(pc_recorded_);
    PassTape nt;
    if (!record_pass(b, key, &nt)) return RAD_OK;
    nt.used = ++tape_clock_;
    if (tapes_.size() < kTapes) {
        tapes_.push_back(std::move(nt));
    } else {
        PassTape* lru = &tapes_[0];
        for (PassTape& x : tapes_) if (x.used < lru->used) lru = &x;
        tape_drop(*lru);
        *lru = std::move(nt);
    }
    return RAD_OK;
}

int Ctx::run_step(const RadBatch* b) {

    if (!b) return RAD_E_INVAL;
    if (!arch_step_) {
        RAD_WARN("Ctx::run_step: no architecture step function was installed");
        return RAD_E_STATE;
    }

    /* The mover relocates between steps, never during one. Honouring WeightInfo::generation here
     * costs one relaxed atomic load per step instead of one per weight per issue. */
    if (weights_dirty_.exchange(false, std::memory_order_acq_rel))
        bind_weights(/*refresh_only=*/true);
    sync_residency();

    batch_       = b;
    step_status_ = RAD_OK;
    step_fail_.clear();
    prof_n_      = 0;
    in_step_     = true;

    staging_ = stager_ != nullptr && stager_->begin_pass(b);
    {
        const int rc = run_pass(b);
        if (rc < 0 && step_status_ >= 0) step_status_ = rc;
    }

    /* EVERY LANE JOINS BACK BEFORE THE STEP ENDS, whatever the architecture did or forgot to do.
     * The engine synchronises the COMPUTE stream to know a step is finished; work still queued on
     * lane 1 would be read by the sampler, or freed under, while it was in flight. Ordering lane 0
     * behind lane 1 here costs one event record and one wait on a step that never used lane 1 --
     * and it means a plugin cannot leave the runtime in a state where that is possible. */
    lane_ = 0;
    lane_join(1, 0);                    /* a no-op, and allocates nothing, when lane 1 is unbuilt */;
    /* Behind the join, so the compute stream is behind every reader of what the stager published;
     * and whatever the pass's status, because a staged weight must never outlive its pass. */
    if (staging_) {
        const int se = stager_->end_pass(stream_);
        if (se < 0 && step_status_ >= 0) step_status_ = se;
        staging_ = false;
    }
    {
        const int rf = (!b || b->draft_pass == 0) ? route_flush() : RAD_OK;
        if (rf < 0 && step_status_ >= 0) step_status_ = rf;
    }

    in_step_ = false;
    if (profiling_) profile_collect();
    flush_issued();
    return step_status_;
}

/* WHICH DECLARED OPS THE RUN PHASE HAS ACTUALLY ISSUED, copied out of the per-step plans into the
 * Program the graph dump reads. Copies flags for ops issued for the first time during the step. */
void Ctx::flush_issued() {
    for (uint32_t i : first_issued_) {
        if (i < program_->ops.size())
            program_->ops[i].issued = true;
    }
    first_issued_.clear();
}

}  /* namespace rad */
