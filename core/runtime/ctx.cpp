/* ctx.cpp -- the run phase outside the hot path: arena and plan bring-up, the step driver, the
 * routing handoff and opt-in per-op timing. The hot path is issue.cpp.
 */
#include <vector>
#include "ctx.h"
#include "xp_state.h"
#include "../device/device.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>

namespace rad {

/* ================================================================== lifecycle */

Ctx::Ctx() = default;
Ctx::~Ctx() { shutdown(); }

int Ctx::init(const CtxDesc& d) {
    if (!d.program) return RAD_E_INVAL;
    program_   = d.program;
    rank_      = d.rank;
    world_size_= d.world_size < 1 ? 1 : d.world_size;
    device_    = d.device;
    record_passes_ = d.record_passes;
    residency_ = d.residency;
    registry_ = d.registry;
    kv_        = d.kv;
    arch_step_ = d.arch_step;
    profiling_ = d.profile_ops;

    /* The rank owns one device. Set it here so allocation lands on the right card, and again from
     * the rank thread in bind_thread() because the backend's current device is thread-local. */
    RAD_TRY(rad_dev_set(device_));

    if (d.stream) {
        stream_     = d.stream;
        own_stream_ = false;
    } else {
        /* Ordinary priority: compute waits on the mover's events (spec §5.4), so a high-priority
         * compute stream would preempt the transfers it is about to wait for. */
        RAD_TRY(rad_stream_create(&stream_, 0));
        own_stream_ = true;
    }

    /* The second lane is created lazily; see lane_init(). */

    int s = prepare();
    if (s < 0) { shutdown(); return s; }
    return RAD_OK;
}

void Ctx::shutdown() {
    oracle_close();
    xp_close();
    /* Teardown sync, before freeing memory the streams may still use: the routing staging is the
     * destination of unwaited copies (spec §5.5) and the arena holds every kernel's operands. */
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

    /* After the syncs: the device half is a grouped GEMM operand, the host half a copy source. */
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
    RAD_TRY(bind_weights());
    RAD_TRY(bind_weight_tables());
    RAD_TRY(bind_ops());
    RAD_TRY(bind_routing());
    RAD_TRY(bind_profiling());
    /* Recorded passes need a recording backend, and are off whenever every issue must be seen:
     * per-op profile, oracle, and a calibration tap. A `gram_accum` sums across passes while its
     * plugin counts the rows it has summed in step and writes the Grams once a token target is
     * reached; a played pass reissues the accumulation without calling step, so the files would
     * hold more rows than their header says and be written late. */
    const bool taps = std::any_of(program_->ops.begin(), program_->ops.end(),
                                  [](const OpInfo& o) { return o.op == "gram_accum"; });
    tape_on_ = record_passes_ && tape_supported() && !profiling_ && !oracle_ && !taps;
    if (taps && record_passes_ && tape_supported() && rank_ == 0)
        RAD_INFO("recorded passes: off, the program accumulates calibration Grams (gram_accum)");
    return RAD_OK;
}

int Ctx::bind_arena() {
    Program& P = *program_;

    /* Sizing is arena_plan_bytes(), shared with the VRAM budget (core/mem/vram_budget.cpp). Here:
     * refuse a plan that did not run, and fill in sizes the plan left at zero. RAD_DOMAIN_HOST
     * buffers (host-site execution, spec §5.1) get a second arena. */
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
    /* The arena is in no pool and scales with max_tok, so core/mem/vram_budget.cpp charges it by
     * name rather than leaving it to --gpu-headroom-mib. This log line must agree with that
     * charge. */
    RAD_INFO("arena: %s device + %d x %s scratch = %s, charged to the VRAM budget before it was "
             "split", humanb(dev_bytes).c_str(), arena_.scratch_regions(),
             humanb(P.scratch_bytes).c_str(), humanb(total).c_str());

    if (host_hw > 0) {
        /* RAD_MEM_HOST_MAPPED, not RAD_MEM_HOST: nothing copies between the two arenas, so device
         * kernels must read host-domain buffers directly over the link (spec §5.1, row three,
         * applied to activations). Plain RAD_MEM_HOST pages are not device-visible
         * (core/device/hip.cpp). Every device access is a link read, so host-domain buffers suit
         * only the hand-off at a host run's edge; the architecture declares the domain. */
        RAD_TRY(host_arena_.init(host_hw, 0, RAD_MEM_HOST_MAPPED));
        /* bind_buffers stores one pointer per buffer, valid only while the mapped allocation's
         * host and device views share an address (true under HIP unified addressing and on the
         * host backend). Checked, not assumed. */
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
    buf_tmpl_.assign(P.buffers.size(), RadTensor{});
    buf_bits_.assign(P.buffers.size(), 0);
    buf_host_.assign(P.buffers.size(), 0);

    for (size_t i = 1; i < P.buffers.size(); ++i) {
        BufferInfo& bi = P.buffers[i];
        RadTensor&  t  = buf_tmpl_[i];

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

        /* rad_buf_ptr is arena_base + BufferInfo::arena_offset. */
        t.data  = a.base() + bi.arena_offset;
        bi.ptr  = t.data;
        buf_bits_[i] = rad_dtype_bits(bi.decl.dtype);
        buf_host_[i] = bi.decl.domain == RAD_DOMAIN_HOST ? 1 : 0;
    }
    level_tmpl_.assign(1, buf_tmpl_);
    level_ = 0;
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
        /* The host arena and the fixed region are not levelled: level 0's template. */
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
    return (int)level_tmpl_.size() - 1;
}

int Ctx::set_level(int level) {
    if (level < 0 || level >= (int)level_tmpl_.size()) return RAD_E_INVAL;
    if (level == level_) return RAD_OK;
    buf_tmpl_ = level_tmpl_[(size_t)level];
    level_ = level;
    return RAD_OK;
}

/* The weights' addresses as the load phase left them. A weight the mover relocates afterwards is
 * read through the ResidencyTable (sync_residency), never through its template. */
int Ctx::bind_weights() {
    Program& P = *program_;
    w_tmpl_.assign(P.weights.size(), RadTensor{});
    w_bits_.assign(P.weights.size(), 0);
    for (auto& g : waited_gen_) g.assign(P.weights.size(), UINT32_MAX);
    w_direct_.assign(P.weights.size(), 0);

    for (size_t i = 1; i < P.weights.size(); ++i) {
        const WeightInfo& wi = P.weights[i];
        RadTensor& t = w_tmpl_[i];
        /* Stored dtype (from the resolved kernel's layout hook) wins over the declared one, since
         * the kernel reads the layout it chose.
         *
         * The shape is deliberately the declared one, not WeightInfo::stored_shape: a row-sharded
         * weight's stored shape is the whole tensor's, not this rank's, so a re-laying kernel
         * derives its own stride anyway (e.g. libr4d's fp8 lm_head appends row scales to each
         * row). Consistency is enforced by the layout tag checked at load. The declared shape is
         * per rank (spec §9). */
        t.dtype = wi.stored_dtype != RAD_DT_INVALID ? wi.stored_dtype : wi.decl.dtype;
        t.rank  = wi.decl.rank;
        for (uint32_t d = 0; d < wi.decl.rank && d < RAD_MAX_RANK; ++d) t.shape[d] = wi.decl.shape[d];
        rad_tensor_pack(&t);
        t.data  = wi.ptr;
        w_bits_[i] = rad_dtype_bits(t.dtype);
    }
    return RAD_OK;
}

int Ctx::bind_ops() {
    Program& P = *program_;

    plans_.assign(P.ops.size(), OpPlan{});
    unflushed_.clear();
    unflushed_.reserve(P.ops.size());
    for (uint32_t i = 1; i < (uint32_t)P.ops.size(); ++i) unflushed_.push_back(i);
    op_domain_ = std::vector<std::atomic<uint8_t>>(P.ops.size());
    band_table_.clear();
    issue_table_.clear();
    gate_params_.clear();

    int64_t n_bands_total = 0;
    for (auto& o : P.ops) n_bands_total += (int64_t)o.bands.size();
    band_table_.reserve((size_t)n_bands_total);
    issue_table_.reserve((size_t)(n_bands_total * RAD_N_DOMAINS));

    for (size_t i = 1; i < P.ops.size(); ++i) {
        OpInfo& oi = P.ops[i];
        OpPlan& p  = plans_[i];
        p.info     = &oi;
        p.band_off = (int32_t)band_table_.size();
        p.n_bands  = (int32_t)oi.bands.size();
        p.plan_off = (int32_t)issue_table_.size();
        p.n_opd_schema = oi.schema ? (int32_t)oi.schema->n_operands : -1;
        /* Matched by name once here, not on the issue path; see OpPlan::billed. */
        p.billed     = (oi.op == "all_reduce");
        p.collective = p.billed;
        p.lora_site  = oi.lora_site;

        for (const Band& b : oi.bands) band_table_.push_back(b.hi);

        for (const Band& b : oi.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                IssuePlan ip{};
                const Resolved& r = b.dom[d];
                if (r && r.row && r.row->info && r.row->info->launch) {
                    ip.row     = r.row;
                    ip.launch  = r.row->info->launch;
                    ip.ranged_div = r.ranged_div;
                    if (r.row->collective) p.collective = true;

                    /* args.p views the frozen geometry inside Program::ops; declare must not
                     * touch a Resolved after this, or these pointers dangle. */
                    ip.args.t             = tbuf_;
                    ip.args.n_t           = 0;
                    ip.args.p             = r.geom.params();
                    ip.args.n_p           = r.geom.n_params();
                    ip.args.scratch       = arena_.scratch();
                    ip.args.scratch_bytes = r.scratch_bytes;
                    ip.args.instance      = r.instance;
                    ip.args.rank          = rank_;
                    ip.args.world_size    = world_size_;
                    /* A launch that ends its own link hold gets the hold word as an extra
                     * `link_gate` parameter after the geometry's. */
                    if (r.row->releases_link)
                        if (void* g = rad_link_gate(stream_)) {
                            std::vector<RadParam>& v =
                                gate_params_.emplace_back(ip.args.p, ip.args.p + ip.args.n_p);
                            RadParam q{};
                            q.key  = "link_gate";
                            q.kind = RAD_P_INT;
                            q.ival = (long long)(uintptr_t)g;
                            v.push_back(q);
                            ip.gate_p   = v.data();
                            ip.gate_n_p = (int)v.size();
                        }

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

        /* Execution site follows the weights (spec §5.1): an op whose every weight is on the host
         * runs its host kernel, everything else the device. An op with no weights stays on the
         * device; the planner overrides through set_op_domain.
         *
         * On the host backend (CPU-only), an op with no device kernel in any band runs its host
         * kernel (libavx or libref). Device rows there come from tests registering fakes. */
        int dom = RAD_DOMAIN_DEVICE;
        if (device_is_host()) {
            bool any_device = false;
            for (int b = 0; b < p.n_bands && !any_device; ++b)
                any_device = issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS +
                                          RAD_DOMAIN_DEVICE].launch != nullptr;
            if (!any_device) dom = RAD_DOMAIN_HOST;
        }
        if (dom == RAD_DOMAIN_DEVICE && !oi.weights.empty()) {
            bool all_host = true;
            for (rad_weight w : oi.weights) {
                const WeightInfo* wi = P.weight(w);
                if (!wi || wi->site != Site::Host) { all_host = false; break; }
            }
            /* The host kernel must also read the stored layout. The device is always correct,
             * so this falls back with a warning rather than refusing. */
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

/* Whether the kernel serving this op on `domain` can read the bytes the container holds.
 *
 * A weight is stored once, in the arrangement its claiming kernel asked for (spec 4.3), and the
 * claimant is a device row wherever one reads the weight: rad_builder.cpp lets a host row claim
 * only what no device row did. A host kernel reading as declared would see e.g. libr4d's fp8a8
 * WMMA fragment order -- same N*K bytes, so every shape check passes -- and compute plausible
 * wrong answers. The load-time tag check in engine_bringup only checks the claiming row.
 *
 * `why` gets the message. The caller picks the severity: the planner's host request falls back
 * to the device, an explicit set_op_domain has no fallback. */
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
            /* A row with no hook, or one that declines this operand, reads the selected plane as
             * stored (as rad_builder.cpp interprets it). One that cannot read the encoding at all
             * cannot take the op. */
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

    /* A kernel existing for every band does not mean it can read the stored weight. */
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
    /* The routed layer count comes from the expert weights, not meta.n_layers, which counts only
     * trunk layers. An MTP head (or any expert block declared past the trunk) reports as a layer
     * beyond that count; without a slot its reports are refused and the heat engine never places
     * its experts. */
    int64_t n_layer = P.meta.n_layers;
    for (const WeightInfo& w : P.weights)
        if (w.decl.group.expert >= 0)
            n_layer = std::max(n_layer, (int64_t)w.decl.group.layer + 1);
    const int64_t n_expert = P.meta.n_expert;
    if (n_layer <= 0 || n_expert <= 0) return RAD_OK;   /* dense: nothing routes */

    route_n_expert_ = n_expert;
    route_.assign((size_t)n_layer, RouteSlot{});

    /* One pinned host block for all banks, and one row per layer on the card. Pinned so the copy
     * is a true async DMA; a pageable destination can make the driver synchronise, and routing
     * must never host-sync (spec §5.5).
     *
     * The card rows are not banked: reports and the end-of-pass flush are both on the compute
     * stream, so the next pass's reports queue behind the flush. A report's copy then targets the
     * same address every step, so recorded passes replay unchanged. */
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
            r.step[i]    = -1;          /* never written (distinct from step 0) */
        }
    }
    /* Local scope is enough: the preceding copy's own packets carry the system-scope release that
     * makes the bank visible to the host. */
    for (RadEvent& e : route_done_) RAD_TRY(rad_event_create_local(&e));
    return RAD_OK;
}

/* Copies the pass's reported rows [route_lo_, route_hi_] to the step's host bank in one transfer
 * and records the bank's event. Called at the end of every pass. */
int Ctx::route_flush() {
    if (route_bank_ < 0 || route_lo_ > route_hi_) return RAD_OK;
    const int64_t n_layer = (int64_t)route_.size();
    const int64_t off = ((int64_t)route_bank_ * n_layer + route_lo_) * route_n_expert_;
    const int64_t n = (int64_t)(route_hi_ - route_lo_ + 1) * route_n_expert_;
    RAD_TRY(rad_memcpy_async(route_pool_ + off, route_dev_ + (int64_t)route_lo_ * route_n_expert_,
                             n * (int64_t)sizeof(int32_t), stream_));
    RAD_TRY(rad_event_record(route_done_[route_bank_], stream_));
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

    /* A handoff only: records the plugin's device buffers for the mover; nothing here reads
     * them. */
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
            /* Async on the compute stream; the host never waits. The heat engine reads it a step
             * later, which is fine since hot experts stay hot (spec §5.5).
             *
             * The copy still sits in stream order, and a copy into host memory must wait out
             * every in-flight link write before completing. So each layer's counts are copied
             * within the card into its row, and the pass's rows cross to the host once at the
             * end (route_flush). No copy at all when the router wrote the row itself
             * (rad_route_counts). Moving the blit to the second lane would need a lane_join per
             * layer to avoid racing the routing kernel, which costs more than the blit. */
            /* Recycling an uncredited bank means the ring is shallower than the host's run-ahead;
             * it is the only way evidence is lost, so it is counted here. A second report of the
             * same step is not a recycle (`reports` counts it). */
            const bool again = s.step[bank] == step;
            if (s.step[bank] >= 0 && !again && !s.credited[bank]) ++route_dropped_;
            s.credited[bank] = false;
            if (route_bank_ != bank) {
                /* Normally a no-op, since every pass flushes at its end; kept out of the tape. */
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

            /* RADIANCE_DEBUG_ROUTING=<layer>: prints that layer's histogram for the first steps.
             * It synchronises the stream, so it is off by default.
             *
             * A layer's counts sum to at most n_tok * top_k. Negative counts or far larger totals
             * mean something outside this path overwrote the buffer, e.g. an expert plane declared
             * at max_tok rows but written rows * top_k deep (refused by the operand guard in
             * issue.cpp). */
            static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
            if (copy && dbg && layer == (int)std::strtol(dbg, nullptr, 10) && step < 3) {
                RAD_TRY(route_flush());
                rad_stream_sync(stream_);
                /* Also read the device buffer directly: a mismatch with the staged copy means the
                 * copy raced. */
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

void Ctx::mark_credited(int layer, int bank) {
    if (layer < 0 || (size_t)layer >= route_.size()) return;
    if (bank < 0 || bank >= kRouteBanks) return;
    route_[(size_t)layer].credited[bank] = true;
}

/* The oldest landed bank in (after_step, before_step) for this layer, or -1.
 *
 * Oldest first, because the heat engine decays a layer's plane on every observation and must see
 * them in device order. A bank is skipped until its event says the copy completed: `step[b]` is
 * stamped at issue, and the staging would otherwise still hold the previous occupant's histogram.
 * A bank of a step still being issued is skipped too (see `before_step` in ctx.h).
 */
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
    prof_wait_ms_.assign(program_->ops.size(), 0.0);
    if (!profiling_) return RAD_OK;

    RAD_WARN("per-op timing is on; it adds synchronisations that change the mover's slot supply, "
             "so this run is not comparable with an unprofiled one (spec §16)");

    /* A fixed ring, since the step path does not allocate. 8192 pairs covers a 64-layer model at
     * ~20 ops a layer with room for speculation; overflow stops timing with a warning.
     *
     * Card-local events: a host-wait event takes one of a process's few thousand interrupt
     * signals, and a large prefill step would exhaust them. The host only waits on the stream
     * (profile_collect), and a card-scope release does not wait out in-flight host writes that
     * would be charged to the bracketed op. */
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
    /* The profiling sync, once at the end of the step so ordering within the step is
     * undisturbed. */
    if (prof_n_ > 0) {
        /* Lane 1 is already joined into the compute stream, so this covers every launch. */
        ++n_host_syncs_;
        if (rad_stream_sync(stream_) < 0) prof_n_ = 0;
    }
    /* Idle time before each op, per lane. A lane's first op in the step is not charged: its gap
     * is mostly this collection's own sync. */
    const ProfSlot* prev[2] = { nullptr, nullptr };
    for (size_t i = 0; i < prof_n_; ++i) {
        ProfSlot& s = prof_ring_[i];
        float ms = 0.0f;
        if (rad_event_elapsed_ms(s.a, s.b, &ms) < 0) continue;
        prof_ms_[(size_t)s.op]    += (double)ms;
        prof_calls_[(size_t)s.op] += 1;
        const ProfSlot*& p = prev[s.lane & 1];
        float gap = 0.0f;
        if (p && rad_event_elapsed_ms(p->b, s.a, &gap) >= 0 && gap > 0.0f)
            prof_wait_ms_[(size_t)s.op] += (double)gap;
        p = &s;
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
            "\nper-op timing, rank %d. Not comparable with an unprofiled run:\n"
            "enabling this restores synchronisations that change the mover's slot supply.\n"
            "A row marked `host` was timed on the host clock around a host-site launch and\n"
            "includes everything that call waited on -- page faults, link reads. The rest are\n"
            "device event pairs around a kernel. `wait us` is how long, a call, the op's lane\n"
            "sat idle before it started -- stream waits and issue delays, in no op's time.\n\n"
            "  %10s %12s %10s %5s %10s %10s  %s\n", rank_, "calls", "total ms", "us/call", "site",
            "join us", "wait us", "op / geometry");

    /* Slowest first. */
    std::vector<size_t> order;
    for (size_t i = 1; i < prof_calls_.size(); ++i) if (prof_calls_[i]) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return prof_ms_[a] > prof_ms_[b]; });

    double total = 0.0, waited = 0.0;
    for (size_t i : order) {
        const OpInfo& o = program_->ops[i];
        const bool ho = i < prof_host_.size() && prof_host_[i];
        char jbuf[32] = "         -";
        if (ho && i < prof_join_ms_.size())
            std::snprintf(jbuf, sizeof jbuf, "%10.2f",
                          prof_join_ms_[i] * 1000.0 / (double)prof_calls_[i]);
        /* The op index and first weight identify the layer, since every layer's instance has
         * the same geometry. */
        const char* wname = !o.weights.empty() && o.weights[0] < program_->weights.size()
                                ? program_->weights[o.weights[0]].name.c_str() : "-";
        const double wait = i < prof_wait_ms_.size() ? prof_wait_ms_[i] : 0.0;
        fprintf(stderr, "  %10lld %12.3f %10.2f %5s %s %10.2f  %s  %s  #%d %s\n",
                (long long)prof_calls_[i], prof_ms_[i],
                prof_ms_[i] * 1000.0 / (double)prof_calls_[i], ho ? "host" : "dev", jbuf,
                wait * 1000.0 / (double)prof_calls_[i],
                o.op.c_str(), o.base.str().c_str(), (int)o.index, wname);
        total += prof_ms_[i];
        waited += wait;
    }
    fprintf(stderr, "  %10s %12.3f   (waited %.3f ms besides)\n\n", "total", total, waited);
}

/* ================================================================== the step driver */

/* Lazy, so an architecture that never asks for a second lane runs exactly the single-lane code
 * path. */
int Ctx::lane_init() {
    if (lane1_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    RAD_TRY(rad_stream_create(&lane1_, 0));
    own_lane1_ = true;
    /* Local scope: a lane join only orders this card's two queues. */
    for (int i = 0; i < 2; ++i)
        if (!lane_ev_[i]) RAD_TRY(rad_event_create_local(&lane_ev_[i]));
    /* A new lane starts behind everything lane 0 has issued. Its first use is mid-step, where a
     * preceding rad_lane_join(0, 1) found no lane 1 and did nothing, so without this the first op
     * on the new stream could read lane 0's output before it is written. */
    RAD_TRY(rad_event_record(lane_ev_[0], stream_));
    RAD_TRY(rad_event_wait(lane1_, lane_ev_[0]));
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
    /* No lane 1 yet: nothing to order, and a later lane 1 starts behind lane 0 (lane_init). Every
     * single-lane step takes this at the end of run_step; it must not build a stream. */
    if (!lane1_) return RAD_OK;
    RadStream fs = from ? lane1_ : stream_, ts = to ? lane1_ : stream_;
    RAD_TRY(rad_event_record(lane_ev_[from], fs));
    RAD_TRY(rad_event_wait(ts, lane_ev_[from]));
    scratch_open_[from] = false;        /* `to` is now behind every scratch user `from` issued */
    return RAD_OK;
}

/* ================================================================== recorded passes */
/* See "recorded passes" in ctx.h. */

/* The key below reads every RadBatch field a pass's shape or arguments can depend on, by hand. A
 * field added to RadBatch changes its size: decide whether the key must read it, then update the
 * size here. `agen_b` and `vgen_b` are set only on generator passes, which are never recorded
 * (below). */
static_assert(sizeof(void*) != 8 || sizeof(RadBatch) == 336,
              "RadBatch changed: check that Ctx::pass_key reads every new field a pass depends on");
static_assert(sizeof(void*) != 8 || sizeof(RadKVGroupBatch) == 72,
              "RadKVGroupBatch changed: check that Ctx::pass_key reads every new field");

uint64_t Ctx::pass_key(const RadBatch* b) const {
    /* Not recordable: encoder passes are shaped by their media, and passes carrying encoder rows
     * issue the scatter only when they have some. */
    if (b->enc || b->n_mm_rows > 0) return 0;
    /* A generator pass is shaped by what RadBatch::gen_b, agen_b or vgen_b carries, which no key
     * below reads. */
    if (b->gen) return 0;
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
    /* Null on a step with no adapter row, which then issues no adapter op (Ctx::issue_lora): the
     * two are different passes. The slots themselves are read on the card. */
    ptr(b->lora_slot);
    for (int g = 0; g < b->n_kv_groups; ++g) {
        const RadKVGroupBatch& k = b->kv[g];
        mix((uint64_t)k.group);  ptr(k.slot_mapping); ptr(k.block_table);
        mix((uint64_t)k.block_table_pitch); ptr(k.seqused); ptr(k.state_index);
        mix((uint64_t)k.state_index_pitch); mix((uint64_t)k.block_size); mix((uint64_t)k.max_blocks);
    }
    mix((uint64_t)level_);
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
int Ctx::record_pass(const RadBatch* b, uint64_t key, PassTape* out) {
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
    RadTape t = tape_end(step_status_ >= 0);
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
    if (rank_ == 0 && bit >= 0 && bit < 64 && !(tape_told_ & (1ull << bit))) {
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

/* Marks which dispatches of a recorded pass may start before the ones ahead of them finish.
 *
 * One walk per queue, in order. The ops since the queue's last ordered packet are in flight
 * together; the next op may join them if it reads nothing they write and writes nothing they
 * read or write. Ordered regardless: the dispatch after an event, every dispatch after an op's
 * first (an op's dispatches depend on each other), a dispatch no op owns (e.g. a routing copy),
 * and opaque ops on either side. A run is cut at kTapeOverlapRun, the bound on which the backend
 * reclaims argument slots.
 *
 * Host work between played stretches needs nothing here: a host op joins its dependencies, and
 * anything host work puts on a queue (table patch, mover wait) makes the backend order the next
 * played packet on that queue. */
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
        /* An event packet: the next dispatch is ordered, since nothing after a wait may start
         * before its event completes. */
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
        /* Ordered: everything before it has finished; it starts a new in-flight set. */
        const int32_t own = part_of[(size_t)i];
        q->infl.clear();
        if (own >= 0) q->infl.push_back(own);
        q->opaque = own < 0 || haz_ops_[(size_t)own].opaque;
        q->run = 0;
    }
    return RAD_OK;
}

/* Compares two recordings of one key: the backend's records and the host work between them. */
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
    if (a.host.size() != b.host.size() || a.route.size() != b.route.size()) {
        *why = fmt("%zu host ops and %zu routing reports against %zu and %zu", a.host.size(),
                   a.route.size(), b.host.size(), b.route.size());
        return false;
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
    if (a.xp.size() != b.xp.size()) {
        *why = fmt("%zu prefetch calls against %zu", a.xp.size(), b.xp.size());
        return false;
    }
    for (size_t i = 0; i < a.xp.size(); ++i) {
        const TapeXp& x = a.xp[i];
        const TapeXp& y = b.xp[i];
        const RadPrefetchNext& u = x.next;
        const RadPrefetchNext& v = y.next;
        if (u.layer != v.layer || u.router != v.router || u.n_expert != v.n_expert || u.top_k != v.top_k ||
            u.n_parts != v.n_parts || u.parts != v.parts || u.count_at != v.count_at || x.h != y.h ||
            x.r0 != y.r0 || x.rows != y.rows || x.n_embd != y.n_embd) {
            *why = fmt("prefetch call %zu (for layer %d) differs", i, x.next.layer);
            return false;
        }
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
    for (const TapeStep& st : t.steps) {
        int rc = RAD_OK;
        switch (st.kind) {
        case kTapePlay:
            rc = tape_play(t.dev, st.a, st.b);
            if (rc < 0) step_fail_ = fmt("a recorded pass could not be submitted: %s",
                                         rad_dev_last_error());
            break;
        case kTapeTable: {
            WeightTable& wt = wtabs_[(size_t)st.a];
            if (wt.full || !wt.pend.empty()) {
                lane_ = st.lane;
                rc = table_refresh((rad_op)st.b, st.opd, wt);
                lane_ = l0;
            }
            break;
        }
        case kTapeHost: {
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
        case kTapeXp: {
            const TapeXp& p = t.xp[(size_t)st.a];
            lane_ = st.lane;
            rc = xp_ ? xp_predict_run(p.next, p.h, p.r0, p.rows, p.n_embd) : RAD_OK;
            lane_ = l0;
            break;
        }
        case kTapeXpMark:
            lane_ = st.lane;
            rc = xp_ ? xp_mark_run() : RAD_OK;
            lane_ = l0;
            break;
        case kTapeXpStep:
            lane_ = st.lane;
            rc = xp_ ? xp_step_run(st.a) : RAD_OK;
            lane_ = l0;
            break;
        }
        if (rc < 0) {
            if (step_status_ >= 0) step_status_ = rc;
            return rc;
        }
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

/* Runs one pass: played from its tape, or issued live (and possibly recorded or audited). */
int Ctx::run_pass(const RadBatch* b) {
    const auto bump = [](std::atomic<int64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); };
    scratch_cur_[0] = scratch_cur_[1] = 0;
    tape_path_ = kStepTapeNone;
    /* A pass the stager armed for is issued live: see runtime/stager.h. */
    if (!tape_on_ || staging_) { bump(pc_unkeyed_); arch_step_(this, b); return RAD_OK; }
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
        tape_path_ = kStepTapePlay;
        pc_dispatched_.fetch_add(t->dispatches, std::memory_order_relaxed);
        pc_unordered_.fetch_add(t->unordered, std::memory_order_relaxed);
        return play_pass(*t);
    }

    if (t) {
        bump(pc_audited_);
        tape_path_ = kStepTapeAudit;
        /* Audit: issue the pass live, recording it, and compare. Runs before the first play and
         * then every kAuditPlays plays. */
        PassTape fresh;
        if (!record_pass(b, key, &fresh)) return RAD_OK;
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

    /* Non-decode keys are recorded on their second sighting, so one-off passes (prefill chunks)
     * cost no recording and do not evict useful tapes. Decode passes recur every step until a
     * bound crosses its bucket, so they are recorded on first sight. */
    if (b->phase != RAD_PHASE_DECODE && !tape_seen(key)) {
        bump(pc_first_);
        arch_step_(this, b);
        return RAD_OK;
    }

    bump(pc_recorded_);
    tape_path_ = kStepTapeRecord;
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

    /* Before the residency sync, which carries any weights a declining stager gave back into the
     * tables (runtime/stager.h). */
    staging_ = stager_ != nullptr && stager_->begin_pass(b);

    sync_residency();

    batch_       = b;
    step_status_ = RAD_OK;
    step_fail_.clear();
    prof_n_      = 0;
    {
        const int rc = run_pass(b);
        if (rc < 0 && step_status_ >= 0) step_status_ = rc;
    }
    /* Freeze this pass's evidence before anything else runs on this thread: a draft pass shares the
     * step number and must not overwrite the trunk step's record, and the next step's ahead pass
     * would otherwise be the one a reader sees. */
    if (b->draft_pass == 0)
        last_ev_ = StepEvidence{ b->step, tape_path_, staging_ };

    /* Always join lane 1 back into lane 0 before the step ends: the engine syncs only the
     * compute stream, so work left on lane 1 could be read by the sampler or freed while in
     * flight. */
    lane_ = 0;
    {
        const int lj = lane_join(1, 0);   /* no-op when lane 1 is unbuilt */
        if (lj < 0 && step_status_ >= 0) step_status_ = lj;
    }
    /* After the join, so the compute stream is behind every reader of staged weights; regardless
     * of status, because a staged weight must never outlive its pass. */
    if (staging_) {
        const int se = stager_->end_pass(stream_);
        if (se < 0 && step_status_ >= 0) step_status_ = se;
        staging_ = false;
    }
    {
        const int rf = route_flush();
        if (rf < 0 && step_status_ >= 0) step_status_ = rf;
    }

    if (profiling_) profile_collect();
    flush_issued();
    return step_status_;
}

/* Copies OpPlan::issued into the Program for the graph dump. Walks only ops not yet seen issued
 * and compacts the list in place (no allocation). */
void Ctx::flush_issued() {
    size_t w = 0;
    for (size_t r = 0; r < unflushed_.size(); ++r) {
        const uint32_t i = unflushed_[r];
        if (i < plans_.size() && i < program_->ops.size() && plans_[i].issued)
            program_->ops[i].issued = true;
        else
            unflushed_[w++] = i;
    }
    unflushed_.resize(w);
}

}  /* namespace rad */
