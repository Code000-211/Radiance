/* mover.cpp -- the mover's mechanics. See mover.h for why any of it is shaped this way.
 *
 * Everything here runs on the DISPATCH THREAD. The transfers are asynchronous -- their own
 * stream, gated behind an event recorded after the layer's launches -- but every table edit is
 * applied by this thread at a layer boundary, draining a queue of transfers that have landed. No
 * mutex on the read side, because that side is the critical path of every layer, and no
 * background thread mutating placement mid-layer, because placement would then depend on host
 * timing and stop being reproducible.
 */
#include "mover.h"
#include "../device/device.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <queue>
#include <unordered_map>

namespace rad {

/* ------------------------------------------------------------------ the file tier */
int FileTier::open(const std::string& path) {
    /* No RAD_DIO_ALLOW_BUFFERED. The weight tier is read on the dispatch path in whole movement
     * units; a silent buffered fallback turns it into a page-cache pump that is several times
     * slower, evicts the container's own mapped pages, and looks like nothing at all in a
     * profile. An operator who pointed the tier at tmpfs wants to be told, not accommodated. */
    const int rc = f_.open(path.c_str(), RAD_DIO_READ);
    if (rc < 0)
        RAD_ERR("file tier: cannot open %s with O_DIRECT. The tier needs a filesystem that "
                "supports it -- tmpfs and some overlay mounts do not.", path.c_str());
    return rc;
}

int FileTier::read(uint64_t offset, int64_t bytes, void* dst, int64_t dst_cap) {
    if (!f_.is_open()) return RAD_E_STATE;
    if (!dst || bytes <= 0) return RAD_E_INVAL;
    /* The whole extent, rounded up: a movement unit's stride is already aligned, and reading the
     * padding costs one block against a branch on the step path. */
    const int64_t want = direct_align(bytes);
    if (want > dst_cap) {
        RAD_ERR("file tier: a %s read needs a %s aligned buffer, the staging slot is %s",
                humanb(bytes).c_str(), humanb(want).c_str(), humanb(dst_cap).c_str());
        return RAD_E_SCRATCH;
    }
    RAD_TRY(f_.read_at(dst, (int64_t)offset, want));
    ++reads_;
    bytes_ += (uint64_t)bytes;
    return RAD_OK;
}

/* ------------------------------------------------------------------ residency */
void Residency::init(const Program& prog, const Plan& plan) {
    (void)plan;
    slot_.assign(prog.weights.size(), WeightSlot{});
    noted_.assign(prog.weights.size(), 0);
    changed_.clear();
    ++epoch_;
}

/* The generation bump is the whole of the staleness protocol, and it is deliberately not a lock:
 * at this instant BOTH copies of the bytes are valid -- the mover wrote a shadow slot and has not
 * freed the old one -- so whichever side of this write a reader lands on names bytes that are
 * correct. The quarantine is what makes that sentence stay true.
 *
 * Both writers bump. `publish` settles a slot whose transfer has landed; `begin_transfer` opens
 * one, and it MUST bump too, because the run phase memoises its stream wait per generation and a
 * new event at an old generation is a wait that never happens (residency_iface.h). */
void Residency::publish(rad_weight w, void* ptr) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = nullptr;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

void Residency::begin_transfer(rad_weight w, void* ptr, RadEvent ready) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = ready;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

/* ------------------------------------------------------------------ the mover */
Mover::~Mover() {
    /* The stream first, drained. What it still holds are the gate waits and the move records
     * that name the events below and the copies into and out of the memory below, so each of
     * those goes only once nothing queued refers to it. */
    if (mover_) {
        rad_stream_sync(mover_);
        rad_stream_destroy(mover_);
    }
    for (Move& m : moves_) if (m.ev) rad_event_destroy(m.ev);
    for (RadEvent e : lead_) if (e) rad_event_destroy(e);
    for (RadEvent e : win_ev_) if (e) rad_event_destroy(e);
    if (gate_) rad_event_destroy(gate_);
    /* Only what this object allocated itself. A Pool reservation belongs to the Pool. */
    if (owns_memory_) {
        for (ClassPools& c : pools_) {
            if (c.slab_base) rad_dev_free(c.slab_base, RAD_MEM_DEVICE);
            if (c.pool_base) rad_dev_free(c.pool_base, RAD_MEM_HOST_PINNED);
        }
        if (static_vram_) rad_dev_free(static_vram_, RAD_MEM_DEVICE);
        if (stage_) rad_dev_free(stage_, RAD_MEM_HOST_PINNED);
        if (file_vram_) rad_dev_free(file_vram_, RAD_MEM_DEVICE);
        if (windows_) rad_dev_free(windows_, RAD_MEM_HOST_PINNED);
    }
    /* The pageable region is always ours: Pools has no pageable pool, because pageable host
     * memory is not a device-visible budget and nothing else contends for it. */
    if (pageable_) rad_dev_free(pageable_, RAD_MEM_HOST);
}

/* One place the mover asks for memory. Everything device-side and everything pinned comes out of
 * the budgets the operator stated (spec §6); the fallback exists so the placement test can run
 * without standing a whole Pools up, and it announces itself. */
void* Mover::reserve(const char* what, int64_t bytes, int kind) {
    if (bytes <= 0) return nullptr;
    if (budget_) {
        Pool& p = kind == RAD_MEM_DEVICE ? budget_->weights() : budget_->host_pinned();
        return p.reserve(what, bytes, RAD_ALIGN_UNIT);
    }
    owns_memory_ = true;
    return rad_dev_alloc(bytes, kind);
}

int Mover::init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget) {
    prog_ = &prog;
    plan_ = &plan;
    cfg_ = cfg;
    budget_ = budget;
    stats_ = MoverStats{};
    if (!budget_)
        RAD_WARN("mover: no Pools handed over; allocating its slab and host pool directly. That "
                 "double-spends --vram-weights-mib against whatever else already reserved it, so "
                 "it is a test path and not a deployment.");

    unit_.assign(plan.units.size(), UnitState{});
    pools_.clear();
    pools_.resize(plan.classes.size());

    /* ---- how many slots each class needs, counted before anything is allocated ------------- */
    std::vector<int64_t> n_slab(plan.classes.size(), 0), n_pool(plan.classes.size(), 0);
    int64_t static_need = 0, pageable_need = 0;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        UnitState& us = unit_[u];
        us.tier = mu.tier;
        us.site = mu.site;
        us.klass = mu.slab_class;
        us.bytes = mu.bytes;
        if (mu.slab_class < 0) continue;
        const int64_t stride = plan.classes[(size_t)mu.slab_class].stride;

        if (mu.tier == Tier::VRAM && mu.slab_slot >= 0) {
            n_slab[(size_t)mu.slab_class] = std::max(n_slab[(size_t)mu.slab_class],
                                                     (int64_t)mu.slab_slot + 1);
        } else if (mu.tier == Tier::VRAM) {
            us.static_off = align_up(static_need, RAD_ALIGN_UNIT);
            static_need = us.static_off + mu.bytes;
        } else if (mu.tier == Tier::HostPinned) {
            us.pool_slot = n_pool[(size_t)mu.slab_class]++;
        } else if (mu.tier == Tier::HostPageable) {
            us.static_off = align_up(pageable_need, RAD_ALIGN_UNIT);
            pageable_need = us.static_off + mu.bytes;
        }
        (void)stride;
    }

    /* ---- allocate ------------------------------------------------------------------------- */
    int64_t total_shadow = 0;
    for (size_t c = 0; c < plan.classes.size(); ++c) {
        const SlabClass& sc = plan.classes[c];
        ClassPools& cp = pools_[c];
        cp.stride = sc.stride;

        /* Shadow slots sit PAST the resident ones, so a slot index is a slot index and the
         * resident count is untouched by turning the engine on. Under a static plan the plan
         * asked for none and the slab is exactly the resident set. */
        /* THE PLAN'S NUMBER, not a config cap. The planner reserved these out of
         * --vram-weights-mib and the report told the operator how many; allocating a different
         * count here would mean the mover works against one number while the operator was charged
         * for another, and the direction that is wrong in is an overrun. */
        const int64_t shadow = plan.dynamic ? sc.n_shadow : 0;
        /* Shadow slots and staging slots are both "slab slots that belong to no unit", and the
         * mover takes them off one free list because at the point of use they are the same thing:
         * somewhere to copy into that nothing is reading. They are budgeted separately by the
         * planner because they are the price of two different features. */
        const int64_t slab_slots = n_slab[c] + shadow + sc.n_stage;
        /* The flex slots are not here: their memory is the KV cache's and is not carved yet.
         * lend_from() adds them once it is. */
        cp.n_fixed = slab_slots;
        cp.n_work = shadow + sc.n_stage;
        cp.slab.init(slab_slots, sc.stride);
        if (slab_slots > 0) {
            /* Tail padding: a kernel that assembles a packed code window from a wide read
             * legitimately reads a few bytes past the last slot's data. Inside the slab that
             * lands in a neighbour; at the end it runs off the allocation and faults the card. */
            cp.slab_base = reserve(sc.name.c_str(), slab_slots * sc.stride + RAD_ALIGN_UNIT,
                                   RAD_MEM_DEVICE);
            if (!cp.slab_base) {
                RAD_ERR("mover: no VRAM for the '%s' slab: %lld slots of %s",
                        sc.name.c_str(), (long long)slab_slots, humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
        }
        for (int64_t s = n_slab[c]; s < slab_slots; ++s) cp.slab.add_free(s);
        total_shadow += shadow + sc.n_stage;

        cp.pool.init(n_pool[c], sc.stride);
        if (n_pool[c] > 0) {
            cp.pool_base = reserve(sc.name.c_str(), n_pool[c] * sc.stride, RAD_MEM_HOST_PINNED);
            if (!cp.pool_base) {
                RAD_ERR("mover: no pinned host memory for the '%s' pool: %lld slots of %s",
                        sc.name.c_str(), (long long)n_pool[c], humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
            /* The pool's base as the CARD sees it. On a unified backend this is the host address
             * itself, but going through the device layer is what makes that a checked fact
             * rather than an assumption -- and without it a zero-copy expert is a bounce-buffer
             * memcpy and there is no hybrid tier at all. */
            cp.pool_dev = rad_dev_device_ptr(cp.pool_base);
            /* AND A FAILURE HERE IS NAMED RATHER THAN PAPERED OVER. Falling back silently to the
             * host pointer hands a kernel an address it cannot dereference -- so a zero-copy
             * expert would fault, or read whatever the backend happens to map there.
             *
             * It is not refused HERE because this is not where the consequence lands: a unit with
             * no pool address gets a null base from unit_base(), republish() publishes the null,
             * and the load phase already refuses by name on the first weight that needs one. That
             * is the right place -- a pool nothing zero-copies out of is not an error. This line
             * is what turns that refusal from "no address after mover init" into a cause. */
            if (!cp.pool_dev) {
                RAD_WARN("mover: the '%s' host pool did not resolve to a device address; the "
                         "zero-copy tier cannot be served out of it. In a deployment the pool is "
                         "RAD_MEM_HOST_MAPPED (core/mem/pools.cpp) and this does not fire; where "
                         "it does, the pool is ordinary host memory and a kernel handed this "
                         "address would fault.", sc.name.c_str());
                cp.pool_dev = cp.pool_base;
            }
        }
        /* The pool free list starts EMPTY: every pooled unit occupies its slot. The engine primes
         * off the shadow slab slots, runs ahead by at most that many, and settles into
         * alternating pairs -- a promotion frees a pool slot, which is what a demotion needs. */

        /* AND NOTHING HERE IS MEMSET. A fresh pool holds garbage until the load phase writes it,
         * which is correct -- a weight is read only after it is loaded -- and zeroing it would be
         * a full pass over the host tier for no information. It is also the one asymmetry between
         * the two device backends: rad_memset_async on RAD_MEM_HOST refuses on a real card
         * (pageable memory the GPU has never been shown) and quietly succeeds on the host backend,
         * so a memset added here passes in tests and fails on a card. If a host buffer ever
         * genuinely needs clearing, clear it with std::memset on the host. */
    }

    if (static_need > 0) {
        static_vram_ = reserve("static weights", static_need, RAD_MEM_DEVICE);
        if (!static_vram_) {
            /* THE ARITHMETIC, not just the refusal: the slab is reserved first and takes its
             * shadow and staging slots with it, so "no VRAM" here means the budget covered the
             * resident set and not the fixed weights beside it. An operator cannot act on the
             * sentence without the two numbers. */
            if (budget_)
                RAD_ERR("mover: the weight budget is %s and the slab already took %s of it, "
                        "leaving %s",
                        humanb(budget_->weights().bytes()).c_str(),
                        humanb(budget_->weights().used()).c_str(),
                        humanb(budget_->weights().avail()).c_str());
            RAD_ERR("mover: no VRAM for the %s of statically placed weights",
                    humanb(static_need).c_str());
            return RAD_E_NOMEM;
        }
        static_bytes_ = static_need;
    }
    if (pageable_need > 0) {
        pageable_ = rad_dev_alloc(pageable_need, RAD_MEM_HOST);
        if (!pageable_) return RAD_E_NOMEM;
        pageable_bytes_ = pageable_need;
    }

    /* ---- the file tier -------------------------------------------------------------------- */
    bool want_file = false;
    for (const MoveUnit& mu : plan.units) if (mu.tier == Tier::SSD) want_file = true;
    if (want_file) {
        if (cfg_.container_path.empty()) {
            RAD_ERR("mover: the plan put weights on the file tier but no container path was "
                    "given; a file-tier unit with nowhere to read from is an error, not a slow "
                    "path");
            return RAD_E_INVAL;
        }
        RAD_TRY(file_.open(cfg_.container_path));
        /* THE WIDEST UNIT THAT IS ACTUALLY ON THE FILE TIER, not the widest class in the plan.
         *
         * A staging slot only ever receives a unit read from the container, so sizing it off every
         * class charges the host pool for classes that are never staged. The widest class is
         * typically the embedding table -- its own class, gigabytes wide, and never on the file
         * tier: in VRAM, or in the pinned host pool by --embedding-placement -- so sizing on it
         * reserves several times that in pinned memory to receive a weight that never arrives, and
         * the file tier is then refused for want of memory it did not need. */
        int64_t widest = 0;
        for (const MoveUnit& mu : plan.units) {
            if (mu.tier != Tier::SSD || mu.slab_class < 0) continue;
            widest = std::max(widest, plan.classes[(size_t)mu.slab_class].stride);
        }
        stage_stride_ = FileTier::direct_align(widest);
        if (cfg_.stage_slots > 0 && stage_stride_ > 0) {
            stage_ = reserve("file-tier staging", (int64_t)cfg_.stage_slots * stage_stride_, RAD_MEM_HOST_PINNED);
            if (!stage_) {
                RAD_ERR("mover: no pinned memory for the file tier's %d staging slots of %s",
                        cfg_.stage_slots, humanb(stage_stride_).c_str());
                return RAD_E_NOMEM;
            }
        }
    }

    /* ---- the file stager's buffers ---------------------------------------------------------
     *
     * Exactly what the plan charged: the two VRAM buffers out of the weight budget and the read
     * windows out of the host pool's file-staging reserve. Allocated only when a routed unit is
     * actually on the file tier, which is when the plan sized them. */
    if (plan.file_stage_vram > 0) {
        if (!file_.ok()) {
            RAD_ERR("mover: the plan reads routed experts from the container and the file tier "
                    "did not open");
            return RAD_E_STATE;
        }
        file_vram_ = (char*)reserve("file-tier layer buffers", plan.file_stage_vram, RAD_MEM_DEVICE);
        if (!file_vram_) {
            RAD_ERR("mover: no VRAM for the two %s buffers a routed layer is read into from the "
                    "container", humanb(plan.file_layer_max).c_str());
            return RAD_E_NOMEM;
        }
        /* Two halves of what was charged, each at least the largest layer and aligned, so the
         * tail pad the plan added stays past the second buffer's end. */
        file_buf_bytes_ = (plan.file_stage_vram - RAD_ALIGN_UNIT) / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        if (file_buf_bytes_ < plan.file_layer_max) return RAD_E_STATE;
        if (plan.file_windows < (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) {
            RAD_ERR("mover: the plan reads routed experts from the container but reserved no read "
                    "windows for them");
            return RAD_E_STATE;
        }
        windows_ = (char*)reserve("file-tier read windows",
                                  (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES,
                                  RAD_MEM_HOST_PINNED);
        if (!windows_) {
            RAD_ERR("mover: no pinned memory for the file tier's %d read windows of %s",
                    RAD_PLACE_FILE_WINDOWS, humanb(RAD_PLACE_FILE_WINDOW_BYTES).c_str());
            return RAD_E_NOMEM;
        }
        n_windows_ = RAD_PLACE_FILE_WINDOWS;
    }

    /* ---- streams, events, move records ------------------------------------------------------ */
    RAD_TRY(rad_stream_create(&mover_, /*high_priority=*/0));
    /* EVERY EVENT HERE IS LOCAL. They order the mover's queue against this card's compute queue
     * and tell the host WHEN a slot is free; the bytes they order are read by this card's kernels
     * and never by the host through them. A copy out to host memory carries the system scope on
     * its own packets. */
    RAD_TRY(rad_event_create_local(&gate_));
    /* A window's event is waited on by the HOST, before the window is read into again. */
    win_ev_.assign((size_t)n_windows_, nullptr);
    win_rec_.assign((size_t)n_windows_, 0);
    for (RadEvent& e : win_ev_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    const int lead = std::max(1, cfg_.max_lead_dispatches);
    lead_.assign((size_t)lead + 1, nullptr);
    for (RadEvent& e : lead_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    /* TWO records per shadow slot, not one. A promotion holds a slab slot for its whole life, so
     * there can never be more promotions in flight than there are shadows -- but a swap is two
     * transfers, and the matching demotion needs a record of its own or the pair can never be in
     * flight together and the engine never issues a demotion at all. That failure is silent: it
     * shows up as `no_record` where the real symptom is a resident set that only ever grows.
     * The floor is what makes a one-shadow plan still able to hold a swap. */
    const int64_t n_records = std::max<int64_t>(2 * total_shadow, 4);
    moves_.assign((size_t)n_records, Move{});
    for (Move& m : moves_) RAD_TRY(rad_event_create_local(&m.ev));

    /* ---- publish the initial addresses ------------------------------------------------------ */
    res_.init(prog, plan);
    for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
        unit_[(size_t)u].slab_slot = plan.units[(size_t)u].slab_slot;
        republish(u);
    }

    RAD_INFO("mover: %lld slab slots (%lld shadow), %lld pinned pool slots, %s static vram%s",
             (long long)[&] { int64_t n = 0; for (auto& c : pools_) n += c.slab.size(); return n; }(),
             (long long)total_shadow,
             (long long)[&] { int64_t n = 0; for (auto& c : pools_) n += c.pool.size(); return n; }(),
             humanb(static_bytes_).c_str(), file_.ok() ? ", file tier open" : "");
    /* The pageable host region is allocated here and charged to no budget -- Pools has no
     * pageable pool because the memory is not device-visible and nothing else contends for it.
     * That makes the receipt the only place it is ever named. */
    if (pageable_bytes_ > 0)
        RAD_INFO("mover: %s of pageable host memory for the units no tier could hold",
                 humanb(pageable_bytes_).c_str());
    return RAD_OK;
}

int Mover::bind_compute_stream(RadStream s) {
    if (!s) return RAD_E_INVAL;
    compute_ = s;
    return RAD_OK;
}

void* Mover::unit_base(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return nullptr;
    const UnitState& us = unit_[(size_t)unit];
    if (us.klass < 0) return nullptr;
    const ClassPools& cp = pools_[(size_t)us.klass];
    if (us.slab_slot >= 0 && (cp.is_flex(us.slab_slot) || cp.slab_base))
        return cp.slot_ptr(us.slab_slot);
    if (us.tier == Tier::VRAM && us.static_off >= 0 && static_vram_)
        return (char*)static_vram_ + us.static_off;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0 && cp.pool_dev)
        return (char*)cp.pool_dev + us.pool_slot * cp.stride;
    if (us.tier == Tier::HostPageable && us.static_off >= 0 && pageable_)
        return (char*)pageable_ + us.static_off;
    /* A file-tier unit that is not staged has NO address, and saying so is the point: a kernel
     * cannot read one off disk, so a caller that meets a null here must stage it or refuse. */
    return nullptr;
}

void Mover::republish(int32_t unit) {
    const MoveUnit& mu = plan_->units[(size_t)unit];
    char* base = (char*)unit_base(unit);
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.publish(mu.weights[i], base ? base + mu.offset[i] : nullptr);
}

/* Is a COPY INTO this unit's slot still in flight? The release path needs it: a slot whose
 * staging copy has not landed cannot be given back, because the quarantine barrier is recorded on
 * the COMPUTE stream and would say nothing about an H2D still in flight on the mover's.
 *
 * Copying and only Copying. "Any record that is not Free" is the same question asked too widely:
 * a record past Copying has its bytes in place, and treating it as in-flight holds the slot for
 * as long as the record takes to recycle rather than as long as the copy takes to land. */
bool Mover::unit_in_flight(int32_t unit) const {
    for (const Move& m : moves_)
        if (m.state == Move::State::Copying && m.unit == unit) return true;
    return false;
}

int Mover::free_move_record() const {
    const size_t n = moves_.size();
    for (size_t k = 0; k < n; ++k) {
        const size_t i = (rec_cursor_ + k) % n;
        if (moves_[i].state == Move::State::Free) { rec_cursor_ = (i + 1) % n; return (int)i; }
    }
    return -1;
}

/* The ordering event, recorded ONCE PER DISPATCH and only when a move is actually going out.
 * Everything the mover issues this dispatch waits on it, and it is downstream of every launch
 * made up to now. Once per dispatch rather than once per move: re-recording per move is a dozen
 * event records a layer restating one ordering constraint. */
int Mover::record_gate() {
    if (!compute_) {
        RAD_ERR("mover: no compute stream bound; a transfer cannot be ordered against the "
                "kernels that read the slot it targets");
        return RAD_E_STATE;
    }
    if (gate_dispatch_ == dispatch_) return RAD_OK;
    RAD_TRY(rad_event_record(gate_, compute_));
    gate_dispatch_ = dispatch_;
    return RAD_OK;
}

int Mover::begin_dispatch(int dispatch, HeatEngine* heat) {
    dispatch_ = dispatch;
    ++stats_.dispatches;
    RAD_TRY(retire(heat));

    /* THE LEAD GATE. This is a host wait and it is NOT a transfer synchronisation: no transfer
     * waits on the host and no host thread waits on a transfer. What it bounds is how far ahead
     * of the cards this thread may queue work, and it exists because a freed slot returns to the
     * free list only when its quarantine event signals -- and that event is on the compute
     * stream. Let the host run and the stream carries a whole step, the recycle latency becomes
     * that depth, and the mover finds no free slot to move into on almost every round. */
    if (compute_ && cfg_.max_lead_dispatches > 0) {
        RadEvent old = lead_[(size_t)lead_next_];
        if (stats_.dispatches > (uint64_t)lead_.size()) RAD_TRY(rad_event_sync(old));
        RAD_TRY(rad_event_record(old, compute_));
        lead_next_ = (lead_next_ + 1) % (int)lead_.size();
    }
    return RAD_OK;
}

/* ---- the exact prefetch, which is the whole of layer offload's policy -------------------------
 *
 * Declare enumerated every op with its weight operands in order, so the deadline for each staged
 * unit is a fact and not a guess. WHEN to start is the supply question -- start as early as a
 * slot is free, and stop at the first unit that cannot get one rather than skipping ahead,
 * because the schedule is in deadline order and serving a later unit before an earlier one is how
 * a prefetch turns into a cache.
 */
int Mover::prefetch_to_op(int32_t op_index) {
    if (!plan_) return RAD_E_STATE;

    /* A backwards op index is the next step starting. The schedule is per step, so the cursor
     * rewinds; nothing else has to know a step boundary happened. */
    if (op_index < last_op_) prefetch_cursor_ = 0;
    last_op_ = op_index;

    /* RELEASE FIRST, so the slots this op finished with are available to the units it is about to
     * need. A staged unit's slot goes back once its last_use_op is behind us: the declared order
     * is what makes that safe, because a weight read outside [first_use_op, last_use_op] would
     * contradict the order declare recorded. */
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.site != Site::DeviceStaged || us.slab_slot < 0) continue;
        if (unit_in_flight(u)) continue;   /* its copy has not landed; the slot is not ours yet */
        const MoveUnit& mu = plan_->units[(size_t)u];
        if (mu.last_use_op >= 0 && mu.last_use_op < op_index) release_slot(u);
    }

    while (prefetch_cursor_ < (int32_t)plan_->prefetch.size()) {
        const Plan::Prefetch& pf = plan_->prefetch[(size_t)prefetch_cursor_];
        UnitState& us = unit_[(size_t)pf.unit];
        if (us.slab_slot >= 0) { ++prefetch_cursor_; continue; }   /* already in a slot */
        if (us.klass < 0) { ++prefetch_cursor_; continue; }

        const int64_t slot = pools_[(size_t)us.klass].slab.take();
        if (slot < 0) { ++stats_.starved_slab; break; }
        const int rc = start_stage(pf.unit, slot);
        if (rc < 0) {
            pools_[(size_t)us.klass].slab.add_free(slot);
            return rc;
        }
        ++prefetch_cursor_;
    }
    return RAD_OK;
}

int Mover::start_stage(int32_t unit, int64_t slab_slot) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());

    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    char* dst = cp.slot_ptr(slab_slot);

    const void* src = nullptr;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0)
        src = (char*)cp.pool_base + us.pool_slot * cp.stride;
    else if (us.tier == Tier::HostPageable && us.static_off >= 0)
        src = (char*)pageable_ + us.static_off;
    else if (us.tier == Tier::SSD) {
        /* The file tier's cliff, and it is a BLOCKING host read in the middle of a dispatch. It
         * is counted so a run that leans on it is visible as a run to re-budget, not as a mystery
         * in the throughput. */
        if (!file_.ok() || !stage_) return RAD_E_STATE;
        char* stage = (char*)stage_ + (int64_t)stage_next_ * stage_stride_;
        stage_next_ = (stage_next_ + 1) % std::max(1, cfg_.stage_slots);
        /* WEIGHT BY WEIGHT, each from its own plane into its own place in the unit. A container
         * entry is a logical weight's planes at 256-byte alignment and a unit lays its weights out
         * at their declared alignment, so the two arrangements are not the same bytes and one
         * read of the unit's span would put every weight after the first at the wrong offset. */
        const MoveUnit& mu = plan_->units[(size_t)unit];
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            RAD_TRY(file_.read(w.file_offset, w.stored_bytes, stage + mu.offset[k],
                               stage_stride_ - mu.offset[k]));
        }
        ++stats_.ssd_reads;
        stats_.ssd_bytes += (uint64_t)us.bytes;
        src = stage;
    }
    if (!src) return RAD_E_STATE;

    RAD_TRY(rad_event_wait(mover_, gate_));
    RAD_TRY(rad_memcpy_async(dst, src, us.bytes, mover_));
    RAD_TRY(rad_event_record(moves_[(size_t)rec].ev, mover_));

    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Copying;
    m.kind = Move::Kind::Stage;
    m.unit = unit;
    m.slab_slot = slab_slot;
    m.pool_slot = -1;
    stats_.h2d_bytes += (uint64_t)us.bytes;
    stats_.prefetch_bytes += (uint64_t)us.bytes;
    ++stats_.prefetched;

    /* PUBLISH THE DESTINATION NOW, not at retire, and with the event. A staging copy is the one
     * move whose destination the run phase will read: the old address (the pinned pool) stays
     * valid for everything already issued, and everything issued from here on names the slab slot
     * and is ordered behind `ev` on the compute stream. That stream wait is the only thing
     * standing between a staged weight and a race, and it costs no host time.
     *
     * begin_transfer bumps the generation, which is what makes the run phase actually perform
     * that wait rather than skipping it as already done. */
    us.slab_slot = slab_slot;
    char* base = (char*)unit_base(unit);
    const MoveUnit& mu = plan_->units[(size_t)unit];
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.begin_transfer(mu.weights[i], base + mu.offset[i], m.ev);
    return RAD_OK;
}

bool Mover::is_unit_resident(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return false;
    return unit_[(size_t)unit].tier == Tier::VRAM && unit_[(size_t)unit].slab_slot >= 0;
}

int32_t Mover::unit_klass(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return -1;
    return unit_[(size_t)unit].klass;
}

int64_t Mover::unit_bytes(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return 0;
    return unit_[(size_t)unit].bytes;
}

bool Mover::has_free_slab_slot(size_t klass) const {
    if (klass >= pools_.size()) return false;
    return pools_[klass].slab.n_free() > 0;
}

int64_t Mover::free_slab_slots(size_t klass) const {
    if (klass >= pools_.size()) return 0;
    return pools_[klass].slab.n_free();
}

int Mover::prefetch_unit_async(int32_t unit, int32_t victim, HeatEngine* heat, int lease_dispatches) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;
    if (heat && heat->busy(unit)) return RAD_E_STATE;
    if (unit < 0 || (size_t)unit >= unit_.size()) return RAD_E_INVAL;
    UnitState& us = unit_[(size_t)unit];
    if (us.tier == Tier::VRAM && us.slab_slot >= 0) return RAD_OK;
    if (us.klass < 0 || us.pool_slot < 0) return RAD_E_STATE;
    ClassPools& cp = pools_[(size_t)us.klass];

    int64_t slot = -1;
    if (victim >= 0) {
        if ((size_t)victim >= unit_.size()) return RAD_E_INVAL;
        UnitState& vs = unit_[(size_t)victim];
        if (vs.slab_slot < 0 || vs.klass != us.klass) return RAD_E_INVAL;

        int64_t pool_slot = vs.pool_slot;
        if (pool_slot < 0) {
            pool_slot = cp.pool.take();
            if (pool_slot < 0) {
                ++stats_.starved_pool;
                return RAD_E_FULL;
            }
            char* dst = (char*)cp.pool_base + pool_slot * cp.stride;
            const void* src = cp.slot_ptr(vs.slab_slot);
            if (rad_memcpy_async(dst, src, vs.bytes, mover_) < 0) {
                cp.pool.add_free(pool_slot);
                return RAD_E_DEVICE;
            }
            stats_.d2h_bytes += (uint64_t)vs.bytes;
        }

        slot = vs.slab_slot;
        vs.slab_slot = -1;
        vs.pool_slot = pool_slot;
        vs.tier = Tier::HostPinned;
        vs.site = Site::DeviceZeroCopy;
        republish(victim);
        if (heat) heat->set_tier(victim, Tier::HostPinned);
        ++stats_.demotions;
        ++stats_.dropped;
    } else {
        slot = cp.slab.take();
        if (slot < 0) {
            ++stats_.starved_slab;
            return RAD_E_FULL;
        }
    }

    if (slot < 0) return RAD_E_FULL;
    int rc = promote_to(unit, slot);
    if (rc < 0) {
        cp.slab.add_free(slot);
        return rc;
    }
    if (heat) {
        const int lease = (lease_dispatches > 0) ? lease_dispatches : heat->min_residency_dispatches();
        heat->register_speculative_prefetch(unit, lease);
    }
    ++stats_.prefetched;
    stats_.prefetch_bytes += (uint64_t)us.bytes;
    return RAD_OK;
}

/* ---- staging within a pass (see mover.h) ---------------------------------------------------- */
int Mover::stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                       RadEvent done, std::vector<int32_t>* staged) {
    if (!plan_ || !dst || !staged || !done) return RAD_E_INVAL;
    staged->clear();
    std::vector<int64_t> at;
    at.reserve(units.size());
    int64_t off = 0;
    bool waited = false;
    const char* pending_src = nullptr;
    char* pending_dst = nullptr;
    int64_t pending_bytes = 0;
    int32_t pending_klass = -1;
    size_t queued = 0;
    auto flush = [&]() -> int {
        if (!pending_src) return RAD_OK;
        const int rc = rad_memcpy_async(pending_dst, pending_src, pending_bytes, mover_);
        if (rc < 0) {
            staged->resize(queued);
            return rc;
        }
        stats_.h2d_bytes += (uint64_t)pending_bytes;
        stats_.staged_bytes += (uint64_t)pending_bytes;
        queued = staged->size();
        pending_src = nullptr;
        pending_bytes = 0;
        return RAD_OK;
    };
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) {
            RAD_TRY(flush());
            return RAD_E_INVAL;
        }
        const UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::HostPinned || us.pool_slot < 0 || us.slab_slot >= 0 || us.klass < 0)
            continue;
        if (unit_in_flight(u)) continue;
        const int64_t need = align_up(us.bytes, RAD_ALIGN_UNIT);
        if (off + need > cap) { ++stats_.stage_overflow; continue; }
        if (!waited) {
            if (after) RAD_TRY(rad_event_wait(mover_, after));
            waited = true;
        }
        const ClassPools& cp = pools_[(size_t)us.klass];
        const char* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
        /* Both ranges must touch without padding, and stay in one pool allocation: adjacent
         * virtual addresses in different classes need not share a device registration. */
        if (pending_src &&
            (us.klass != pending_klass || src != pending_src + pending_bytes ||
             dst + off != pending_dst + pending_bytes))
            RAD_TRY(flush());
        if (!pending_src) {
            pending_src = src;
            pending_dst = dst + off;
            pending_klass = us.klass;
        }
        pending_bytes += us.bytes;
        staged->push_back(u);
        at.push_back(off);
        off += need;
    }
    RAD_TRY(flush());
    if (staged->empty()) return RAD_OK;
    RAD_TRY(rad_event_record(done, mover_));
    for (size_t k = 0; k < staged->size(); ++k) {
        const MoveUnit& mu = plan_->units[(size_t)(*staged)[k]];
        for (size_t i = 0; i < mu.weights.size(); ++i)
            res_.publish(mu.weights[i], dst + at[k] + mu.offset[i]);
    }
    ++stats_.stage_layers;
    stats_.staged_units += (uint64_t)staged->size();
    return RAD_OK;
}

void Mover::unstage_units(const std::vector<int32_t>& staged) {
    for (int32_t u : staged) republish(u);
}

/* ---- a routed layer read from the container (see mover.h) ---------------------------------- */
namespace {

/* One weight's bytes: where they are in the container and where they go. */
struct FileExt {
    int64_t foff = 0, bytes = 0;
    char*   dst = nullptr;
};

/* THE GAP A RUN READS THROUGH rather than starting another pread. The weights of one routed
 * layer sit together in the container, each starting on RAD_ALIGN_UNIT, so between two that are
 * both on the file tier there is either alignment padding or a run of units that are not -- a
 * resident expert or a pooled one. Reading padding is free next to a second pread; reading a
 * resident expert is a megabyte or ten the drive delivers for nothing. 256 KiB is above every
 * padding and below every expert this engine serves. */
constexpr int64_t kReadThrough = (int64_t)256 << 10;

}  /* namespace */

int Mover::stage_file_units(const std::vector<int32_t>& units, char* dst, int64_t cap,
                            RadEvent after, RadEvent done, std::vector<int32_t>* staged) {
    if (!plan_ || !dst || !staged || !done) return RAD_E_INVAL;
    staged->clear();
    if (!file_.ok() || !windows_ || n_windows_ <= 0) return RAD_E_STATE;

    /* The layout in `dst`, unit by unit in the order given, and every weight's extent. */
    std::vector<FileExt> ext;
    std::vector<int64_t> at;
    int64_t off = 0;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) return RAD_E_INVAL;
        const UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) continue;
        const int64_t need = align_up(us.bytes, RAD_ALIGN_UNIT);
        if (off + need > cap) {
            RAD_ERR("file stager: a routed layer's file-tier units need more than the %s buffer "
                    "the plan sized for the largest one", humanb(cap).c_str());
            return RAD_E_SCRATCH;
        }
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            if (w.stored_bytes <= 0) continue;          /* nothing to read, and nothing to place */
            ext.push_back(FileExt{ (int64_t)w.file_offset, w.stored_bytes, dst + off + mu.offset[k] });
        }
        staged->push_back(u);
        at.push_back(off);
        off += need;
    }
    if (staged->empty()) return RAD_OK;
    /* CONTAINER ORDER, so a window is one long run of the drive and not a seek per weight. */
    std::stable_sort(ext.begin(), ext.end(),
                     [](const FileExt& a, const FileExt& b) { return a.foff < b.foff; });

    if (after) RAD_TRY(rad_event_wait(mover_, after));

    const int64_t A = DirectFile::alignment();
    const int64_t W = RAD_PLACE_FILE_WINDOW_BYTES;
    struct Seg { int64_t fs = 0, fe = 0, ws = 0; };     /* file [fs, fe) read to window offset ws */
    struct Cp  { int64_t wo = 0, bytes = 0; char* dst = nullptr; };
    std::vector<Seg> segs;
    std::vector<Cp>  cps;
    size_t  i = 0;
    int64_t piece = 0;                                  /* bytes of ext[i] already placed */
    while (i < ext.size()) {
        const int w = win_next_;
        win_next_ = (win_next_ + 1) % n_windows_;
        char* win = windows_ + (int64_t)w * W;
        if (win_rec_[(size_t)w]) {
            /* The window's copies from the round before must have left it. */
            const auto t0 = std::chrono::steady_clock::now();
            RAD_TRY(rad_event_sync(win_ev_[(size_t)w]));
            stats_.file_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count();
            win_rec_[(size_t)w] = 0;
        }

        /* Pack the window: extend the current run while the next weight starts within the
         * read-through gap of its end, start another run otherwise, and split a weight the window
         * cannot finish across this window and the next. */
        segs.clear();
        cps.clear();
        while (i < ext.size()) {
            const FileExt& e = ext[i];
            const int64_t fo = e.foff + piece, left = e.bytes - piece;
            int64_t wo;
            if (!segs.empty() && fo >= segs.back().fs && fo <= segs.back().fe + kReadThrough) {
                wo = segs.back().ws + (fo - segs.back().fs);
            } else {
                const int64_t ws = segs.empty()
                    ? 0 : segs.back().ws + align_up(segs.back().fe - segs.back().fs, A);
                const int64_t fs = fo / A * A;
                wo = ws + (fo - fs);
                if (wo >= W) break;
                segs.push_back(Seg{ fs, fs, ws });
            }
            const int64_t take = std::min(left, W - wo);
            if (take <= 0) break;
            Seg& s = segs.back();
            s.fe = std::max(s.fe, fo + take);
            cps.push_back(Cp{ wo, take, e.dst + piece });
            piece += take;
            if (piece == e.bytes) { ++i; piece = 0; }
            if (take < left) break;                     /* the window is full */
        }
        if (cps.empty()) return RAD_E_STATE;            /* a window that takes nothing cannot end */

        const auto t0 = std::chrono::steady_clock::now();
        for (const Seg& s : segs) {
            RAD_TRY(file_.read((uint64_t)s.fs, s.fe - s.fs, win + s.ws, W - s.ws));
            ++stats_.file_preads;
        }
        stats_.file_read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0).count();
        for (const Cp& c : cps) RAD_TRY(rad_memcpy_async(c.dst, win + c.wo, c.bytes, mover_));
        RAD_TRY(rad_event_record(win_ev_[(size_t)w], mover_));
        win_rec_[(size_t)w] = 1;
    }
    RAD_TRY(rad_event_record(done, mover_));

    for (size_t k = 0; k < staged->size(); ++k) {
        const int32_t u = (*staged)[k];
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t j = 0; j < mu.weights.size(); ++j)
            res_.publish(mu.weights[j], dst + at[k] + mu.offset[j]);
        stats_.file_bytes += (uint64_t)unit_[(size_t)u].bytes;
    }
    ++stats_.file_layers;
    stats_.file_units += (uint64_t)staged->size();
    stats_.h2d_bytes += (uint64_t)off;
    return RAD_OK;
}

void Mover::release_slot(int32_t unit) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;    /* nothing to copy; it is already "landed" */
    m.kind = Move::Kind::Release;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    republish(unit);                     /* the pointer stops naming the slot before the barrier */
}

int Mover::issue(const Swap& s, HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    auto one = [&](int32_t unit, bool promote) -> bool {
        if (unit < 0) return false;
        UnitState& us = unit_[(size_t)unit];
        if (us.klass < 0) return false;
        ClassPools& cp = pools_[(size_t)us.klass];

        /* A UNIT THAT KEPT ITS HOST COPY IS DROPPED, NOT DEMOTED: its pool slot still holds its
         * bytes, so there is nothing to copy and no second pool slot to take. */
        if (!promote && us.pool_slot >= 0 && us.slab_slot >= 0) return drop(unit, heat) >= 0;

        if (promote) {
            const int64_t slot = cp.slab.take();
            if (slot < 0) { ++stats_.starved_slab; return false; }
            if (promote_to(unit, slot) < 0) { cp.slab.add_free(slot); return false; }
            return true;
        }

        const int rec = free_move_record();
        if (rec < 0) { ++stats_.no_record; return false; }
        if (record_gate() < 0) return false;
        Move& m = moves_[(size_t)rec];
        {
            const int64_t g = cp.pool.take();
            if (g < 0) { ++stats_.starved_pool; return false; }
            char* dst = (char*)cp.pool_base + g * cp.stride;
            if (us.slab_slot < 0) { cp.pool.add_free(g); return false; }
            const void* src = cp.slot_ptr(us.slab_slot);
            /* D2H out of the slab. The slab layout IS the pool layout, so this is a straight copy
             * with no transform -- and IT IS NOT FREE. It is tempting to read a demotion as
             * running in the direction the zero-copy tier leaves idle and therefore as nearly
             * free; it is not. The two costs are additive: the directions share one budget and a
             * demotion costs about what a promotion does. That is the exchange rate heat.h's
             * min_gain is tuned against. */
            if (rad_event_wait(mover_, gate_) < 0) { cp.pool.add_free(g); return false; }
            if (rad_memcpy_async(dst, src, us.bytes, mover_) < 0) {
                cp.pool.add_free(g);
                return false;
            }
            (void)rad_event_record(m.ev, mover_);
            m.kind = Move::Kind::Demote;
            m.slab_slot = us.slab_slot;
            m.pool_slot = g;
            stats_.d2h_bytes += (uint64_t)us.bytes;
        }
        m.state = Move::State::Copying;
        m.unit = unit;
        return true;
    };

    const bool up = one(s.promote, true);
    const bool down = one(s.demote, false);
    /* A half-issued swap is legal and self-correcting: the two halves create each other's supply,
     * so the missing one goes out on a later dispatch. What is NOT legal is leaving a unit busy
     * that no move will ever publish -- the heat engine would never consider it again and would
     * go quiet with no counter saying why. */
    if (heat) {
        if (!up) heat->set_busy(s.promote, false);
        if (!down) heat->set_busy(s.demote, false);
    }
    return (up || down) ? RAD_OK : RAD_E_FULL;
}

/* ------------------------------------------------------------------ the loan */
static char* align_down_ptr(char* p) {
    const uintptr_t v = (uintptr_t)p;
    return (char*)(v - v % (uintptr_t)RAD_ALIGN_UNIT);
}

int Mover::lend_from(int lender, const std::vector<LoanStrip>& strips) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;      /* a static plan never moves a unit */
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender) {
                RAD_ERR("mover: lend_from called twice for one lender; its slots are carved once");
                return RAD_E_STATE;
            }

    /* WHO GETS THE SLOTS. A class with units in the pinned pool that could be promoted, in
     * proportion to their bytes -- and never more slots than it has such units, because a slot no
     * unit can ever fill is VRAM lent to nothing. */
    const size_t nc = pools_.size();
    std::vector<int64_t> want(nc, 0), got(nc, 0);
    for (size_t u = 0; u < unit_.size(); ++u) {
        const UnitState& us = unit_[u];
        if (us.klass < 0 || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[u].movable) continue;
        ++want[(size_t)us.klass];
    }
    std::vector<double> weight(nc, 0.0);
    for (size_t c = 0; c < nc; ++c)
        weight[c] = pools_[c].stride > 0 ? (double)want[c] * (double)pools_[c].stride : 0.0;

    /* EVERY STRETCH, TOP DOWN. The classes take turns in proportion to their weight, so each
     * stretch holds each class's share and the share does not depend on the order the stretches
     * arrive in. */
    struct Cut { int64_t need; char* p; };
    std::vector<std::vector<Cut>> cuts(nc);
    for (const LoanStrip& st : strips) {
        if (!st.top || st.bytes <= (int64_t)RAD_ALIGN_UNIT || st.unit_bytes <= 0 ||
            st.unit_tokens <= 0)
            continue;
        char* const bottom = st.top - st.bytes;
        /* THE TOP ALIGNMENT UNIT STAYS EMPTY. A kernel assembling a packed code window reads a
         * few bytes past the end of the slot it was handed; inside the stretch that lands in the
         * slot above, and above the top of the stretch is memory that is not the slab's -- the
         * next layer's blocks, or at the end of the carve a region nothing may have mapped. */
        char* cur = align_down_ptr(st.top - RAD_ALIGN_UNIT);
        for (;;) {
            int best = -1;
            double best_k = 0.0;
            for (size_t c = 0; c < nc; ++c) {
                if (weight[c] <= 0.0 || got[c] >= want[c]) continue;
                if (cur - bottom < pools_[c].stride) continue;
                if (align_down_ptr(cur - pools_[c].stride) < bottom) continue;
                const double k = (double)(got[c] + 1) * (double)pools_[c].stride / weight[c];
                if (best < 0 || k < best_k) { best = (int)c; best_k = k; }
            }
            if (best < 0) break;
            char* p = align_down_ptr(cur - pools_[(size_t)best].stride);
            /* The loan that covers this slot reaches its lowest byte. */
            const int64_t depth = st.top - p;
            const int64_t need = (depth + st.unit_bytes - 1) / st.unit_bytes * st.unit_tokens;
            cuts[(size_t)best].push_back(Cut{ need, p });
            ++got[(size_t)best];
            cur = p;
        }
    }

    /* SMALLEST LOAN FIRST, so a loan of any size is a prefix of the slots and the one it gives
     * back first is the deepest -- which is the one the owner wants back first. */
    int64_t n_new = 0, bytes_new = 0;
    for (size_t c = 0; c < nc; ++c) {
        if (cuts[c].empty()) continue;
        ClassPools& cp = pools_[c];
        std::stable_sort(cuts[c].begin(), cuts[c].end(),
                         [](const Cut& x, const Cut& y) { return x.need < y.need; });
        ClassPools::Lent l;
        l.lender = lender;
        l.seg = cp.slab.add_segment((int64_t)cuts[c].size());
        l.first = cp.slab.seg_lo(l.seg);
        /* THE SLOT NUMBER AND THE POINTER LIST AGREE BY CONSTRUCTION: segments are appended in the
         * order their pointers are, so slot_ptr's `slot - n_fixed` indexes straight into them. */
        if (l.first != cp.n_fixed + (int64_t)cp.flex_ptr.size()) {
            RAD_ERR("mover: a lent segment starts at slot %lld with %zu lent slots before it",
                    (long long)l.first, cp.flex_ptr.size());
            return RAD_E_STATE;
        }
        for (const Cut& k : cuts[c]) { cp.flex_ptr.push_back(k.p); l.need.push_back(k.need); }
        cp.lent.push_back(std::move(l));
        n_new += (int64_t)cuts[c].size();
        bytes_new += (int64_t)cuts[c].size() * cp.stride;
    }
    if (n_new <= 0) return RAD_OK;

    /* A RECORD FOR EVERY SLOT THAT CAN BE LENT. A fill issues a promotion into every free slot
     * at once and a recall drops every unit past the loan at once, and each of those is a record
     * until its barrier clears; sized at the shadow count, the records -- not the link -- would
     * decide how fast a loan is used. */
    const size_t had = moves_.size();
    moves_.resize(had + (size_t)n_new);
    for (size_t i = had; i < moves_.size(); ++i) RAD_TRY(rad_event_create_local(&moves_[i].ev));

    RAD_INFO("mover: %lld slots can be lent out of the %s (%s), filled as it lends them",
             (long long)n_new, lender == kLendKV ? "KV cache" : "activation arena",
             humanb(bytes_new).c_str());
    count_flex();
    return RAD_OK;
}

int Mover::set_loan(int lender, int64_t amount) {
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    loan_[lender] = amount > 0 ? amount : 0;
    for (ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t n = std::upper_bound(l.need.begin(), l.need.end(), loan_[lender]) -
                              l.need.begin();
            if (cp.slab.target(l.seg) != l.first + n) {
                if (l.first + n < cp.slab.target(l.seg)) cp.strand_check = true;
                cp.slab.set_target(l.seg, l.first + n);
            }
        }
    count_flex();
    return RAD_OK;
}

int64_t Mover::loan_held(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t top = cp.slab.top_held(l.seg);
            if (top >= l.first) t = std::max(t, l.need[(size_t)(top - l.first)]);
        }
    return t;
}

int64_t Mover::flex_capacity() const {
    int64_t n = 0;
    for (const ClassPools& cp : pools_) n += (int64_t)cp.flex_ptr.size() * cp.stride;
    return n;
}

int64_t Mover::loan_ceiling(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender && !l.need.empty()) t = std::max(t, l.need.back());
    return t;
}

void Mover::count_flex() {
    int64_t used = 0, offered = 0;
    for (const ClassPools& cp : pools_) {
        if (cp.flex_ptr.empty()) continue;
        used += cp.slab.held_from(cp.n_fixed) * cp.stride;
        for (const ClassPools::Lent& l : cp.lent)
            offered += (cp.slab.target(l.seg) - l.first) * cp.stride;
    }
    flex_used_.store(used, std::memory_order_relaxed);
    flex_offered_.store(offered, std::memory_order_relaxed);
}

int Mover::drop(int32_t unit, HeatEngine* heat) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;       /* nothing to copy: the host copy never went away */
    m.kind = Move::Kind::Drop;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    us.tier = Tier::HostPinned;
    us.site = Site::DeviceZeroCopy;
    republish(unit);                        /* the pointer stops naming the slot before the barrier */
    ++stats_.demotions;
    ++stats_.dropped;
    if (heat) {
        heat->set_tier(unit, Tier::HostPinned);
        heat->set_busy(unit, true);
    }
    return RAD_OK;
}

int Mover::promote_to(int32_t unit, int64_t slot) {
    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());
    if (us.pool_slot < 0) return RAD_E_STATE;
    const void* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
    Move& m = moves_[(size_t)rec];
    RAD_TRY(rad_event_wait(mover_, gate_));
    RAD_TRY(rad_memcpy_async(cp.slot_ptr(slot), src, us.bytes, mover_));
    (void)rad_event_record(m.ev, mover_);
    m.kind = Move::Kind::Promote;
    m.slab_slot = slot;
    /* INTO A LENT SLOT, THE HOST COPY STAYS. The loan can be recalled at any time and a unit that
     * kept its pool slot goes back by a change of address; one that gave it up would need a pool
     * slot and a copy across the link at the moment the cache is asking for its memory. The
     * quarantine then has no pool slot to free. */
    m.pool_slot = cp.is_flex(slot) ? -1 : us.pool_slot;
    m.state = Move::State::Copying;
    m.unit = unit;
    stats_.h2d_bytes += (uint64_t)us.bytes;
    return RAD_OK;
}

int Mover::fill(size_t klass, int64_t room, HeatEngine* heat) {
    if (room <= 0) return RAD_OK;
    struct Cand { float h; int32_t layer; int32_t unit; };
    std::vector<Cand> cand;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.klass != (int32_t)klass || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[(size_t)u].movable) continue;
        if (heat && heat->busy(u)) continue;
        cand.push_back(Cand{ heat ? heat->heat(u) : 0.0f, plan_->units[(size_t)u].layer, u });
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.h > b.h; });

    /* HOTTEST FIRST, AND ON AN EXACT TIE THE LAYER HOLDING THE FEWEST RESIDENT UNITS -- the
     * statement `pick` makes, and it has to keep holding as the batch lands. Every promotion here
     * is issued before any of them retires, so the heat engine's own count does not move during
     * the batch; it is counted here instead. Without that a fill of a cold slab, where every heat
     * is zero, would put its whole batch into whichever layer was thinnest when it began. */
    std::unordered_map<int32_t, int32_t> added;
    int64_t issued = 0;
    for (size_t i = 0; i < cand.size() && issued < room;) {
        size_t j = i;
        while (j < cand.size() && cand[j].h == cand[i].h) ++j;
        std::map<int32_t, std::vector<int32_t>> by_layer;            /* one heat, per layer */
        for (size_t k = j; k-- > i;) by_layer[cand[k].layer].push_back(cand[k].unit);
        using Key = std::pair<int32_t, int32_t>;                      /* (residents, layer) */
        std::priority_queue<Key, std::vector<Key>, std::greater<Key>> q;
        for (auto& kv : by_layer) {
            const int32_t rc = heat ? heat->layer_residents(kv.second.back()) : 0;
            q.push({ rc + added[kv.first], kv.first });
        }
        while (!q.empty() && issued < room) {
            const Key k = q.top();
            q.pop();
            std::vector<int32_t>& us = by_layer[k.second];
            const int32_t u = us.back();
            us.pop_back();
            /* A LENT SLOT AND NEVER A WORKING ONE. The shadow and staging slots are what lets a
             * swap happen at all, and a unit filled into one sits in the budgeted range where no
             * loan can give it back -- the reserve would be gone for good the first time the loan
             * shrank. */
            const int64_t slot = pools_[klass].slab.take_at_least(pools_[klass].n_fixed);
            if (slot < 0) return RAD_OK;
            if (heat) heat->set_busy(u, true);
            if (promote_to(u, slot) < 0) {                   /* no record left: next time */
                pools_[klass].slab.add_free(slot);
                if (heat) heat->set_busy(u, false);
                return RAD_OK;
            }
            ++issued;
            ++added[k.second];
            if (!us.empty()) q.push({ k.first + 1, k.second });
        }
        i = j;
    }
    return RAD_OK;
}

int Mover::balance(HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    for (size_t c = 0; c < pools_.size(); ++c) {
        ClassPools& cp = pools_[c];
        if (cp.flex_ptr.empty()) continue;
        bool drained = false;

        /* ---- EMPTY WHAT THE LOAN NO LONGER COVERS, FIRST AND ALL OF IT. Each is a drop -- a
         * change of address, with no copy -- so there is no link to ration it against, and the
         * cache is waiting for these bytes. A unit still busy is on its way in; it goes on the
         * next call, once its promotion has retired. The sweep is over every unit, so it runs
         * only while a shrink may have stranded one (ClassPools::strand_check). */
        if (cp.strand_check) {
            bool left = false;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.slab.covered(us.slab_slot)) continue;
                /* Anything else still past the loan -- a promotion that has not retired, a staged
                 * unit -- becomes droppable later, so the sweep has to come back for it. */
                if (us.tier != Tier::VRAM || us.site != Site::Device ||
                    (heat && (heat->busy(u) || heat->is_leased(u)))) {
                    left = true;
                    continue;
                }
                if (drop(u, heat) < 0) { left = true; break; }
                drained = true;
            }
            cp.strand_check = left;
        }

        /* ---- THE WORKING RESERVE. The shadow and staging slots have to stay free whatever the
         * loan is, or the mover cannot turn over: a loan that shrank past units sitting in the
         * slots that were serving as the reserve leaves the slab too full by that many, and the
         * coldest lent units go -- drops, never copies. A slot already on its way back below the
         * target counts as free, because it is, a barrier from now; draining against it again is
         * how a slab empties itself. */
        int64_t pending = 0;
        for (const Move& m : moves_) {
            if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
            if (unit_[(size_t)m.unit].klass != (int32_t)c || !cp.slab.covered(m.slab_slot)) continue;
            if (m.kind == Move::Kind::Drop || m.kind == Move::Kind::Demote ||
                m.kind == Move::Kind::Release)
                ++pending;
        }
        int64_t short_by = cp.n_work - (cp.slab.n_free() + pending);
        while (short_by > 0) {
            int32_t victim = -1;
            float coldest = 0.0f;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || !cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot < 0) continue;
                if (heat && (heat->busy(u) || heat->is_leased(u))) continue;
                const float h = heat ? heat->heat(u) : 0.0f;
                if (victim < 0 || h < coldest) { victim = u; coldest = h; }
            }
            if (victim < 0 || drop(victim, heat) < 0) break;
            drained = true;
            --short_by;
        }
        /* AND WITH NO LENT UNIT LEFT TO DROP, THE COLDEST BUDGETED ONE GOES BACK TO THE POOL -- a
         * real demotion, one a call. A swap can leave the free slots in the lent range, and a loan
         * that then shrinks away takes them with it; the reserve has to be rebuilt out of the
         * budgeted slots or the mover stops turning over. */
        if (short_by > 0) {
            int32_t victim = -1;
            float coldest = 0.0f;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot >= 0) continue;
                if (!plan_->units[(size_t)u].movable) continue;
                if (heat && (heat->busy(u) || heat->is_leased(u))) continue;
                const float h = heat ? heat->heat(u) : 0.0f;
                if (victim < 0 || h < coldest) { victim = u; coldest = h; }
            }
            if (victim >= 0) {
                if (heat) heat->set_busy(victim, true);
                Swap s;
                s.demote = victim;
                if (issue(s, heat) >= 0) drained = true;
            }
        }
        /* A drain and a fill in the same call fight: the fill would take the slots the drain is
         * trying to hand back. */
        if (drained) continue;

        /* ---- THEN FILL, every free slot past the working reserve. */
        RAD_TRY(fill(c, cp.slab.n_free() - cp.n_work, heat));
    }
    count_flex();
    return RAD_OK;
}

int Mover::idle_tick(HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;
    RAD_TRY(retire(heat));
    return balance(heat);
}

int Mover::reclaim(int lender, int64_t amount, HeatEngine* heat) {
    RAD_TRY(set_loan(lender, amount));
    if (loan_held(lender) <= loan_[lender]) return RAD_OK;
    if (!compute_ || !mover_) {
        RAD_ERR("mover: a loan cannot be recalled with no compute stream bound; nothing orders "
                "the kernels that may still read the slots");
        return RAD_E_STATE;
    }
    /* NO COPY IS STILL WRITING A SLOT: everything the mover issued has landed, and retiring it
     * publishes the promotions that landed in slots about to be taken back. */
    RAD_TRY(rad_stream_sync(mover_));
    RAD_TRY(retire(heat));

    /* EVERY UNIT PAST THE LOAN GOES BACK TO ITS HOST COPY, which is a change of address. */
    std::vector<std::pair<size_t, int64_t>> freed;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        UnitState& us = unit_[(size_t)u];
        if (us.klass < 0 || us.slab_slot < 0) continue;
        const ClassPools& cp = pools_[(size_t)us.klass];
        if (!cp.is_flex(us.slab_slot) || cp.slab.covered(us.slab_slot)) continue;
        if (us.pool_slot < 0) {
            RAD_ERR("mover: unit %d sits in lent slot %lld with no host copy to go back to",
                    (int)u, (long long)us.slab_slot);
            return RAD_E_STATE;
        }
        freed.push_back({ (size_t)us.klass, us.slab_slot });
        us.slab_slot = -1;
        us.tier = Tier::HostPinned;
        us.site = Site::DeviceZeroCopy;
        republish(u);
        ++stats_.demotions;
        ++stats_.dropped;
        if (heat) {
            heat->set_tier(u, Tier::HostPinned);
            heat->set_busy(u, false);
        }
    }

    /* AND NO KERNEL STILL NAMES ONE. The compute stream is the only reader of the slab -- the
     * quarantine barrier rests on the same fact -- so once it has drained, nothing already issued
     * can reach a slot past the loan, and everything issued from here reads the host copies. */
    RAD_TRY(rad_stream_sync(compute_));
    RAD_TRY(retire(heat));
    for (const auto& f : freed) pools_[f.first].slab.add_free(f.second);

    /* EVERY RECORD WAITING ON ONE OF THOSE SLOTS IS DONE WAITING: its barrier is behind the
     * drain. A promotion's record has nothing to free -- its unit kept the pool slot -- and a
     * drop's frees the slot it left. */
    for (Move& m : moves_) {
        if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
        const int32_t klass = unit_[(size_t)m.unit].klass;
        if (klass < 0) continue;
        ClassPools& cp = pools_[(size_t)klass];
        if (!cp.is_flex(m.slab_slot) || cp.slab.covered(m.slab_slot)) continue;
        cp.slab.add_free(m.slab_slot);
        if (heat && m.kind != Move::Kind::Stage && m.kind != Move::Kind::Release)
            heat->set_busy(m.unit, false);
        m.state = Move::State::Free;
        m.unit = -1;
        m.slab_slot = m.pool_slot = -1;
    }
    /* Drain any stranded held slots in the recalled lender's segment: compute stream has drained,
     * so no device kernel can access them. */
    for (ClassPools& cp : pools_) {
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            while (true) {
                const int64_t top = cp.slab.top_held(l.seg);
                if (top < l.first) break;
                cp.slab.add_free(top);
            }
        }
    }
    count_flex();
    if (loan_held(lender) > loan_[lender]) {
        RAD_ERR("mover: recalled the %s loan to %lld and a slot at %lld is still held",
                lender == kLendKV ? "KV" : "arena", (long long)loan_[lender],
                (long long)loan_held(lender));
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int Mover::retire(HeatEngine* heat) {
    for (Move& m : moves_) {
        if (m.state != Move::State::Copying) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        UnitState& us = unit_[(size_t)m.unit];
        switch (m.kind) {
            case Move::Kind::Promote:
                us.slab_slot = m.slab_slot;
                /* The source slot goes back on quarantine, not to us -- unless the destination
                 * was lent, in which case it is kept (see issue()). */
                if (m.pool_slot >= 0) us.pool_slot = -1;
                us.tier = Tier::VRAM;
                us.site = Site::Device;
                ++stats_.promotions;
                if (heat) heat->set_tier(m.unit, Tier::VRAM);
                break;
            case Move::Kind::Demote:
                us.pool_slot = m.pool_slot;
                us.slab_slot = -1;
                us.tier = Tier::HostPinned;
                /* Demoted, not exiled: the unit is now the zero-copy tier's, so the GEMM streams
                 * it out of the pinned pool as it computes. There is no third dispatch path. */
                us.site = Site::DeviceZeroCopy;
                ++stats_.demotions;
                if (heat) heat->set_tier(m.unit, Tier::HostPinned);
                break;
            case Move::Kind::Stage:
                /* The destination was published at issue, with the event. All that is left is to
                 * clear the event: the copy has landed, so the wait is no longer needed and the
                 * common case -- a staged unit read many times before it is released -- costs no
                 * runtime call at all. No generation bump: the address did not change, and
                 * bumping would make the run phase re-wait on an event that has already
                 * signalled. */
                for (rad_weight w : plan_->units[(size_t)m.unit].weights) res_.settle(w);
                /* ...AND STRAIGHT TO FREE, SKIPPING THE QUARANTINE. The barrier exists to protect
                 * a slot on its way back to the free list from kernels that still name it; a
                 * staging copy returns no slot -- the slot stays with the unit until a later
                 * Release hands it back, and THAT is the move the barrier belongs to.
                 *
                 * Sending a Stage through the quarantine anyway is not merely redundant, it
                 * stalls the prefetch: the record stays occupied for two more dispatches, and
                 * unit_in_flight() therefore keeps reporting the unit busy, which blocks the very
                 * Release that would recycle its slot. With a ring two deep that is the whole
                 * supply. Invisible on the host backend, where every copy lands within the
                 * dispatch that issued it; on a real card the copies take tens of microseconds
                 * and the pipeline simply stops turning over. */
                m.state = Move::State::Free;
                m.unit = -1;
                m.slab_slot = m.pool_slot = -1;
                continue;
            case Move::Kind::Release:
            case Move::Kind::Drop:
                break;
        }
        /* PUBLISH. Both copies of the bytes are valid at this instant -- the mover wrote a shadow
         * slot and has not freed the source -- which is what makes this edit safe to make from
         * this thread with nothing synchronised: whichever side of the write a reader lands on
         * names bytes that are correct. */
        republish(m.unit);
        m.state = Move::State::Published;
    }

    /* QUARANTINE, and it is a SECOND pass on purpose. The slot being freed was named by kernels
     * issued in earlier layers and those may still be running. Re-recording the move's event on
     * the COMPUTE stream is the barrier: when it signals, every launch made before the publish
     * has retired and the slot is safe to write. It has to sit downstream of the publish in
     * stream order for that sentence to be true, which is why it cannot be folded into the loop
     * above. A slot reused too early is not a crash -- it is fluent text with a new hash. */
    for (Move& m : moves_) {
        if (m.state != Move::State::Published) continue;
        if (!compute_) {
            RAD_ERR("mover: no compute stream to quarantine the slot just freed by unit %d",
                    (int)m.unit);
            return RAD_E_STATE;
        }
        RAD_TRY(rad_event_record(m.ev, compute_));
        m.state = Move::State::Quarantine;
        ++stats_.quarantined;
    }

    for (Move& m : moves_) {
        if (m.state != Move::State::Quarantine) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        UnitState& us = unit_[(size_t)m.unit];
        ClassPools& cp = pools_[(size_t)us.klass];
        switch (m.kind) {
            case Move::Kind::Promote: cp.pool.add_free(m.pool_slot); break;
            case Move::Kind::Demote:  cp.slab.add_free(m.slab_slot); break;
            case Move::Kind::Release: cp.slab.add_free(m.slab_slot); break;
            case Move::Kind::Drop:    cp.slab.add_free(m.slab_slot); break;
            case Move::Kind::Stage:   break;   /* never reaches here; freed as soon as it lands */
        }
        if (heat && (m.kind == Move::Kind::Promote || m.kind == Move::Kind::Demote ||
                     m.kind == Move::Kind::Drop))
            heat->set_busy(m.unit, false);
        m.state = Move::State::Free;
        m.unit = -1;
        m.slab_slot = m.pool_slot = -1;
    }
    count_flex();
    return RAD_OK;
}

std::string Mover::report() const {
    const double gib = 1.0 / 1073741824.0;
    std::string s = fmt("mover: %llu dispatches, %llu promotions, %llu demotions, %llu staged "
                        "(%.2f GiB h2d, %.2f GiB d2h)\n",
                        (unsigned long long)stats_.dispatches,
                        (unsigned long long)stats_.promotions,
                        (unsigned long long)stats_.demotions,
                        (unsigned long long)stats_.prefetched,
                        (double)stats_.h2d_bytes * gib, (double)stats_.d2h_bytes * gib);
    /* Supply, not policy. These three are the difference between a mover at its budget and one
     * that never got to spend it, and without them they look identical from the outside. */
    s += fmt("  %llu found no free move record, %llu starved of a slab slot, %llu of a pool "
             "slot; %llu slots quarantined\n",
             (unsigned long long)stats_.no_record, (unsigned long long)stats_.starved_slab,
             (unsigned long long)stats_.starved_pool, (unsigned long long)stats_.quarantined);
    if (stats_.ssd_reads)
        s += fmt("  file tier: %llu demand reads (%.2f GiB), each a blocking pread in the middle "
                 "of a dispatch. Raise --host-pool-mib.\n",
                 (unsigned long long)stats_.ssd_reads, (double)stats_.ssd_bytes * gib);
    /* The routed layers read from the container, and what they cost the issuing thread: read is
     * the drive, wait is a window whose copies had not drained when it came round again. */
    if (stats_.file_layers) {
        const double rs = (double)stats_.file_read_ns * 1e-9;
        s += fmt("  file stager: %llu routed layers read, %llu units, %.2f GiB in %llu preads; "
                 "%.2f s reading (%.2f GB/s), %.2f s waiting on windows\n",
                 (unsigned long long)stats_.file_layers, (unsigned long long)stats_.file_units,
                 (double)stats_.file_bytes * gib, (unsigned long long)stats_.file_preads, rs,
                 rs > 0 ? (double)stats_.file_bytes / rs * 1e-9 : 0.0,
                 (double)stats_.file_wait_ns * 1e-9);
    }
    return s;
}

}  /* namespace rad */
