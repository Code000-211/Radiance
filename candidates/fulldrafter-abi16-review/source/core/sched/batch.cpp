/* batch.cpp -- building the step batch once. See batch.h for why.
 */
#include "batch.h"
#include "mm/processor.h"
#include "derive.h"
#include "device/device.h"
#include "tree_state.h"

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace rad {

/* Staging arrays are carved out of one slab per set rather than allocated one by one: the copies
 * are still per array (each carries only the prefix this step used, and those lengths differ by an
 * order of magnitude between a decode step and a prefill chunk), but the allocator is called twice
 * at init instead of two dozen times and every array in a set shares one mapping. */
static const int64_t k_carve_align = 256;

/* The longest range a build's upload names: one round of the workgroup that copies it, whose loads
 * are all in flight before its first store (rad_memcpy_ranges_async). */
static const int64_t k_range_max = 16384;

/* A HOST-SIDE BOUND, ROUNDED UP ONTO A COARSE GRID: to a multiple of `min_q`, and above 64 of
 * 1/32 to 1/64 of the value's power of two if that is coarser -- so ~3% over at a long context and
 * at most `min_q` over at a short one.
 *
 * `max_ctx_len` and the block-table pitch are ceilings -- a kernel reads each row's own length off
 * the card -- and they are the only arguments of a decode pass that move while a request runs:
 * attention and the QSA indexer size their grids by them. Exact, they change the pass's arguments
 * every step or two; rounded, a pass is the same dispatches with the same arguments for many steps
 * at a time, which is what lets a recorded pass be played instead of issued (see Ctx::run_step).
 * The price is grid slack that exits on its first compare, and at a short context, where the
 * floor is most of the bound, the whole attention term is small. */
/* THE FLOOR IS 1024 TOKENS. Every crossing re-records each kind of pass a step runs, twice -- the
 * recording and the audit before its first play -- at a few hundred microseconds of a rank thread
 * each, and at a 256-token floor a short-context decode crosses often enough to spend a tenth of
 * its rank threads recording. It is not higher because a block may choose its PATH by this bound
 * and not only its grid: rad_block_attn_gated_fp8.h keeps the cheaper dense attention while the
 * bound says every sequence is inside the indexer's budget, and a floor at that budget takes the
 * dense path away from every short context -- a verify step's attention at eight sequences takes
 * 48 us a layer at 2048 instead of 15. */
static const int64_t k_bound_tokens = 1024;

static inline int64_t bound_bucket(int64_t v, int64_t min_q) {
    if (v <= 0) return v;
    int64_t q = min_q < 1 ? 1 : min_q;
    if (v > 64) {
        const int lg = 63 - __builtin_clzll((unsigned long long)v);
        if (((int64_t)1 << (lg - 5)) > q) q = (int64_t)1 << (lg - 5);
    }
    return (v + q - 1) / q * q;
}

/* A negative device status inside build() aborts the step; there is nothing to fall back to and a
 * partially staged batch is worse than no batch. */
#define RAD_TRY_NULL(expr)                                                                     \
    do { int _s = (expr);                                                                      \
         if (_s < 0) { RAD_ERR("step batch: %s -> %s", #expr, rad_strerror(_s));               \
                       return nullptr; } } while (0)

template <class T>
static T* carve(char* base, int64_t& off, int64_t n) {
    int64_t o = off;
    if (n < 0) n = 0;
    off = align_up(off + n * (int64_t)sizeof(T), k_carve_align);
    return base ? reinterpret_cast<T*>(base + o) : nullptr;
}

/* The full token sequence of a request is prompt ++ output ++ draft, and n_computed indexes into
 * it. One accessor for all three keeps prefill, decode, recompute-after-preemption and speculative
 * verify on the same path: a preempted request re-prefills its own generated output because that
 * is simply the tail of the same sequence, and a verify reads past the committed end into the
 * drafter's proposal. No special case anywhere. */
static inline int32_t token_at(const StepEntry& e, int64_t i) {
    /* A placeholder entry -- a drafter context row whose request has already gone -- has no
     * tokens, and the pass it belongs to reads none. */
    if (!e.req) return 0;
    const Request& r = *e.req;
    int64_t np = (int64_t)r.prompt.size();
    if (i < np) return r.prompt[(size_t)i];
    i -= np;
    int64_t no = (int64_t)r.output.size();
    if (i < no) return r.output[(size_t)i];
    i -= no;
    return (e.draft && i >= 0 && i < e.n_draft) ? e.draft[(size_t)i] : 0;
}

/* ------------------------------------------------------------------ set layout */
/* Walked with null bases to measure and with real ones to assign. Keeping both in one function is
 * what stops them drifting, which is the classic way a hand-carved slab ends up with two arrays
 * sharing bytes. Every array is carved in PAIRS, host and device at the same offset, including the
 * host-only ones: that is what lets one copy of a prefix move a whole build, and it puts the
 * block tables -- the only part whose size moves from build to build -- last. */
void BatchBuilder::carve(Set& s, char* h) {
    char* const d = s.dev.empty() ? nullptr : static_cast<char*>(s.dev[0]);
    const int64_t T = max_tok_, S = max_seqs_;
    /* The most logit rows one step can want: one per sequence, or one per verified draft position
     * per sequence at a speculative step. It is what bounds Geom::max_logit_rows on the plugin
     * side, and the two have to agree or the plugin gathers into a buffer smaller than out_ids. */
    /* ...or one row a TOKEN, which an MTP history pass is: it gathers the trunk`s hidden state
     * for every position the step just computed, and names those rows through the same array. */
    const int64_t spec_factor = cfg_.n_spec > 0 ? (cfg_.n_spec >= 16 ? cfg_.n_spec + 1 : 16) : 1;
    int64_t O = S * spec_factor;
    if (O < T) O = T;
    /* ...and in the KL mode a row at every token as well, after the sampled ones
     * (Request::score). RadBuildCtx::max_out_rows says the same number to the plugin. */
    if (cfg_.kld()) O = T + S;

    int64_t oh = 0, od = 0;
    auto pair = [&](int32_t** ph, int32_t** pd, int64_t n) {
        *ph = ::rad::carve<int32_t>(h, oh, n);
        *pd = ::rad::carve<int32_t>(d, od, n);
    };
    /* Twice the step: its own tokens, then up to a step of the prompt that follows
     * (RadBatch::n_ahead). */
    pair(&s.h_tok,    &s.d_tok,    2 * T);
    pair(&s.h_pos,    &s.d_pos,    T);
    pair(&s.h_parent, &s.d_parent, T);
    auto pair16 = [&](uint16_t** ph, uint16_t** pd, int64_t n) {
        *ph = ::rad::carve<uint16_t>(h, oh, n);
        *pd = ::rad::carve<uint16_t>(d, od, n);
    };
    pair16(&s.h_tree_ancestors, &s.d_tree_ancestors, S * 16);
    pair(&s.h_cu,     &s.d_cu,     S + 1);
    pair(&s.h_sid,    &s.d_sid,    S);
    pair(&s.h_q,      &s.d_q,      S);
    pair(&s.h_ctx,    &s.d_ctx,    S);
    pair(&s.h_acc,    &s.d_acc,    S);
    pair(&s.h_cktok,  &s.d_cktok,  S);
    pair(&s.h_ckslot, &s.d_ckslot, S);
    pair(&s.h_out,    &s.d_out,    O);
    pair(&s.h_dout,   &s.d_dout,   S);
    pair(&s.h_rows,   &s.d_rows,   4 * S);
    pair(&s.h_first,  &s.d_first,  S * (int64_t)s.groups.size());
    s.h_grp = ::rad::carve<AdvGroup>(h, oh, (int64_t)s.groups.size());
    s.d_grp = ::rad::carve<AdvGroup>(d, od, (int64_t)s.groups.size());
    for (size_t gi = 0; gi < s.groups.size(); ++gi) {
        Group& g = s.groups[gi];
        pair(&g.h_slot, &g.d_slot, T);
        pair(&g.h_used, &g.d_used, S);
        pair(&g.h_sidx, &g.d_sidx, S * g.sidx_pitch);
    }
    /* THE ROTARY PLANES AND THE ENCODER-ROW TARGETS, only for a program with an encoder: nothing
     * else can move a rotary position off the token index, and a text-only deployment should not
     * upload twelve bytes a token of copies of `positions` every step. */
    pair(&s.h_rope,  &s.d_rope,  mm_on_ ? 3 * T : 0);
    pair(&s.h_mmrow, &s.d_mmrow, mm_on_ ? T : 0);
    /* Host-only: derived metadata is copied into the arena the buffer plan already placed. Carved
     * in the pair anyway, so the device offsets stay the host's -- a few kilobytes of slab nobody
     * reads. */
    {
        int32_t* unused = nullptr;
        char* hd = ::rad::carve<char>(h, oh, derived_bytes_);
        (void)::rad::carve<char>(d, od, derived_bytes_);
        s.h_derived = hd;
        pair(&s.h_convslot, &unused, S * (int64_t)kvgeom_.g.size());
    }
    /* ---- the block tables, LAST ------------------------------------------------------------
     *
     * A group's table is carved for the worst step -- max_seqs rows of max_ctx/block_size entries
     * -- because the slab has to hold one. At a long context and a small block that is over a
     * megabyte per paged group, and a model may declare several: the tables are nearly the whole
     * slab. A build fills n_seq rows of `pitch`, THIS step's longest table, and the device reads
     * them at that pitch (block_table_pitch below).
     *
     * ON THE CARD each group's rows have a FIXED place in this region, so the rows a build leaves
     * alone are still there for the next one (BtMirror). IN THE HOST SPAN the region is the
     * build's payload and range list instead, sized by what changed: the carve's host half only
     * says where the payload starts. */
    {
        int64_t bt_n = 0;
        s.bt_base = od;
        for (Group& g : s.groups) {
            g.bt_fix = bt_n;
            bt_n += align_up(S * g.max_blocks * (int64_t)sizeof(int32_t), k_carve_align) /
                    (int64_t)sizeof(int32_t);
        }
        pair(&s.bt_h0, &s.bt_d0, bt_n);
        for (Group& g : s.groups) g.d_bt = s.bt_d0 ? s.bt_d0 + g.bt_fix : nullptr;
    }
    s.host_bytes = oh;
    s.dev_bytes  = od;
}

void BatchBuilder::size_set(Set& s, const std::vector<KVGroupInfo>& groups) {
    s.groups.resize(groups.size());
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        const KVGeom::G& e = kvgeom_.g[gi];
        s.groups[gi].g = (rad_kvgroup)gi;
        s.groups[gi].block_size = e.block_size;
        int64_t mb = 1;
        if (e.paged) {
            if (e.kind == RAD_KV_WINDOW && e.window > 0 && e.window < max_ctx_) {
                /* THE MANAGER'S OWN CAP, TO THE BLOCK, BOUNDED BY THE CONTEXT. A windowed group
                 * recycles below the window, so its block table never holds more than
                 * window_blocks_for(window, block, speculative reserve) entries -- and the
                 * reserve is part of that number, because a verify reserves its draft rows ahead
                 * of the committed end before anyone knows which of them commit. `ensure` is
                 * never asked for a position past the context either, so a window close to it
                 * needs no more rows than the context has blocks.
                 *
                 * A rectangle one row narrower than the table makes build()'s pitch clamp drop
                 * the NEWEST entries, which every kernel reads as the current context: a
                 * silently narrower attention, not a refusal. */
                const int64_t rows_by_ctx = (max_ctx_ + e.block_size - 1) / e.block_size;
                mb = window_blocks_for(e.window, e.block_size,
                                       speculative_window_reserve(cfg_.n_spec));
                if (rows_by_ctx < mb) mb = rows_by_ctx;
            } else {
                mb = (max_ctx_ + e.block_size - 1) / e.block_size + 1;
            }
        }
        s.groups[gi].max_blocks = mb;
        s.groups[gi].sidx_pitch = e.sidx_width > 0 ? e.sidx_width : 1;
    }
    carve(s, nullptr);
}

int BatchBuilder::alloc_set(Set& s, const std::vector<KVGroupInfo>& groups) {
    size_set(s, groups);

    /* ONE SLAB PER RANK, on that rank's card. rad_dev_alloc has no device argument and uses the
     * current one, so the bind is part of the allocation. */
    s.dev.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        s.dev[r] = rad_dev_alloc(s.dev_bytes, RAD_MEM_DEVICE);
        if (!s.dev[r]) {
            RAD_ERR("step batch staging: could not reserve %s on rank %zu's card",
                    humanb(s.dev_bytes).c_str(), r);
            return RAD_E_NOMEM;
        }
    }
    RAD_TRY(rad_dev_set(ranks_[0].device));

    /* THE RING HOLDS THREE WORST-CASE BUILDS, which is the most a host running ahead of the card
     * can have in flight before a long-context prefill has to wait for its own upload two builds
     * back; a decode build's span is its fixed prefix and a few rows of tables, so hundreds fit. */
    int64_t worst_pay = 0;
    for (const Group& g : s.groups) worst_pay += max_seqs_ * g.max_blocks * (int64_t)sizeof(int32_t);
    const int64_t worst_ranges = worst_pay / k_range_max + max_seqs_ * (int64_t)s.groups.size() + 1;
    ranges_.reserve((size_t)worst_ranges);
    ring_.bytes = 3 * (s.host_bytes + worst_ranges * (int64_t)sizeof(RadCopyRange) + k_carve_align);
    ring_.host = static_cast<char*>(rad_dev_alloc(ring_.bytes, RAD_MEM_HOST_PINNED));
    if (!ring_.host) {
        RAD_ERR("step batch staging: could not reserve %s of pinned host staging",
                humanb(ring_.bytes).c_str());
        return RAD_E_NOMEM;
    }
    ring_.head = 0;
    ring_.first = 0;
    ring_.live = 0;

    /* Local: the host waits on these only to know an upload has READ its span, which a card-wide
     * release says as well as a system-wide one. */
    ev_.assign(kFlights, std::vector<RadEvent>(ranks_.size(), nullptr));
    ev_free_.clear();
    ev_free_.reserve(kFlights);
    for (int f = 0; f < kFlights; ++f) {
        for (size_t r = 0; r < ranks_.size(); ++r) {
            RAD_TRY(rad_dev_set(ranks_[r].device));
            if (rad_event_create_as(&ev_[(size_t)f][r], RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT) < 0)
                return RAD_E_DEVICE;
        }
        ev_free_.push_back(kFlights - 1 - f);
    }
    RAD_TRY(rad_dev_set(ranks_[0].device));

    for (Out& o : out_) {
        o.kv.assign(ranks_.size(), std::vector<RadKVGroupBatch>(s.groups.size()));
        o.batch.assign(ranks_.size(), RadBatch{});
    }
    pitch_.assign(s.groups.size(), 1);
    bt_mirror_.assign(s.groups.size(), BtMirror{});
    for (BtMirror& m : bt_mirror_) {
        m.seq.assign((size_t)max_seqs_, 0);
        m.len.assign((size_t)max_seqs_, -1);
    }
    ranges_.clear();
    return RAD_OK;
}

void BatchBuilder::forget_tables() {
    for (BtMirror& m : bt_mirror_) {
        m.pitch = -1;
        std::fill(m.len.begin(), m.len.end(), -1);
    }
}

int BatchBuilder::free_set(Set& s) {
    /* Destroying an event does not wait for its queued copy. In particular a
     * sampler refusal can leave this batch's upload pending at teardown. Drain
     * before freeing either end of the copy (and mapped winner metadata). */
    for (const auto& r : ranks_) {
        const int bind = rad_dev_set(r.device);
        const int st = bind < 0 ? bind : r.stream ? rad_stream_sync(r.stream) : RAD_OK;
        if (st < 0) {
            RAD_ERR("batch teardown: stream did not retire: %s; keeping staging alive",
                    rad_strerror(st));
            return st;
        }
    }
    for (size_t f = 0; f < ev_.size(); ++f)
        for (size_t r = 0; r < ev_[f].size(); ++r) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            if (ev_[f][r]) rad_event_destroy(ev_[f][r]);
        }
    ev_.clear();
    ev_free_.clear();
    for (size_t r = 0; r < s.dev.size(); ++r) {
        if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
        if (s.dev[r]) rad_dev_free(s.dev[r], RAD_MEM_DEVICE);
    }
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    s.dev.clear();
    if (ring_.host) { rad_dev_free(ring_.host, RAD_MEM_HOST_PINNED); ring_.host = nullptr; }
    ring_.first = 0;
    ring_.live = 0;
    s.groups.clear();
    for (Out& o : out_) { o.kv.clear(); o.batch.clear(); }
    return RAD_OK;
}

/* A SPAN OF THE RING, contiguous, of `need` bytes. A span that would run past the end starts over
 * at zero. Every flight overlapping the span is retired first -- its uploads have run, on every
 * rank -- oldest first, which is the order they were issued in and so the order the card
 * finishes them. */
int BatchBuilder::reserve(int64_t need, int64_t* off) {
    if (need > ring_.bytes) {
        RAD_ERR("step batch staging: a build of %s does not fit the %s ring",
                humanb(need).c_str(), humanb(ring_.bytes).c_str());
        return RAD_E_FULL;
    }
    int64_t at = ring_.head;
    if (at + need > ring_.bytes) at = 0;
    auto retire_oldest = [&]() -> int {
        const Flight& f = ring_.fifo[ring_.first];
        for (RadEvent e : ev_[(size_t)f.ev]) RAD_TRY(rad_event_sync(e));
        ev_free_.push_back(f.ev);
        ring_.first = (ring_.first + 1) % kFlights;
        --ring_.live;
        return RAD_OK;
    };
    for (;;) {
        bool clash = false;
        for (int k = 0; k < ring_.live && !clash; ++k) {
            const Flight& f = ring_.fifo[(ring_.first + k) % kFlights];
            clash = f.begin < at + need && at < f.end;
        }
        if (!clash) break;
        RAD_TRY(retire_oldest());
    }
    /* And a flight needs an event row: out of rows is out of flights. */
    while (ev_free_.empty() || ring_.live == kFlights) RAD_TRY(retire_oldest());
    *off = at;
    return RAD_OK;
}

/* THE SAME ARRAY ON ANOTHER RANK'S SLAB. Every slab is carved by the same walk() from the same
 * sizes, so an array's offset is rank-independent: the byte offset off rank 0's base, re-applied
 * to rank r's. */
template <class T>
static inline T* on_rank(void* base_r, void* base_0, T* p0) {
    return p0 ? (T*)((char*)base_r + ((char*)p0 - (char*)base_0)) : nullptr;
}

/* ------------------------------------------------------------------ init */
int BatchBuilder::init(Program& prog, const Config& cfg, const ChunkGeometry& geo,
                       const KVGeom& kvg, const std::vector<RankIO>& ranks) {
    prog_ = &prog;
    cfg_ = cfg;
    geo_ = geo;
    kvgeom_ = kvg;
    ranks_ = ranks;
    if (ranks_.empty()) return RAD_E_INVAL;
    /* A null stream is the default stream and is legal; a null Program is not -- a derived
     * buffer would have no arena to land in. */
    for (const RankIO& r : ranks_)
        if (!r.prog) return RAD_E_INVAL;
    RAD_TRY(layout(prog, cfg));
    RAD_TRY(alloc_set(set_, prog.kv_groups));
    if (tree_state_enabled(prog)) {
        /* THE STAGING REGION, PLUS THE DLOOP CONTEXT SPAN IN ITS TAIL.
         *
         * The span is [start,count] a sequence, and it belongs in this region rather than in a
         * buffer of its own because this one is already allocated once, host-mapped and reused
         * every step -- so the span costs no per-step allocation and its upload rides the same
         * stream as the path and the reset flags it travels with.
         *
         * Only a full-context backend declares `dloop_context_span` at all; the tail is reserved
         * unconditionally so `init` and `device_bytes` agree about the slab's size whether or not
         * one is present. */
        /* paths[S*P] + cu{0,1} + lengths[S] + starts[S] + absolute[S*P] + resets[S], P the pitch. */
        span_at_ = max_seqs_ * (2 * tree_path_pitch_ + 3) + 2;
        tree_paths_ = static_cast<int32_t*>(rad_dev_alloc(
            (span_at_ + max_seqs_ * 2) * sizeof(int32_t), RAD_MEM_HOST_MAPPED));
        if (!tree_paths_) return RAD_E_NOMEM;
        dloop_full_ = dloop_full_enabled(prog);
    }
    out_next_ = 0;
    out_last_ = -1;
    if (mm_on_) RAD_TRY(alloc_media());

    /* THE RESERVATION AND WHAT ACTUALLY CROSSES ARE DIFFERENT NUMBERS, so the line says both.
     * Nearly all of the slab is block-table rectangles carved for the worst step, and a step
     * copies only the rows it filled -- an operator reading the reservation alone would price
     * the per-step cost far above what it is. */
    RAD_INFO("step batch: %s pinned staging ring + %s device slab a rank, %d KV group(s), "
             "%d derived buffer(s); of the slab, %s is block tables carved for the worst step "
             "and only the filled rows cross",
             humanb(ring_.bytes).c_str(), humanb(set_.dev_bytes).c_str(),
             (int)prog.kv_groups.size(), (int)derived_.size(),
             humanb(set_.dev_bytes - set_.bt_base).c_str());
    return RAD_OK;
}

/* WHAT init() WILL ALLOCATE ON EVERY RANK'S CARD, before any of it exists: the VRAM budget is
 * resolved before the builder is, and a reservation it does not charge comes out of
 * --gpu-headroom-mib -- the room a kernel's first launch allocates from, where running out is a
 * fault inside the HIP runtime rather than an error. The slab's block tables grow with
 * --max-num-seqs and the media staging with --max-num-batched-tokens, so neither is small enough
 * to leave to the headroom. The same layout() and carve walk as init(), so the two cannot drift. */
int64_t BatchBuilder::device_bytes(Program& prog, const Config& cfg, const KVGeom& kvg) {
    BatchBuilder b;
    b.prog_ = &prog;
    b.cfg_ = cfg;
    b.kvgeom_ = kvg;
    if (b.layout(prog, cfg) < 0) return -1;
    b.size_set(b.set_, prog.kv_groups);
    const MediaBytes m = b.mm_on_ ? b.media_bytes() : MediaBytes{};
    return b.set_.dev_bytes + m.mm + m.pix + m.crd + m.cu;
}

int BatchBuilder::layout(Program& prog, const Config& cfg) {
    max_tok_ = cfg.max_tok;
    max_seqs_ = cfg.max_seqs;
    tree_path_pitch_ = std::max<int64_t>(cfg.n_spec, prog.ctx.max_spec) > 15
        ? static_cast<int64_t>(radiance::RAD_TREE_MAX_NODES) : 16;

    if (max_tok_ <= 0 || max_seqs_ <= 0) return RAD_E_INVAL;
    if ((int)kvgeom_.g.size() > RAD_SCHED_MAX_KV_GROUPS) {
        RAD_ERR("%d KV groups exceeds the %d this build stages for",
                (int)kvgeom_.g.size(), RAD_SCHED_MAX_KV_GROUPS);
        return RAD_E_FULL;
    }
    if (kvgeom_.g.size() != prog.kv_groups.size()) return RAD_E_INVAL;

    /* The block-table staging rectangle. A sequence never holds more blocks than its context (or
     * its window, which is the whole point of declaring one: a 4K-window layer in a 128K context
     * needs a row 32x shorter). The +1 is because a window's blocks are not aligned to the
     * window's start, so it can straddle one extra block. */
    max_ctx_ = cfg.max_ctx > 0 ? cfg.max_ctx : prog.meta.n_ctx_train;
    if (max_ctx_ <= 0) max_ctx_ = cfg.max_tok;

    /* The conv rolling window: width-1 entries of history plus room for the whole speculative
     * window, and a rejection then reads back at the slot the last accepted token left (spec §10).
     * The state tensor's own extent (conv_window_entries), so the cursor derived from it stays
     * inside the slot the kernels index. */
    conv_window_.assign(prog.kv_groups.size(), 1);
    conv_group_ = -1;
    n_conv_ = 0;
    for (size_t i = 0; i < prog.kv_groups.size(); ++i) {
        const KVGroupInfo& g = prog.kv_groups[i];
        if (g.decl.kind != RAD_KV_CONV) continue;
        ++n_conv_;
        conv_group_ = n_conv_ == 1 ? (int32_t)i : -1;   /* -1 once there is more than one */
        if (g.decl.conv_width > 0)
            conv_window_[i] = (int32_t)conv_window_entries(g.decl, cfg.n_spec);
    }

    /* ------------------------------------------------------------ derived buffers */
    derived_.clear();
    derived_bytes_ = 0;
    for (size_t bi = 1; bi < prog.buffers.size(); ++bi) {
        BufferInfo& b = prog.buffers[bi];
        if (b.decl.kind != RAD_BUF_DERIVED) continue;

        int idx = derive_index(b.decl.derive);
        if (idx < 0) {
            std::string known;
            for (int i = 0; i < derive_count(); ++i) {
                if (i) known += ", ";
                known += derive_row(i)->name;
            }
            RAD_ERR("buffer '%s' declares derivation '%s', which the core does not compute. "
                    "Known: %s", b.name.c_str(), b.decl.derive ? b.decl.derive : "(null)",
                    known.c_str());
            return RAD_E_NOTFOUND;
        }
        if (b.decl.dtype != RAD_I32) {
            RAD_ERR("derived buffer '%s' is %s; every derivation the core computes is i32",
                    b.name.c_str(), rad_dtype_name(b.decl.dtype));
            return RAD_E_DTYPE;
        }

        int64_t n = 1;
        for (uint32_t d = 0; d < b.decl.rank; ++d) n *= b.decl.shape[d];
        if (n <= 0) {
            RAD_ERR("derived buffer '%s' has an empty declared extent", b.name.c_str());
            return RAD_E_SHAPE;
        }

        Derived d;
        d.handle = (rad_buf)bi;
        d.name = derive_row(idx)->name;
        d.fn_index = idx;
        d.capacity = n;
        d.buf = &b;
        d.off = derived_bytes_;

        /* WHICH CONV GROUP THIS ONE IS ABOUT. `conv_state_index` is the derivation that reads a
         * rolling window, and a model may declare several of different depths -- so the buffer
         * names its group and 0 means "the only one of its kind". With more than one conv group
         * and no name the answer is a guess between two windows, and the wrong depth produces a
         * plausible cursor pointing at history the kernel never wrote. Refused instead. */
        if (std::strcmp(d.name, "conv_state_index") == 0) {
            if (b.decl.kv != 0) {
                if (b.decl.kv >= prog.kv_groups.size() ||
                    prog.kv_groups[b.decl.kv].decl.kind != RAD_KV_CONV) {
                    RAD_ERR("derived buffer '%s' names KV group %u, which is not a RAD_KV_CONV "
                            "group", b.name.c_str(), (unsigned)b.decl.kv);
                    return RAD_E_INVAL;
                }
                d.conv_grp = (int32_t)b.decl.kv;
            } else if (n_conv_ == 1) {
                d.conv_grp = conv_group_;
            } else {
                RAD_ERR("derived buffer '%s' runs 'conv_state_index' and names no KV group, but "
                        "%d RAD_KV_CONV groups are declared. Set RadBufDecl::kv -- a cursor "
                        "computed against the wrong window's depth points at history that was "
                        "never written.", b.name.c_str(), n_conv_);
                return RAD_E_INVAL;
            }
        }
        derived_.push_back(d);
        derived_bytes_ = align_up(derived_bytes_ + n * (int64_t)sizeof(int32_t), k_carve_align);
    }

    /* Two layers declaring the same derivation get two BufferInfos and two arena slots -- that is
     * what the buffer plan produced -- but they share ONE computation: run_derivations computes
     * each distinct derivation once per step and copies the result to every destination. The cost
     * §8 is about is the recomputation, not the bytes.
     *
     * SORTED BY (derivation, GROUP), because two buffers running one derivation over different
     * conv groups are different answers and the sharing rule below is "same as the previous". */
    std::sort(derived_.begin(), derived_.end(),
              [](const Derived& a, const Derived& b) {
                  return a.fn_index != b.fn_index ? a.fn_index < b.fn_index
                                                  : a.conv_grp < b.conv_grp;
              });

    mm_on_ = prog.encoder.modalities != 0;
    return RAD_OK;
}

/* ------------------------------------------------------------------ media staging */
BatchBuilder::MediaBytes BatchBuilder::media_bytes() {
    const RadEncoderDecl& e = prog_->encoder;
    n_embd_ = e.n_embd;
    enc_patches_ = e.max_patches;
    enc_dim_ = e.patch_dim;
    /* The most segments one pass can hold: a segment is at least one output row, `merge` patches. */
    enc_segs_ = e.max_patches / (e.merge > 0 ? e.merge : 1) + 1;
    MediaBytes m;
    m.mm  = max_tok_ * n_embd_ * (int64_t)sizeof(uint16_t);
    m.pix = enc_patches_ * enc_dim_ * (int64_t)sizeof(uint16_t);
    m.crd = 4 * enc_patches_ * (int64_t)sizeof(int32_t);
    m.cu  = (enc_segs_ + 1) * (int64_t)sizeof(int32_t);
    return m;
}

int BatchBuilder::alloc_media() {
    const MediaBytes m = media_bytes();
    const int64_t mm_bytes = m.mm, pix_bytes = m.pix, crd_bytes = m.crd, cu_bytes = m.cu;
    mm_dev_.assign(ranks_.size(), nullptr);
    enc_pix_dev_.assign(ranks_.size(), nullptr);
    enc_coord_dev_.assign(ranks_.size(), nullptr);
    enc_cu_dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        mm_dev_[r]        = rad_dev_alloc(mm_bytes, RAD_MEM_DEVICE);
        enc_pix_dev_[r]   = rad_dev_alloc(pix_bytes, RAD_MEM_DEVICE);
        enc_coord_dev_[r] = rad_dev_alloc(crd_bytes, RAD_MEM_DEVICE);
        enc_cu_dev_[r]    = rad_dev_alloc(cu_bytes, RAD_MEM_DEVICE);
        if (!mm_dev_[r] || !enc_pix_dev_[r] || !enc_coord_dev_[r] || !enc_cu_dev_[r]) {
            RAD_ERR("media staging: could not reserve %s on rank %zu's card",
                    humanb(mm_bytes + pix_bytes + crd_bytes + cu_bytes).c_str(), r);
            return RAD_E_NOMEM;
        }
    }
    RAD_TRY(rad_dev_set(ranks_[0].device));
    mm_host_        = static_cast<uint16_t*>(rad_dev_alloc(mm_bytes, RAD_MEM_HOST_PINNED));
    enc_pix_host_   = static_cast<uint16_t*>(rad_dev_alloc(pix_bytes, RAD_MEM_HOST_PINNED));
    enc_coord_host_ = static_cast<int32_t*>(rad_dev_alloc(crd_bytes, RAD_MEM_HOST_PINNED));
    enc_cu_host_    = static_cast<int32_t*>(rad_dev_alloc(cu_bytes, RAD_MEM_HOST_PINNED));
    if (!mm_host_ || !enc_pix_host_ || !enc_coord_host_ || !enc_cu_host_) {
        RAD_ERR("media staging: could not reserve %s of pinned host staging",
                humanb(mm_bytes + pix_bytes + crd_bytes + cu_bytes).c_str());
        return RAD_E_NOMEM;
    }
    if (rad_event_create_as(&mm_ev_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT) < 0) return RAD_E_DEVICE;
    RAD_INFO("media staging: %s a rank for a step's encoder rows and %s for an encoder pass of "
             "%lld patches", humanb(mm_bytes).c_str(),
             humanb(pix_bytes + crd_bytes + cu_bytes).c_str(), (long long)enc_patches_);
    return RAD_OK;
}

void BatchBuilder::free_media() {
    for (size_t r = 0; r < mm_dev_.size(); ++r) {
        if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
        if (mm_dev_[r])        rad_dev_free(mm_dev_[r], RAD_MEM_DEVICE);
        if (enc_pix_dev_[r])   rad_dev_free(enc_pix_dev_[r], RAD_MEM_DEVICE);
        if (enc_coord_dev_[r]) rad_dev_free(enc_coord_dev_[r], RAD_MEM_DEVICE);
        if (enc_cu_dev_[r])    rad_dev_free(enc_cu_dev_[r], RAD_MEM_DEVICE);
    }
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    mm_dev_.clear(); enc_pix_dev_.clear(); enc_coord_dev_.clear(); enc_cu_dev_.clear();
    if (mm_host_)        rad_dev_free(mm_host_, RAD_MEM_HOST_PINNED);
    if (enc_pix_host_)   rad_dev_free(enc_pix_host_, RAD_MEM_HOST_PINNED);
    if (enc_coord_host_) rad_dev_free(enc_coord_host_, RAD_MEM_HOST_PINNED);
    if (enc_cu_host_)    rad_dev_free(enc_cu_host_, RAD_MEM_HOST_PINNED);
    mm_host_ = nullptr; enc_pix_host_ = nullptr; enc_coord_host_ = nullptr; enc_cu_host_ = nullptr;
    if (mm_ev_) { rad_event_destroy(mm_ev_); mm_ev_ = nullptr; }
    mm_ev_live_ = false;
}

/* THE ENCODER ROWS OF ONE ENTRY. A row reads its embedding at token `ctx_len + j + shift` -- the
 * draft head's history pass shifts its tokens one ahead, and its embeddings with them -- and the
 * rows that land in a media run take that run's encoder row instead. Consecutive tokens of one run
 * are consecutive rows of its item, so a run is one copy. */
int64_t BatchBuilder::stage_mm_rows(const StepEntry& e, int64_t k, int64_t shift, int64_t n_rows,
                                    int32_t* rows) {
    const mm::PromptMedia& pm = *e.req->media;
    const int64_t prompt = (int64_t)e.req->prompt.size();
    int64_t added = 0;
    for (int32_t j = 0; j < e.n_tokens; ) {
        const int64_t ti = (int64_t)e.ctx_len + j + shift;
        if (ti >= prompt) break;                 /* media is only ever in the prompt */
        const mm::RopeLayout::Run* run = pm.rope.run_at(ti);
        if (!run) { ++j; continue; }
        const mm::Item& it = *pm.items[(size_t)run->item].item;
        if (!it.encoded || (int64_t)it.embd.size() < it.n_rows() * n_embd_) {
            RAD_ERR("step batch: request %llu reaches a media item that has not been encoded",
                    (unsigned long long)e.seq);
            return RAD_E_STATE;
        }
        /* The rest of this run inside this entry, and inside the prompt. */
        int64_t take = std::min<int64_t>((int64_t)run->tok + run->n - ti, e.n_tokens - j);
        take = std::min<int64_t>(take, prompt - ti);
        const int64_t row = (int64_t)run->row0 + (ti - run->tok);
        std::memcpy(mm_host_ + (n_rows + added) * n_embd_, it.embd.data() + row * n_embd_,
                    (size_t)(take * n_embd_) * sizeof(uint16_t));
        for (int64_t q = 0; q < take; ++q) rows[n_rows + added + q] = (int32_t)(k + j + q);
        added += take;
        j += (int32_t)take;
    }
    return added;
}

void BatchBuilder::fini() {
    free_media();
    if (free_set(set_) < 0) return; /* mapped winner metadata may still be in use */
    if (tree_paths_) rad_dev_free(tree_paths_, RAD_MEM_HOST_MAPPED);
    tree_paths_ = nullptr;
    derived_.clear();
}

/* ------------------------------------------------------------------ derivations */
int BatchBuilder::run_derivations(const StepPlan& plan, Set& s) {
    if (derived_.empty()) return RAD_OK;

    DeriveInput in;
    in.plan = &plan;
    in.cu_seqlens = s.h_cu;
    in.positions = s.h_pos;
    in.n_tok = plan.n_tok;
    in.n_seq = (int64_t)plan.e.size();
    in.state_chunk = geo_.state_chunk;

    /* derived_ is sorted by derivation, so "the same one as the previous entry" is the whole
     * sharing rule and it needs no map. IT IS A PAIR NOW: two buffers running the same derivation
     * over DIFFERENT KV GROUPS are different answers, and sharing them would give one group the
     * other's cursors. */
    int     last_fn = -1;
    int32_t last_grp = -1;
    const Derived* last = nullptr;
    for (Derived& d : derived_) {
        int32_t* host = (int32_t*)(s.h_derived + d.off);

        /* THE GROUP'S OWN SLOTS AND ITS OWN WINDOW. `conv_state_index` computes a cursor inside a
         * rolling window, so the depth has to be that group's: one number shared across groups
         * fits only one of them, and the wrong depth produces a plausible cursor pointing at
         * history the kernel never wrote. */
        const int32_t grp = d.conv_grp;
        in.conv_slot   = (grp >= 0) ? s.h_convslot + (int64_t)grp * max_seqs_ : nullptr;
        in.conv_window = (grp >= 0 && grp < (int32_t)conv_window_.size()) ? conv_window_[grp] : 1;

        if (d.fn_index == last_fn && d.conv_grp == last_grp && last) {
            d.produced = last->produced;
            std::memcpy(host, s.h_derived + last->off, (size_t)d.produced * sizeof(int32_t));
        } else {
            const DeriveRow* row = derive_row(d.fn_index);
            int64_t n = 0;
            int st = row->fn(in, host, d.capacity, &n);
            if (st < 0) {
                RAD_ERR("derivation '%s' for buffer '%s' failed: %s (declared capacity %lld)",
                        row->name, d.buf->name.c_str(), rad_strerror(st), (long long)d.capacity);
                return st;
            }
            d.produced = n;
            last_fn = d.fn_index;
            last_grp = d.conv_grp;
            last = &d;
        }

        /* INTO EVERY RANK'S ARENA, on that rank's stream. A derived buffer is an ordinary
         * transient the core fills, so it lives in the per-rank arena the buffer plan placed --
         * copying only into rank 0's leaves every other rank reading whatever its arena held. */
        for (const RankIO& io : ranks_) {
            if (d.handle >= io.prog->buffers.size()) return RAD_E_STATE;
            void* dst = io.prog->buffers[d.handle].ptr;
            if (!dst) {
                RAD_ERR("derived buffer '%s' has no arena slot; the buffer plan has not run",
                        d.buf->name.c_str());
                return RAD_E_STATE;
            }
            RAD_TRY(rad_memcpy_async(dst, host, d.produced * (int64_t)sizeof(int32_t),
                                     io.stream));
        }
    }
    return RAD_OK;
}


/* ------------------------------------------------------------------ per-group fill */
int BatchBuilder::build_groups(const StepPlan& plan, const IKVManager& kv, Set& s, Out& o) {
    const int64_t n_seq = (int64_t)plan.e.size();
    int32_t* const pay = s.bt_h0;   /* the payload: the table entries this build restages */
    int64_t np = 0;
    ranges_.clear();
    /* `n` entries of the payload from word `src` to word `dst` of the card's table region, as
     * ranges of at most k_range_max bytes. A run that continues the last range extends it, which
     * is what makes a whole restage of consecutive rows a handful of ranges and not one a row. */
    auto stage = [&](int64_t dst, int64_t src, int64_t n) {
        int64_t db = s.bt_base + dst * (int64_t)sizeof(int32_t);
        int64_t sb = s.bt_base + src * (int64_t)sizeof(int32_t);
        int64_t nb = n * (int64_t)sizeof(int32_t);
        if (!ranges_.empty()) {
            RadCopyRange& b = ranges_.back();
            if (b.src + b.bytes == sb && b.dst + b.bytes == db && b.bytes < k_range_max) {
                const int64_t t = std::min(nb, k_range_max - b.bytes);
                b.bytes += t; db += t; sb += t; nb -= t;
            }
        }
        while (nb > 0) {
            const int64_t t = std::min(nb, k_range_max);
            ranges_.push_back(RadCopyRange{ sb, db, t });
            db += t; sb += t; nb -= t;
        }
    };

    /* Every group's row, not just the first: a sequence with no state in a group must read -1
     * there, and a stale slot from the previous step is the worst possible value. */
    for (int64_t gi = 0; gi < (int64_t)kvgeom_.g.size(); ++gi)
        for (int64_t i = 0; i < n_seq; ++i) s.h_convslot[gi * max_seqs_ + i] = -1;

    for (size_t gi = 0; gi < s.groups.size(); ++gi) {
        Group& g = s.groups[gi];
        const KVGeom::G& L = kvgeom_.g[gi];

        /* THIS STEP's pitch, which build() measured before it sized the span. RadKVGroupBatch
         * carries it, so the device reads the rows at the width they were staged. A new pitch
         * moves every row, so nothing the card holds is where this build reads it. */
        const int64_t pitch = pitch_[gi];
        BtMirror& m = bt_mirror_[gi];
        if (L.paged && m.pitch != pitch) {
            m.pitch = pitch;
            std::fill(m.len.begin(), m.len.end(), -1);
        }

        int64_t t = 0;
        for (int64_t i = 0; i < n_seq; ++i) {
            const StepEntry& e = plan.e[(size_t)i];

            int64_t first = 0;
            if (L.paged) {
                /* The block table is the MANAGER's, read by sequence id. Request::blocks is not
                 * the source of truth: two allocators disagreeing about what a sequence owns is
                 * the bug where a freed sequence's blocks are handed out from under it. */
                const std::vector<int32_t>& bt = kv.block_table(e.seq, (int32_t)gi);
                first = kv.first_block_pos(e.seq, (int32_t)gi);
                for (int32_t j = 0; j < e.n_tokens; ++j)
                    g.h_slot[t + j] = (plan.mtp_frontier || (e.n_write > 0 && j >= e.n_write))
                                          ? -1
                                          : kvgeom_.slot((int)gi, bt, first,
                                                         (int64_t)e.ctx_len + j);
                /* THE ROW, FROM WHAT THE CARD ALREADY HOLDS OF IT. The same sequence in the same
                 * row keeps every entry below the lowest one the manager changed and below the end
                 * of both the old and the new table; a decode step's row is then nothing, or the
                 * one block it appended. Anything else -- a new sequence, one that moved rows -- is
                 * the whole row, and its copies in other rows stop being current, since its marks
                 * are taken here. The marks are taken for every row either way. */
                const int64_t nb = std::min<int64_t>((int64_t)bt.size(), pitch);
                const int64_t chg = kv.take_table_changes(e.seq, (int32_t)gi);
                int64_t lo = 0, hi = pitch;
                if (m.len[(size_t)i] >= 0 && m.seq[(size_t)i] == e.seq) {
                    lo = std::min(std::min(chg, m.len[(size_t)i]), nb);
                    hi = std::max(nb, m.len[(size_t)i]);
                } else {
                    for (int64_t k = 0; k < max_seqs_; ++k)
                        if (m.seq[(size_t)k] == e.seq) m.len[(size_t)k] = -1;
                }
                if (lo < hi) {
                    int32_t* w = pay + np;
                    const int64_t nc = nb > lo ? nb - lo : 0;
                    if (nc > 0) std::memcpy(w, bt.data() + lo, (size_t)nc * sizeof(int32_t));
                    std::fill(w + nc, w + (hi - lo), -1);
                    stage(g.bt_fix + i * pitch + lo, np, hi - lo);
                    np += hi - lo;
                }
                m.seq[(size_t)i] = e.seq;
                m.len[(size_t)i] = nb;
            } else {
                /* A linear or conv group owns one state instance per sequence, so every token of
                 * the sequence maps to it and the "block table" row is that slot -- a kernel that
                 * only reads block tables then needs no second path. */
                int32_t st = kv.state_slot(e.seq, (int32_t)gi);
                for (int32_t j = 0; j < e.n_tokens; ++j) g.h_slot[t + j] = st;
                pay[np + i] = st;
                /* EVERY conv group, at its own row. This array feeds `DeriveInput::conv_slot`
                 * and NOTHING ELSE -- a conv KERNEL reads its slot from
                 * RadKVGroupBatch::state_index, which is filled per group a few lines below and
                 * always has been. See the note at conv_window_ in init(). */
                if (kvgeom_.g[gi].kind == RAD_KV_CONV)
                    s.h_convslot[(int64_t)gi * max_seqs_ + i] = st;
            }
            s.h_first[(int64_t)gi * max_seqs_ + i] = (int32_t)first;
            t += e.n_tokens;

            /* THE TOKENS THIS SEQUENCE ACTUALLY HOLDS, WHICH IS NOT min(length, window).
             *
             * `seqused` is a count from index 0 of the BLOCK TABLE, and the table of a windowed
             * group is REBASED at `first_block_pos` once the window slides -- kvgeom_.slot() above
             * subtracts exactly that. So the count the kernel needs is the held span, and the held
             * span is a whole number of BLOCKS: recycling leaves `window_blocks` of them, whose
             * last one is partially filled by however far past a block boundary the sequence has
             * got. `min(length, window)` reports the full window every time and therefore claims
             * up to block_size - 1 positions that were never written this pass -- and they are the
             * ones at the END of the table, which every attention kernel reads as the NEWEST
             * context. Their bytes are whatever the recycled block held.
             *
             * A NON-CAUSAL windowed group is where this bites: every query row attends to those
             * stale slots at full weight, and acceptance falls off a cliff at exactly the context
             * where the window first slides. A causal group hides the same defect, because a stale
             * slot past the query is masked anyway.
             *
             * No clamp to `window` any more: block granularity means the held span can exceed the
             * window by less than a block, and dropping that from the FRONT of the count would
             * drop the NEWEST tokens rather than the oldest. Attending to a few extra old
             * positions is what paging a sliding window costs, and it is what vLLM does too. */
            /* AND THE COUNT STOPS WHERE THE WRITES DO. Masking the slot alone is not enough:
             * `seqused` is what every attention kernel indexes the table with, so a rejected row
             * left in the count is a slot read at full weight whose bytes are the previous
             * tenant's -- and being at the END of the table it reads as the NEWEST context. The
             * write mask and the count are therefore one decision, applied once. */
            const int64_t wrote = e.n_write > 0 ? e.n_write : e.n_tokens;
            int64_t used = plan.mtp_frontier
                ? (int64_t)e.ctx_len - plan.draft_pass + 2 - first
                : (int64_t)e.ctx_len + wrote - first;
            if (used < 0) used = 0;
            g.h_used[i] = (int32_t)used;
            if (L.paged) {
                for (int64_t j = 0; j < g.sidx_pitch; ++j) g.h_sidx[i * g.sidx_pitch + j] = -1;
            } else {
                kv.state_index_row(e.seq, (int32_t)gi, g.h_sidx + i * g.sidx_pitch,
                                   (int)g.sidx_pitch);
            }
        }
        /* A state group's rows are one slot a sequence: restaged whole, since they are n_seq
         * words and one range. */
        if (!L.paged) {
            stage(g.bt_fix, np, n_seq);
            np += n_seq;
        }

        {
            const char* d0 = static_cast<const char*>(s.dev[0]);
            AdvGroup& a = s.h_grp[gi];
            a.block_size = (int32_t)L.block_size;
            a.paged      = L.paged ? 1 : 0;
            a.pitch      = (int32_t)pitch;
            a.pad        = 0;
            a.slot_off   = (const char*)g.d_slot - d0;
            a.used_off   = (const char*)g.d_used - d0;
            a.bt_off     = (const char*)g.d_bt - d0;
            a.first_off  = (const char*)(s.d_first + (int64_t)gi * max_seqs_) - d0;
        }

        for (size_t r = 0; r < ranks_.size(); ++r) {
            void* base = s.dev[r];
            RadKVGroupBatch& b = o.kv[r][gi];
            b.group = g.g;
            b.slot_mapping = on_rank(base, s.dev[0], g.d_slot);
            b.block_table = on_rank(base, s.dev[0], g.d_bt);
            b.block_table_pitch = pitch;
            b.seqused = on_rank(base, s.dev[0], g.d_used);
            b.state_index = on_rank(base, s.dev[0], g.d_sidx);
            b.state_index_pitch = g.sidx_pitch;
            b.block_size = L.block_size;
            b.max_blocks = pitch;

            /* The bytes cross in build()'s one slab copy: a group's arrays are carved out of
             * the same slab as everything else and there is nothing group-shaped about them. */
        }
    }
    pay_words_ = np;
    return RAD_OK;
}

bool BatchBuilder::qsa_dense_for(const StepPlan& plan, int64_t exact_to) const {
    int64_t ctx = 0, q = 0;
    for (const auto& e : plan.e) {
        ctx = std::max<int64_t>(ctx, e.ctx_len);
        q = std::max<int64_t>(q, e.n_tokens);
    }
    return std::min(bound_bucket(ctx, k_bound_tokens), max_ctx_) + q <= exact_to;
}

/* ------------------------------------------------------------------ build */
const RadBatch* BatchBuilder::build(const StepPlan& plan, const IKVManager& kv, int step) {
    if (plan.e.empty()) return nullptr;
    if ((int64_t)plan.e.size() > max_seqs_ || plan.n_tok > max_tok_) {
        RAD_ERR("step plan of %d sequences / %lld tokens exceeds the pools sized at %lld / %lld",
                (int)plan.e.size(), (long long)plan.n_tok,
                (long long)max_seqs_, (long long)max_tok_);
        return nullptr;
    }
    // Tree-layout speculation (Milestone 2 DARTree) is supported via TreeRegistry
    Set& s = set_;
    const int64_t n_seq = (int64_t)plan.e.size();

    /* THE CONTEXT BOUND, rounded (see bound_bucket), before anything that derives from it. */
    int32_t max_ctx = 0, max_q = 0;
    for (const StepEntry& e : plan.e) {
        if (e.ctx_len > max_ctx) max_ctx = e.ctx_len;
        if (e.n_tokens > max_q) max_q = e.n_tokens;
    }
    const int64_t ctx_bound = std::min<int64_t>(bound_bucket(max_ctx, k_bound_tokens), max_ctx_);

    /* THE BLOCK-TABLE PITCH OF EVERY GROUP FIRST, because it sizes the build's span of the ring:
     * the span is the fixed prefix and exactly the rows build_groups will pack. The pitch is a
     * property of THIS STEP, not of the deployment -- a rectangle wide enough for max_ctx would be
     * megabytes staged and copied on every decode step to carry a few hundred live entries.
     *
     * IT IS A FUNCTION OF THE CONTEXT BOUND, so the two move on the same step. Both are in a
     * recorded pass's key, and rounded separately they cross their buckets on different steps --
     * every crossing of either is a pass recorded again, so twice the recordings for nothing. A
     * row holds the sequence's context plus this step's tokens plus the scheduler's draft
     * lookahead, which is at most two speculative windows, and the pitch is that rounded up to
     * the bound's own grid. A table longer than that (none is, today) takes its own rounded
     * length, which is correct and only costs the alignment. */
    const int64_t reach = ctx_bound + max_q + 2 * ((int64_t)std::max(plan.n_spec, 0) + 1);
    int64_t pay_need = 0;   /* the payload if every row were restaged whole */
    for (size_t gi = 0; gi < s.groups.size(); ++gi) {
        int64_t pitch = 1;
        if (kvgeom_.g[gi].paged) {
            int64_t need = 1;
            for (int64_t i = 0; i < n_seq; ++i) {
                const int64_t n = (int64_t)kv.block_table(plan.e[(size_t)i].seq, (int32_t)gi).size();
                if (n > need) need = n;
            }
            const int64_t bs = kvgeom_.g[gi].block_size > 0 ? kvgeom_.g[gi].block_size : 1;
            const int64_t gq = (k_bound_tokens + bs - 1) / bs;
            pitch = ((reach + bs - 1) / bs + gq - 1) / gq * gq;
            if (need > pitch) pitch = bound_bucket(need, gq);
            if (pitch > s.groups[gi].max_blocks) pitch = s.groups[gi].max_blocks;
        }
        pitch_[gi] = pitch;
        pay_need += n_seq * pitch * (int64_t)sizeof(int32_t);
    }
    /* THE SPAN: the fixed prefix, the payload, and the range list behind it, each at its bound --
     * a range a row or state group plus one a k_range_max of payload. What the build turns out to
     * use is usually far less, and only that is held once it is known. */
    const int64_t max_ranges = pay_need / k_range_max + n_seq * (int64_t)s.groups.size() + 1;
    int64_t off = 0;
    const int64_t need = align_up(s.bt_base + align_up(pay_need, 16) +
                                  max_ranges * (int64_t)sizeof(RadCopyRange), k_carve_align);
    RAD_TRY_NULL(reserve(need, &off));
    carve(s, ring_.host + off);
    Out& o = out_[out_next_];

    const int64_t n_tok = plan.n_tok;

    /* ---------------------------------------------------------- tokens, positions, boundaries */
    int64_t k = 0;
    /* PER HALF as well as overall: a mixed step`s max_q is its prefill chunk, and a block that
     * issues the decode rows separately must not select their kernel band with it. */
    int32_t max_q_dec = 0, max_q_pf = 0;
    int n_ck = 0;
    int64_t n_out = 0, n_dout = 0;
    int32_t rope_mixed = 0;
    int64_t n_mm = 0;
    /* THE LEGACY uint16[S,16] ANCESTOR MASK DESCRIBES ONLY TREES OF AT MOST 16 ROWS. A batch
     * verifying a wider tree publishes no mask at all -- a truncated one would read as valid
     * ancestry -- and its attention must walk `spec_parent` instead. */
    bool wide_tree = false;
    s.h_cu[0] = 0;
    for (int64_t i = 0; i < n_seq; ++i) {
        const StepEntry& e = plan.e[(size_t)i];
        s.h_rows[i]                 = e.dev ? 1 : 0;
        s.h_rows[max_seqs_ + i]     = e.dev_tok;
        s.h_rows[2 * max_seqs_ + i] = e.dev_out;
        s.h_rows[3 * max_seqs_ + i] = e.dev_aux;
    }
    for (int64_t i = 0; i < n_seq; ++i) {
        const StepEntry& e = plan.e[(size_t)i];
        /* The advance indexes the card's StepSlots by this, and there are max_seqs of them. */
        if (e.card < 0 || e.card >= max_seqs_) {
            RAD_ERR("step plan row %lld: sequence %llu has card index %d, outside the %lld step "
                    "slots", (long long)i, (unsigned long long)e.seq, (int)e.card,
                    (long long)max_seqs_);
            return nullptr;
        }
        /* THE MTP HEAD READS ONE POSITION AHEAD. Its index i pairs the trunk`s hidden state at i
         * with the EMBEDDING OF THE TOKEN AT i + 1, and predicts i + 2 (rad_block_mtp_fp8.h). The
         * shift is applied once, here, so that no kernel and no plugin carries it -- and
         * token_at() reaches the drafter`s own earlier rounds through `draft` exactly as a verify
         * step reaches its proposal. StepPlan::tok_shift says how far, because it is the HEAD's
         * indexing and not the draft round's: DFlash2 is a draft pass that shifts by nothing. */
        const int64_t shift = plan.tok_shift;
        if (plan.draft_pass == 0 &&
            (e.is_prefill || e.n_tokens > (int32_t)radiance::RAD_TREE_MAX_NODES || !e.produces_token)) {
            radiance::TreeRegistry::instance().clear(e.seq);
        }
        radiance::ActiveTreeRecord trec;
        const bool has_draft_tree = !plan.mtp_frontier && !e.is_prefill && e.produces_token &&
            e.n_tokens <= (int32_t)radiance::RAD_TREE_MAX_NODES &&
            radiance::TreeRegistry::instance().get(e.seq, &trec);
        const bool small_tree = e.n_tokens <= (int32_t)radiance::RAD_TREE_LEGACY_MASK_NODES;
        if (!e.is_prefill && !small_tree) wide_tree = true;
        if (has_draft_tree) {
            s.h_tok[k + 0] = token_at(e, (int64_t)e.ctx_len + shift);
            s.h_pos[k + 0] = static_cast<int32_t>(trec.positions[0]);
            s.h_parent[k + 0] = trec.parents[0];
            for (int32_t j = 1; j < e.n_tokens; ++j) {
                s.h_tok[k + j] = (j - 1 < e.n_draft && e.draft && e.draft[j - 1] >= 0)
                               ? e.draft[j - 1]
                               : token_at(e, (int64_t)e.ctx_len + j + shift);
                s.h_pos[k + j] = static_cast<int32_t>(trec.positions[j]);
                s.h_parent[k + j] = trec.parents[j];
            }
            if (small_tree && s.h_tree_ancestors) {
                for (size_t m = 0; m < radiance::RAD_TREE_LEGACY_MASK_NODES; ++m) {
                    s.h_tree_ancestors[i * 16 + m] = radiance::tree_mask16(trec.ancestors[m]);
                }
            }
        } else {
            for (int32_t j = 0; j < e.n_tokens; ++j) {
                s.h_tok[k + j] = plan.mtp_frontier ? 0
                    : token_at(e, (int64_t)e.ctx_len + j + shift);
                s.h_pos[k + j] = e.ctx_len + (plan.mtp_frontier ? 0 : j);
                s.h_parent[k + j] = j > 0 ? (j - 1) : -1;
            }
            if (small_tree && s.h_tree_ancestors) {
                for (size_t m = 0; m < 16; ++m) {
                    s.h_tree_ancestors[i * 16 + m] = static_cast<uint16_t>((1u << (m + 1)) - 1u);
                }
            }
        }
        /* THE ROTARY POSITION OF EACH ROW, at the row's own position -- the draft head's rows
         * rotate at the index they write, not at the token they embed. Planes of n_tok. */
        if (mm_on_) {
            const mm::RopeLayout* rl =
                (e.req && e.req->media && !e.req->media->rope.empty()) ? &e.req->media->rope : nullptr;
            for (int32_t j = 0; j < e.n_tokens; ++j) {
                int32_t v[3];
                const int64_t p = (int64_t)e.ctx_len + (plan.mtp_frontier ? 0 : j);
                if (rl) rl->at(p, v);
                else v[0] = v[1] = v[2] = (int32_t)p;
                s.h_rope[k + j]             = v[0];
                s.h_rope[n_tok + k + j]     = v[1];
                s.h_rope[2 * n_tok + k + j] = v[2];
                if (v[1] != v[0] || v[2] != v[0]) rope_mixed = 1;
            }
            if (!plan.mtp_frontier && e.req && e.req->media && !e.req->media->items.empty()) {
                /* The staging is the last upload's until that upload has run. */
                if (mm_ev_live_) {
                    if (rad_event_sync(mm_ev_) < 0) return nullptr;
                    mm_ev_live_ = false;
                }
                const int64_t got = stage_mm_rows(e, k, shift, n_mm, s.h_mmrow);
                if (got < 0) return nullptr;
                n_mm += got;
            }
        }
        if (e.ckpt_tok >= 0 && e.ckpt_slot >= 0) {
            /* checkpoint_tok indexes THIS STEP's tokens, so it is the sequence's base plus the
             * offset the scheduler chose -- which the chunk split guarantees is the last token of
             * a checkpoint interval (§7.3). */
            s.h_cktok[n_ck]  = (int32_t)(k + e.ckpt_tok);
            s.h_ckslot[n_ck] = e.ckpt_slot;
            ++n_ck;
        }
        /* Which of this step's tokens the sampler wants logits for. `produces_token` is the
         * scheduler's answer to "is this entry's last token the sequence's last" -- false for the
         * middle chunk of a long prefill, which therefore contributes no row and does not touch
         * the lm_head at all. At a speculative verify every drafted position needs a distribution
         * to accept against, so the entry contributes 1 + n_spec rows and not one. */
        /* AN MTP PASS NAMES ITS OWN ROWS, and they are rows of the PREVIOUS step: the head
         * gathers the trunk`s hidden state for the positions it continues from, and which those
         * are is not a function of this batch at all. One per token of the pass, consecutive from
         * `hidden_row` -- which is a single row for a draft round and a whole chunk for a history
         * pass. See StepEntry::hidden_row. */
        if (e.stage_out && e.n_stage_out > 0) {
            /* AN EXPLICIT ROW LIST WINS, because it is the only form that can express a tree's
             * accepted path. A context pass after a tree verify must read the tap rows of the nodes
             * the target ACCEPTED -- `start + node[j]`, which is not the contiguous run this pass
             * would otherwise describe -- and this is where the drafter gets them. */
            for (int32_t j = 0; j < e.n_stage_out; ++j) s.h_out[n_out++] = e.stage_out[j];
        } else if (e.hidden_row >= 0) {
            const int32_t nr = e.hidden_n > 0 ? e.hidden_n : e.n_tokens;
            for (int32_t j = 0; j < nr; ++j) s.h_out[n_out++] = e.hidden_row + j;
        } else if (has_draft_tree) {
            for (int32_t j = 0; j < e.n_tokens; ++j) {
                s.h_out[n_out++] = (int32_t)(k + j);
            }
        } else if (e.produces_token) {
            const int32_t first = (int32_t)(k + e.n_tokens) - 1 - e.n_spec;
            for (int32_t j = 0; j <= e.n_spec; ++j) s.h_out[n_out++] = first + j;
        } else if (plan.draft_pass == -1) {
            /* A CONTEXT PASS NAMES ITS OWN ROWS EVEN WHEN THEY ARE THE IDENTITY. A drafter's
             * context pass mirrors the trunk's rows positionally, and the gather that reads this
             * list takes it as ONE flat run for the whole pass -- so an entry that stages nothing
             * does not fall back to "my own rows", it shifts every later entry's rows by its own
             * length and the drafter stores another sequence's hidden states. The identity is
             * spelled out for that reason: rows `k .. k + n_tokens - 1` are exactly the trunk rows
             * this entry mirrors.
             *
             * `-1` AND NOT EVERY NEGATIVE: `-2` is the winner-commit release, whose entries read
             * no row list at all and whose staging is sized for the sampler's rows. */
            for (int32_t j = 0; j < e.n_tokens; ++j) s.h_out[n_out++] = (int32_t)(k + j);
        }
        if (e.draft_out) s.h_dout[n_dout++] = (int32_t)(k + e.n_tokens - 1);
        else if (e.src_row >= 0) s.h_dout[n_dout++] = e.src_row;
        /* RADIANCE_DFLASH_TRACE prints what a drafter pass actually consumes -- the pass kind, the
         * absolute positions, and the token ids at them. A block-diffusion drafter reads an ANCHOR
         * the target has already chosen followed by mask ids, and every way of getting that wrong
         * (an anchor one token stale, a mask id that is an ordinary token, a position base that
         * disagrees with the KV the context pass wrote) produces the same symptom from outside:
         * acceptance near zero and output still exactly correct. So it is printed rather than
         * reasoned about. */
        static const bool trace = std::getenv("RADIANCE_DFLASH_TRACE") != nullptr;
        if (trace && plan.draft_pass != 0) {
            std::string ln = fmt("df %s seq %d pos %d..%d tok",
                                 plan.draft_pass > 0 ? "query  " : "context",
                                 (int)e.seq, (int)e.ctx_len, (int)(e.ctx_len + e.n_tokens - 1));
            for (int32_t j = 0; j < e.n_tokens && j < 12; ++j)
                ln += fmt(" %d", (int)s.h_tok[k + j]);
            RAD_INFO("%s", ln.c_str());
        }
        k += e.n_tokens;
        s.h_cu[i + 1] = (int32_t)k;
        s.h_sid[i] = e.card;
        s.h_q[i]   = e.n_tokens;
        s.h_ctx[i] = e.ctx_len;
        s.h_acc[i] = e.n_accepted;
        if (e.is_prefill) { if (e.n_tokens > max_q_pf) max_q_pf = e.n_tokens; }
        else               { if (e.n_tokens > max_q_dec) max_q_dec = e.n_tokens; }
    }
    if (k != n_tok) {
        RAD_ERR("step plan token count %lld does not match its entries (%lld)",
                (long long)n_tok, (long long)k);
        return nullptr;
    }

    /* THE SCORED ROWS (Request::score), after every row the sampler reads: a row at each token of
     * a scoring entry's prefill chunk, in entry order, which is the order Engine::score_rows walks
     * to map them back to a request and a position. They come after so that nothing indexing the
     * sampled rows -- the sampler, the card's acceptance (dev_out) -- sees them at all. A trunk
     * pass only: a draft pass's rows are the head's, not the model's. */
    if (cfg_.kld() && plan.draft_pass == 0 && plan.tok_shift == 0) {
        int64_t at = 0;
        for (int64_t i = 0; i < n_seq; ++i) {
            const StepEntry& e = plan.e[(size_t)i];
            if (e.req && e.req->score && e.is_prefill && e.hidden_row < 0)
                for (int32_t j = 0; j < e.n_tokens; ++j) s.h_out[n_out++] = (int32_t)(at + j);
            at += e.n_tokens;
        }
        if (n_out > max_tok_ + max_seqs_) {
            RAD_ERR("step batch: %lld logits rows past the %lld the KL mode sized for",
                    (long long)n_out, (long long)(max_tok_ + max_seqs_));
            return nullptr;
        }
    }

    /* THE PROMPT THAT FOLLOWS: the last sequence's next tokens, up to a step of them, after this
     * step's own (RadBatch::n_ahead). Only on a trunk pass whose last entry is a prefill chunk with
     * more of its sequence still to run -- a draft pass's tokens are shifted and it reads no such
     * data -- and bounded by the tokens the request has, so a re-prefill after preemption reads
     * its own output as the prompt's continuation, as token_at does. */
    int64_t n_ahead = 0;
    if (n_seq > 0 && plan.draft_pass == 0 && plan.tok_shift == 0) {
        const StepEntry& e = plan.e[(size_t)(n_seq - 1)];
        if (e.req && e.is_prefill) {
            const int64_t next = (int64_t)e.ctx_len + e.n_tokens;
            const int64_t have = (int64_t)e.req->prompt.size() + (int64_t)e.req->output.size();
            n_ahead = std::min<int64_t>(std::max<int64_t>(have - next, 0), max_tok_);
            for (int64_t j = 0; j < n_ahead; ++j) s.h_tok[n_tok + j] = token_at(e, next + j);
        }
    }

    /* From here the manager's change marks are taken, so a build that stops before its upload
     * is issued on every rank leaves the card's tables unknown to the next one. */
    if (int st = build_groups(plan, kv, s, o); st < 0) {
        forget_tables();
        RAD_ERR("step batch: build_groups -> %s", rad_strerror(st));
        return nullptr;
    }
    const int64_t rng_at = s.bt_base + align_up(pay_words_ * (int64_t)sizeof(int32_t), 16);
    const int n_rng = (int)ranges_.size();
    if (n_rng > max_ranges) {
        forget_tables();
        RAD_ERR("step batch: %d upload ranges past the bound of %lld", n_rng,
                (long long)max_ranges);
        return nullptr;
    }
    RadCopyRange* const rng = reinterpret_cast<RadCopyRange*>(ring_.host + off + rng_at);
    std::memcpy(rng, ranges_.data(), (size_t)n_rng * sizeof(RadCopyRange));
    const int64_t used = align_up(rng_at + n_rng * (int64_t)sizeof(RadCopyRange), k_carve_align);

    /* ---------------------------------------------------------- ONE COPY PER RANK
     *
     * carve() lays out the host span and every rank's device slab in the same order, at the same
     * sizes, aligning the OFFSET and not the pointer. So below bt_base the device slab is a
     * byte-for-byte mirror of the host span, and a whole step's staging -- the token and position
     * arrays, the per-sequence arrays, and every KV group's slot map, seqused and state index --
     * crosses in one transfer, with the block-table entries that changed riding the same dispatch
     * as ranges.
     *
     * One copy per ARRAY would carry only the prefix the step used, which is the right trade when
     * the arrays are full and the wrong one when they are not. A decode step has about two
     * kilobytes of live staging spread over dozens of arrays -- one per array, per KV group, per
     * rank -- and at that size the bytes are the smallest part of the cost: each transfer costs a
     * dispatch plus the dependent gap behind it, which together dominate the DMA.
     *
     * The over-copy is the unused tail of the arrays sized by max_num_batched_tokens. At a full
     * prefill chunk it is nothing at all -- the same bytes cross either way and only the launch
     * count differs -- and at decode a few microseconds of DMA buys back dozens of dispatches.
     *
     * It also removes the need for a gate on `num_accepted`: the kernels read that operand
     * whenever the DEPLOYMENT speculates, not only when the step drafted, so copying it
     * conditionally leaves them reading the arena on exactly the step that had something to roll
     * back. Everything in the slab crosses on every step.
     *
     * AND IT STAYS ON THE COMPUTE STREAM, which is not an oversight. Moving it to a third,
     * staging stream -- so that a producer on the second lane could see its batch without
     * queueing behind compute -- is substantially SLOWER at every concurrency, byte-identical.
     * Sharing the queue is load-bearing: the split has to pay for two orderings the queue gives
     * away, and the one that hurts is the copy waiting on the event that says the step which last
     * read this slab is done -- it cannot start until the previous step has drained, where riding
     * the compute stream lets it issue in line. That is three cross-stream round-trips a step a
     * rank against a prize of a few percent. Do not rebuild the staging stream. */
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const int st = rad_memcpy_ranges_async(s.dev[r], ring_.host + off, s.bt_base, rng, n_rng,
                                               ranks_[r].stream);
        if (st < 0) {
            forget_tables();
            RAD_ERR("step batch: rank %zu's upload -> %s", r, rad_strerror(st));
            return nullptr;
        }
    }

    /* Derived metadata reads the host staging, so it runs after the fills and before the event. */
    RAD_TRY_NULL(run_derivations(plan, s));

    /* ---------------------------------------------------------- encoder rows
     * Behind the slab on every rank's stream, so the rows are on the card before any kernel of the
     * pass reads them: rank 0 takes the staged rows, every other rank the zeros that make the
     * vocabulary reduction count each row once. */
    if (n_mm > 0) {
        const int64_t bytes = n_mm * n_embd_ * (int64_t)sizeof(uint16_t);
        for (size_t r = 0; r < ranks_.size(); ++r) {
            const int st = r == 0 ? rad_memcpy_async(mm_dev_[r], mm_host_, bytes, ranks_[r].stream)
                                  : rad_memset_async(mm_dev_[r], 0, bytes, ranks_[r].stream);
            if (st < 0) {
                RAD_ERR("step batch: rank %zu's encoder rows -> %s", r, rad_strerror(st));
                return nullptr;
            }
        }
        if (rad_event_record(mm_ev_, ranks_[0].stream) < 0) return nullptr;
        mm_ev_live_ = true;
    }

    /* ---------------------------------------------------------- the batch, one per rank */
    for (size_t r = 0; r < ranks_.size(); ++r) {
        void* const base = s.dev[r];
        void* const b0   = s.dev[0];
        RadBatch& B = o.batch[r];
        B.phase = plan.phase;
        B.step = step;
        B.n_tok = n_tok;
        B.n_seq = n_seq;
        B.token_ids = on_rank(base, b0, s.d_tok);
        B.positions = on_rank(base, b0, s.d_pos);
        B.cu_seqlens = on_rank(base, b0, s.d_cu);
        B.seq_ids = on_rank(base, b0, s.d_sid);
        B.n_out = n_out;
        B.out_ids = n_out ? on_rank(base, b0, s.d_out) : nullptr;
        B.n_draft_out = n_dout;
        B.draft_out_ids = n_dout ? on_rank(base, b0, s.d_dout) : nullptr;
        B.q_lens = on_rank(base, b0, s.d_q);
        B.ctx_lens = on_rank(base, b0, s.d_ctx);
        B.max_q_len = max_q;
        B.n_seq_decode = plan.n_seq_decode;
        B.n_tok_decode = plan.n_tok_decode;
        B.max_q_len_decode = max_q_dec;
        B.max_q_len_prefill = max_q_pf;
        B.max_ctx_len = (int32_t)ctx_bound;
        B.n_kv_groups = (int)o.kv[r].size();
        B.kv = o.kv[r].data();
        B.draft_pass = plan.draft_pass;
        B.n_spec = plan.n_spec;
        /* PASSED WHENEVER THE DEPLOYMENT SPECULATES, not only on the steps that drafted. A step that
         * declines to draft still FOLLOWS one that did, and the linear kernels need to know how much
         * of that step survived before they can start. Gating this on the step's own draft count
         * makes the kernels fall back to their absent-operand default of 1 on exactly the step that
         * had something to roll back. */
        B.num_accepted = cfg_.n_spec > 0 ? on_rank(base, b0, s.d_acc) : nullptr;
        last_spec_is_tree_ = plan.spec_is_tree;
        B.spec_parent = (plan.spec_is_tree || plan.n_spec > 0) ? (const int32_t*)on_rank(base, b0, s.d_parent) : nullptr;
        B.tree_ancestors = (plan.spec_is_tree || plan.n_spec > 0) && !wide_tree
            ? (const uint16_t*)on_rank(base, b0, s.d_tree_ancestors) : nullptr;
        B.rope_pos = mm_on_ ? on_rank(base, b0, s.d_rope) : nullptr;
        B.rope_mixed = rope_mixed;
        B.n_mm_rows = n_mm;
        B.mm_rows = n_mm ? on_rank(base, b0, s.d_mmrow) : nullptr;
        B.mm_embd = n_mm ? mm_dev_[r] : nullptr;
        B.enc = 0;
        B.enc_n_patch = 0;
        B.enc_pixels = nullptr;
        B.enc_coord = nullptr;
        B.enc_n_seg = 0;
        B.enc_cu = nullptr;
        B.enc_max_seg = 0;
        B.n_checkpoints = n_ck;
        B.checkpoint_tok = n_ck ? on_rank(base, b0, s.d_cktok) : nullptr;
        B.checkpoint_slot = n_ck ? on_rank(base, b0, s.d_ckslot) : nullptr;
        B.n_ahead = n_ahead;
    }
    /* THE FLIGHT, behind everything that reads this span: the upload, and the derivation copies
     * run_derivations issued above. */
    const int ev = ev_free_.back();
    ev_free_.pop_back();
    for (size_t r = 0; r < ranks_.size(); ++r) {
        if (rad_event_record(ev_[(size_t)ev][r], ranks_[r].stream) < 0) {
            RAD_ERR("step batch: could not record rank %zu's staging event", r);
            ev_free_.push_back(ev);
            return nullptr;
        }
    }
    ring_.fifo[(ring_.first + ring_.live) % kFlights] = Flight{ off, off + used, ev };
    ++ring_.live;
    last_n_seq_ = n_seq;
    last_n_tok_ = n_tok;
    ring_.head = off + used;
    out_last_ = out_next_;
    out_next_ = (out_next_ + 1) % kOut;
    return &o.batch[0];
}

/* The batch the last build produced for one rank. Same numbers as build()'s return; different
 * device pointers, because each rank reads its own slab. */
const RadBatch* BatchBuilder::batch_for(int rank) const {
    if (out_last_ < 0) return nullptr;
    const Out& o = out_[out_last_];
    if (rank < 0 || (size_t)rank >= o.batch.size()) return nullptr;
    return &o.batch[(size_t)rank];
}

AdvanceArgs BatchBuilder::advance_args(int rank) const {
    AdvanceArgs a;
    if (rank < 0 || (size_t)rank >= set_.dev.size()) return a;
    const Set& s = set_;
    const char* d0 = static_cast<const char*>(s.dev[0]);
    auto off = [&](const void* p) { return (int64_t)((const char*)p - d0); };
    a.n_seq    = (int32_t)last_n_seq_;
    a.n_groups = (int32_t)s.groups.size();
    a.slab     = static_cast<char*>(s.dev[(size_t)rank]);
    a.tok_off  = off(s.d_tok);
    a.pos_off  = off(s.d_pos);
    a.ctx_off  = off(s.d_ctx);
    a.acc_off  = off(s.d_acc);
    a.out_off  = off(s.d_out);
    a.dout_off = off(s.d_dout);
    a.sid_off  = off(s.d_sid);
    a.cu_off   = off(s.d_cu);
    a.q_off    = off(s.d_q);
    a.row_off  = off(s.d_rows);
    a.row_pitch = (int32_t)max_seqs_;
    a.grp_off  = off(s.d_grp);
    /* THE TREE'S PARENT ARRAY, when the LAST build carried one. The advance that walks a tree's
     * acceptance runs after the build and must read the topology the attention kernel masked
     * with; `last_spec_is_tree_` is what the build itself recorded, so a linear build hands over
     * null and the flat path is taken. */
    if (last_spec_is_tree_ && s.d_parent)
        a.spec_parent = static_cast<const int32_t*>(
            on_rank(static_cast<void*>(s.dev[(size_t)rank]), s.dev[0], s.d_parent));
    if (mm_on_) {
        a.rope_off   = off(s.d_rope);
        a.rope_pitch = (int32_t)last_n_tok_;
    }
    return a;
}

/* ------------------------------------------------------------------ the encoder pass */
const RadBatch* BatchBuilder::build_encoder(const std::vector<EncoderPart>& parts, int step) {
    if (!mm_on_ || parts.empty()) return nullptr;
    int64_t n_patch = 0, n_seg = 0, max_seg = 0;
    enc_cu_host_[0] = 0;
    for (const EncoderPart& p : parts) {
        const mm::Item& it = *p.item;
        const int64_t seg_p = (int64_t)it.grid_h * it.grid_w;
        const int64_t np = seg_p * p.n_seg;
        if (it.patch_dim != enc_dim_ || n_patch + np > enc_patches_ ||
            n_seg + p.n_seg > enc_segs_ || p.seg0 + p.n_seg > it.n_seg) {
            RAD_ERR("encoder pass: %lld patches in %lld segments do not fit the declared pass of "
                    "%lld patches of %lld elements", (long long)(n_patch + np),
                    (long long)(n_seg + p.n_seg), (long long)enc_patches_, (long long)enc_dim_);
            return nullptr;
        }
        std::memcpy(enc_pix_host_ + n_patch * enc_dim_,
                    it.pixels.data() + (int64_t)p.seg0 * seg_p * enc_dim_,
                    (size_t)(np * enc_dim_) * sizeof(uint16_t));
        n_patch += np;
        for (int32_t q = 0; q < p.n_seg; ++q) {
            enc_cu_host_[n_seg + 1] = enc_cu_host_[n_seg] + (int32_t)seg_p;
            ++n_seg;
        }
        if (seg_p > max_seg) max_seg = seg_p;
    }
    /* Coordinates in planes of n_patch, which is what makes planes 0 and 1 a dense [2, n_patch]. */
    {
        std::vector<int32_t> tmp;
        int64_t at = 0;
        for (const EncoderPart& p : parts) {
            const mm::Item& it = *p.item;
            const int64_t seg_p = (int64_t)it.grid_h * it.grid_w;
            const int64_t np = seg_p * p.n_seg;
            tmp.resize((size_t)(4 * np));
            it.coords((int64_t)p.seg0 * seg_p, np, tmp.data());
            for (int c = 0; c < 4; ++c)
                std::memcpy(enc_coord_host_ + c * n_patch + at, tmp.data() + c * np,
                            (size_t)np * sizeof(int32_t));
            at += np;
        }
    }
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadStream st = ranks_[r].stream;
        if (rad_memcpy_async(enc_pix_dev_[r], enc_pix_host_,
                             n_patch * enc_dim_ * (int64_t)sizeof(uint16_t), st) < 0 ||
            rad_memcpy_async(enc_coord_dev_[r], enc_coord_host_,
                             4 * n_patch * (int64_t)sizeof(int32_t), st) < 0 ||
            rad_memcpy_async(enc_cu_dev_[r], enc_cu_host_,
                             (n_seg + 1) * (int64_t)sizeof(int32_t), st) < 0) {
            RAD_ERR("encoder pass: rank %zu's upload failed", r);
            return nullptr;
        }
    }
    Out& o = out_[out_next_];
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = o.batch[r];
        B = RadBatch{};
        B.phase = RAD_PHASE_PREFILL;
        B.step = step;
        B.n_kv_groups = (int)o.kv[r].size();
        B.kv = o.kv[r].data();
        B.enc = 1;
        B.enc_n_patch = n_patch;
        B.enc_pixels = enc_pix_dev_[r];
        B.enc_coord = static_cast<const int32_t*>(enc_coord_dev_[r]);
        B.enc_n_seg = n_seg;
        B.enc_cu = static_cast<const int32_t*>(enc_cu_dev_[r]);
        B.enc_max_seg = (int32_t)max_seg;
    }
    out_last_ = out_next_;
    out_next_ = (out_next_ + 1) % kOut;
    return &o.batch[0];
}

}  /* namespace rad */
