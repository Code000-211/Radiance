/* batch.cpp -- builds the step batch once per step. See batch.h.
 */
#include "batch.h"
#include "mm/processor.h"
#include "derive.h"
#include "device/device.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace rad {

/* Alignment of each staging array carved from a set's slab. One slab per set means a couple of
 * allocations at init instead of two dozen, with every array sharing one mapping. */
static const int64_t k_carve_align = 256;

/* Longest range in a build's upload: one round of the copying workgroup, which issues all its
 * loads before its first store (rad_memcpy_ranges_async). */
static const int64_t k_range_max = 16384;

/* Rounds a host-side bound up to a multiple of `min_q`, or above 64 to 1/32..1/64 of the value's
 * power of two if that is coarser: ~3% slack at long context, at most `min_q` at short.
 *
 * `max_ctx_len` and the block-table pitch are ceilings (kernels read each row's length from the
 * card) and the only decode-pass arguments that move as a request runs. Rounded, a pass keeps the
 * same arguments for many steps, so a recorded pass can be replayed (Ctx::run_step); the cost is
 * grid slack that exits on its first compare.
 *
 * Floor of 1024 tokens: every crossing re-records each pass kind twice (recording plus audit), a
 * few hundred microseconds of rank thread each, which a 256 floor makes frequent at short context.
 * Not higher, because a block may pick its path by this bound: rad_block_attn_gated_fp8.h keeps
 * dense attention while every sequence is inside the indexer's budget; with a 2048 floor, a
 * verify step's attention at eight sequences costs 48 us a layer instead of 15. */
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

/* A negative device status inside build() aborts the step: there is no fallback, and a partially
 * staged batch is worse than none. */
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

/* Token i of prompt ++ output ++ draft, the sequence n_computed indexes. One accessor keeps
 * prefill, decode, recompute after preemption and speculative verify on one path. */
static inline int32_t token_at(const StepEntry& e, int64_t i) {
    /* A placeholder entry (a drafter context row whose request is gone) has no tokens, and its
     * pass reads none. */
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
/* Called with null bases to measure and with real ones to assign, so the two cannot drift. Every
 * array is carved as a host/device pair at the same offset, host-only ones included, so one copy
 * of a prefix moves a whole build. The block tables, whose size varies per build, go last. */
void BatchBuilder::carve(Set& s, char* h) {
    char* const d = s.dev.empty() ? nullptr : static_cast<char*>(s.dev[0]);
    const int64_t T = max_tok_, S = max_seqs_;
    /* Most logit rows one step can want: one per verified position per sequence, or one per
     * token for an MTP history pass. Must agree with Geom::max_logit_rows on the plugin side, or
     * the plugin gathers into a buffer smaller than out_ids. */
    int64_t O = S * (1 + (cfg_.n_spec > 0 ? (int64_t)cfg_.n_spec : 0));
    if (O < T) O = T;
    /* KL mode adds a row at every token after the sampled ones (Request::score); keep in sync
     * with RadBuildCtx::max_out_rows. */
    if (cfg_.kld()) O = T + S;

    int64_t oh = 0, od = 0;
    auto pair = [&](int32_t** ph, int32_t** pd, int64_t n) {
        *ph = ::rad::carve<int32_t>(h, oh, n);
        *pd = ::rad::carve<int32_t>(d, od, n);
    };
    /* Twice the step: its own tokens, then up to a step of the following prompt
     * (RadBatch::n_ahead). */
    pair(&s.h_tok,    &s.d_tok,    2 * T);
    pair(&s.h_pos,    &s.d_pos,    T);
    pair(&s.h_parent, &s.d_parent, T);
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
    /* Rotary planes and encoder-row targets, only for a program with an encoder; nothing else
     * moves a rotary position off the token index. */
    pair(&s.h_rope,  &s.d_rope,  mm_on_ ? 3 * T : 0);
    pair(&s.h_mmrow, &s.d_mmrow, mm_on_ ? T : 0);
    /* Each row's adapter slot, carved only when the engine serves adapters, so an engine without
     * them lays the slab out as if the array did not exist. */
    pair(&s.h_lora,  &s.d_lora,  lora_on_ ? T : 0);
    /* Host-only (derived metadata is copied into the planned arena), but carved on both sides to
     * keep device offsets equal to host offsets. */
    {
        int32_t* unused = nullptr;
        char* hd = ::rad::carve<char>(h, oh, derived_bytes_);
        (void)::rad::carve<char>(d, od, derived_bytes_);
        s.h_derived = hd;
        pair(&s.h_convslot, &unused, S * (int64_t)kvgeom_.g.size());
    }
    /* ---- the block tables, last ------------------------------------------------------------
     *
     * Each group's table is carved for the worst step (max_seqs rows of max_ctx/block_size
     * entries), nearly the whole slab at long context. A build fills n_seq rows at `pitch`, this
     * step's longest table, and the device reads them at that pitch (block_table_pitch below).
     * On the card each group's rows have a fixed place, so rows a build leaves alone persist
     * (BtMirror). In the host span this region holds the payload and range list, sized by what
     * changed; the host half of the carve only marks where the payload starts. */
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
            /* A window row holds the window behind the step's first query plus the step's own
             * tokens (KVManager::ensure keeps both), plus a verify row's speculative lookahead. */
            const int64_t wspan = e.window + max_tok_ + 2 * ((int64_t)std::max(cfg_.n_spec, 0) + 1);
            int64_t span = (e.kind == RAD_KV_WINDOW && e.window > 0 && wspan < max_ctx_)
                         ? wspan : max_ctx_;
            mb = (span + e.block_size - 1) / e.block_size + 1;
        }
        s.groups[gi].max_blocks = mb;
        s.groups[gi].sidx_pitch = e.sidx_width > 0 ? e.sidx_width : 1;
    }
    carve(s, nullptr);
}

int BatchBuilder::alloc_set(Set& s, const std::vector<KVGroupInfo>& groups) {
    size_set(s, groups);

    /* One slab per rank, on its card. rad_dev_alloc uses the current device, hence the bind. */
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

    /* The ring holds three worst-case builds before a long-context prefill waits on an upload two
     * builds back. A decode build's span is far smaller, so hundreds fit. */
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

    /* Local events suffice: the host waits on them only to know an upload has read its span. */
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

void BatchBuilder::free_set(Set& s) {
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
}

/* A contiguous span of `need` bytes, wrapping to zero if it would run past the end. Every flight
 * overlapping it is retired first (its uploads have run on every rank), oldest first, which is
 * issue order and so completion order. */
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
    /* A flight also needs a free event row. */
    while (ev_free_.empty() || ring_.live == kFlights) RAD_TRY(retire_oldest());
    *off = at;
    return RAD_OK;
}

/* The same array on another rank's slab. Every slab has the same carve, so the byte offset from
 * rank 0's base applies to rank r's. */
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
    /* A null stream (the default stream) is legal; a null Program is not, since derived buffers
     * need its arena. */
    for (const RankIO& r : ranks_)
        if (!r.prog) return RAD_E_INVAL;
    RAD_TRY(layout(prog, cfg));
    RAD_TRY(alloc_set(set_, prog.kv_groups));
    out_next_ = 0;
    out_last_ = -1;
    if (mm_on_) RAD_TRY(alloc_media());

    /* Report both the reservation and the block-table share of it: tables are carved for the
     * worst step, but a step uploads only the rows it filled. */
    RAD_INFO("step batch: %s pinned staging ring + %s device slab a rank (%s of it block "
             "tables), %d KV group(s), %d derived buffer(s)",
             humanb(ring_.bytes).c_str(), humanb(set_.dev_bytes).c_str(),
             humanb(set_.dev_bytes - set_.bt_base).c_str(), (int)prog.kv_groups.size(),
             (int)derived_.size());
    return RAD_OK;
}

/* What init() will allocate on every rank's card, for the VRAM budget, which is resolved before
 * the builder. An uncharged reservation would eat --gpu-headroom-mib, where running out faults
 * inside the HIP runtime; the block tables grow with --max-num-seqs and the media staging with
 * --max-num-batched-tokens. Uses the same layout() and carve as init(). */
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

    if (max_tok_ <= 0 || max_seqs_ <= 0) return RAD_E_INVAL;
    if ((int)kvgeom_.g.size() > RAD_SCHED_MAX_KV_GROUPS) {
        RAD_ERR("%d KV groups exceeds the %d this build stages for",
                (int)kvgeom_.g.size(), RAD_SCHED_MAX_KV_GROUPS);
        return RAD_E_FULL;
    }
    if (kvgeom_.g.size() != prog.kv_groups.size()) return RAD_E_INVAL;

    /* Block-table rows are sized by context, or by window for a window group (size_set). The +1
     * block there is because a window is not block-aligned and can straddle one extra block. */
    max_ctx_ = cfg.max_ctx > 0 ? cfg.max_ctx : prog.meta.n_ctx_train;
    if (max_ctx_ <= 0) max_ctx_ = cfg.max_tok;

    /* Conv rolling window: width-1 entries of history plus the whole speculative window, so a
     * rejection reads back at the slot the last accepted token left (spec §10). Taken from the
     * state tensor's extent (conv_window_entries) so the derived cursor stays inside the slot. */
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

        /* `conv_state_index` reads a rolling window, and conv groups may differ in depth, so the
         * buffer names its group; 0 means the only conv group. With several conv groups and no
         * name, refuse: the wrong depth yields a cursor into history never written. */
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
                RAD_ERR("derived buffer '%s' runs 'conv_state_index' with %d RAD_KV_CONV "
                        "groups declared; set RadBufDecl::kv to the one it reads",
                        b.name.c_str(), n_conv_);
                return RAD_E_INVAL;
            }
        }
        derived_.push_back(d);
        derived_bytes_ = align_up(derived_bytes_ + n * (int64_t)sizeof(int32_t), k_carve_align);
    }

    /* Buffers declaring the same derivation keep separate arena slots but share one computation
     * per step (run_derivations copies the result; §8). Sorted by (derivation, group), since
     * run_derivations shares a result only with the previous entry and different conv groups
     * give different answers. */
    std::sort(derived_.begin(), derived_.end(),
              [](const Derived& a, const Derived& b) {
                  return a.fn_index != b.fn_index ? a.fn_index < b.fn_index
                                                  : a.conv_grp < b.conv_grp;
              });

    mm_on_ = prog.encoder.modalities != 0 || prog.audio_encoder.modalities != 0;
    lora_on_ = cfg.lora.enabled();
    return RAD_OK;
}

/* ------------------------------------------------------------------ media staging */
BatchBuilder::MediaBytes BatchBuilder::media_bytes() {
    /* The larger of the two encoders' passes, field by field, since they run one at a time. */
    enc_patches_ = enc_pix_elems_ = enc_segs_ = 0;
    n_embd_ = 0;
    for (const RadEncoderDecl* e : { &prog_->encoder, &prog_->audio_encoder }) {
        if (e->modalities == 0) continue;
        n_embd_ = std::max<int64_t>(n_embd_, e->n_embd);
        enc_patches_ = std::max<int64_t>(enc_patches_, e->max_patches);
        enc_pix_elems_ = std::max<int64_t>(enc_pix_elems_, e->max_patches * e->patch_dim);
        /* Most segments one pass can hold; a segment is at least one output row. */
        enc_segs_ = std::max<int64_t>(enc_segs_, e->max_patches / (e->merge > 0 ? e->merge : 1) + 1);
    }
    MediaBytes m;
    m.mm  = max_tok_ * n_embd_ * (int64_t)sizeof(uint16_t);
    m.pix = enc_pix_elems_ * (int64_t)sizeof(uint16_t);
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
             "up to %lld patches", humanb(mm_bytes).c_str(),
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

/* Row j reads its embedding at token `ctx_len + j + shift` (the draft head's history pass shifts
 * by one); rows inside a media run take that run's encoder row instead. Consecutive tokens of a
 * run are consecutive rows of its item, so each run is one copy. */
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
        /* The rest of this run within this entry and the prompt. */
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
    free_set(set_);
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

    /* derived_ is sorted by (derivation, group), so an entry matching the previous one on both
     * reuses its result. Different groups never share. */
    int     last_fn = -1;
    int32_t last_grp = -1;
    const Derived* last = nullptr;
    for (Derived& d : derived_) {
        int32_t* host = (int32_t*)(s.h_derived + d.off);

        /* The group's own slots and window depth; `conv_state_index` computes a cursor inside
         * that group's rolling window. */
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

        /* Copy into every rank's arena on that rank's stream; each rank has its own slot. */
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
    int32_t* const pay = s.bt_h0;   /* payload: the table entries this build restages */
    int64_t np = 0;
    ranges_.clear();
    /* Stage `n` payload words from word `src` to word `dst` of the card's table region, as ranges
     * of at most k_range_max bytes. A run continuing the last range extends it, so consecutive
     * whole rows become a few ranges instead of one per row. */
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

    /* Reset every group's row: a sequence with no state in a group must read -1, not a stale slot
     * from the previous step. */
    for (int64_t gi = 0; gi < (int64_t)kvgeom_.g.size(); ++gi)
        for (int64_t i = 0; i < n_seq; ++i) s.h_convslot[gi * max_seqs_ + i] = -1;

    for (size_t gi = 0; gi < s.groups.size(); ++gi) {
        Group& g = s.groups[gi];
        const KVGeom::G& L = kvgeom_.g[gi];

        /* This step's pitch, set by build() before it sized the span and passed in
         * RadKVGroupBatch. A new pitch moves every row, invalidating the mirror. */
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
                /* The manager's block table, by sequence id; Request::blocks is not the source
                 * of truth (see IKVManager). */
                const std::vector<int32_t>& bt = kv.block_table(e.seq, (int32_t)gi);
                first = kv.first_block_pos(e.seq, (int32_t)gi);
                for (int32_t j = 0; j < e.n_tokens; ++j)
                    g.h_slot[t + j] = kvgeom_.slot((int)gi, bt, first, (int64_t)e.ctx_len + j);
                /* Restage relative to what the card holds (BtMirror). The same sequence in the
                 * same row keeps entries below the first changed one and below the end of both
                 * old and new tables, so a decode step stages nothing or one appended block. A
                 * new or moved sequence restages the whole row, and its copies in other rows go
                 * unknown. Change marks are taken for every row. */
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
                    /* Guarded: an empty table's data() may be null, and memcpy's source must
                     * not be, even for zero bytes. */
                    if (nc > 0) std::memcpy(w, bt.data() + lo, (size_t)nc * sizeof(int32_t));
                    std::fill(w + nc, w + (hi - lo), -1);
                    stage(g.bt_fix + i * pitch + lo, np, hi - lo);
                    np += hi - lo;
                }
                m.seq[(size_t)i] = e.seq;
                m.len[(size_t)i] = nb;
            } else {
                /* A linear or conv group has one state slot per sequence; every token maps to it
                 * and the block-table row is that slot, so table-reading kernels need no second
                 * path. */
                int32_t st = kv.state_slot(e.seq, (int32_t)gi);
                for (int32_t j = 0; j < e.n_tokens; ++j) g.h_slot[t + j] = st;
                pay[np + i] = st;
                /* Every conv group, at its own row. Feeds only DeriveInput::conv_slot; conv
                 * kernels read RadKVGroupBatch::state_index (see conv_window_ in batch.h). */
                if (kvgeom_.g[gi].kind == RAD_KV_CONV)
                    s.h_convslot[(int64_t)gi * max_seqs_ + i] = st;
            }
            s.h_first[(int64_t)gi * max_seqs_ + i] = (int32_t)first;
            t += e.n_tokens;

            /* Tokens the sequence holds in this table, not min(length, window). `seqused` counts
             * from index 0 of the block table, which a window group rebases at `first_block_pos`
             * (as slot() does above). min(length, window) would claim up to block_size - 1
             * unwritten slots at the end of the table, read as the newest context; a non-causal
             * window group would attend to them. Do not clamp to `window` either: the held span
             * may exceed it by under a block, and clamping would drop the newest tokens. Attending
             * a few extra old positions is the cost of paging a window (vLLM does the same). */
            int64_t used = (int64_t)e.ctx_len + e.n_tokens - first;
            if (used < 0) used = 0;
            g.h_used[i] = (int32_t)used;
            if (L.paged) {
                for (int64_t j = 0; j < g.sidx_pitch; ++j) g.h_sidx[i * g.sidx_pitch + j] = -1;
            } else {
                kv.state_index_row(e.seq, (int32_t)gi, g.h_sidx + i * g.sidx_pitch,
                                   (int)g.sidx_pitch);
            }
        }
        /* A state group's rows are one slot per sequence: n_seq words, restaged whole. */
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

            /* The group's arrays are uploaded with the rest of the slab in build(). */
        }
    }
    pay_words_ = np;
    return RAD_OK;
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
    if (plan.spec_is_tree) {
        /* spec_parent is in the ABI and allocated, but no drafter emits a tree, so tree parents
         * and positions are not written. Filling in the linear-chain answer would verify a tree
         * against the wrong causal mask. */
        RAD_ERR("tree-layout speculation was requested but is not implemented in this build");
        return nullptr;
    }
    /* A mixed step issues its recurrent state over its decode rows and its chunked scan over its
     * prefill chunks, each half as a contiguous run, so the split is what decides which path a
     * sequence's state is written by. StepPlan::split_ordered() is that rule read back; a plan
     * that breaks it puts a row on the wrong path, which is wrong in a way no later stage can see
     * (a rejected draft left folded into the state, or a chunk stepped as one token), so refuse
     * the build instead of issuing it. Checked here rather than in the scheduler because this is
     * where the split becomes device arrays a kernel cannot undo. */
    int64_t bad = 0;
    if (!plan.split_ordered(&bad)) {
        RAD_ERR("mixed step plan entry %lld is a %s on the %s side of a split at %lld: every "
                "sequence before the split must be a decode row and every one after it a prefill "
                "chunk (Scheduler::step's sort)",
                (long long)bad, plan.e[(size_t)bad].is_prefill ? "chunk" : "decode row",
                bad < (int64_t)plan.n_seq_decode ? "decode" : "prefill",
                (long long)plan.n_seq_decode);
        return nullptr;
    }

    Set& s = set_;
    const int64_t n_seq = (int64_t)plan.e.size();

    /* The context bound, rounded (see bound_bucket), before anything that derives from it. */
    int32_t max_ctx = 0, max_q = 0;
    for (const StepEntry& e : plan.e) {
        if (e.ctx_len > max_ctx) max_ctx = e.ctx_len;
        if (e.n_tokens > max_q) max_q = e.n_tokens;
    }
    const int64_t ctx_bound = std::min<int64_t>(bound_bucket(max_ctx, k_bound_tokens), max_ctx_);

    /* Every group's block-table pitch comes first because it sizes the ring span. The pitch is
     * per step, not max_ctx wide, so decode does not stage megabytes of empty rows.
     *
     * It derives from the context bound so both cross their buckets on the same step (both key a
     * recorded pass; separate rounding would double the re-recordings). A row holds context plus
     * this step's tokens plus at most two speculative windows of draft lookahead, rounded up to
     * the bound's grid. A longer table takes its own rounded length. */
    const int64_t reach = ctx_bound + max_q + 2 * ((int64_t)std::max(plan.n_spec, 0) + 1);
    int64_t pay_need = 0;   /* payload if every row were restaged whole */
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
    /* The span: fixed prefix, payload and range list, each at its bound (one range per row or
     * state group plus one per k_range_max of payload). Only what the build uses is held once
     * known. */
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
    /* Also per half: a mixed step's max_q is its prefill chunk, which must not select the kernel
     * band for separately issued decode rows. */
    int32_t max_q_dec = 0, max_q_pf = 0;
    int n_ck = 0;
    int64_t n_out = 0, n_dout = 0;
    int32_t rope_mixed = 0;
    int64_t n_mm = 0;
    bool lora_rows = false;
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
        /* The advance indexes the card's max_seqs StepSlots by this. */
        if (e.card < 0 || e.card >= max_seqs_) {
            RAD_ERR("step plan row %lld: sequence %llu has card index %d, outside the %lld step "
                    "slots", (long long)i, (unsigned long long)e.seq, (int)e.card,
                    (long long)max_seqs_);
            return nullptr;
        }
        /* Token shift (StepPlan::tok_shift): the MTP head pairs hidden state i with the token
         * at i + 1 and predicts i + 2 (rad_block_mtp_fp8.h). Applied only here, so no kernel or
         * plugin carries it; token_at() reaches earlier draft rounds through `draft`. DFlash2
         * passes use 0. */
        const int64_t shift = plan.tok_shift;
        for (int32_t j = 0; j < e.n_tokens; ++j) {
            s.h_tok[k + j] = token_at(e, (int64_t)e.ctx_len + j + shift);
            s.h_pos[k + j] = e.ctx_len + j;
        }
        /* Every row of a sequence runs with its request's adapter (RadBatch::lora_slot), fixed at
         * admission; a placeholder row with no request is a base row. */
        if (lora_on_) {
            const int32_t ls = e.req ? e.req->lora_slot : -1;
            for (int32_t j = 0; j < e.n_tokens; ++j) s.h_lora[k + j] = ls;
            if (ls >= 0 && e.n_tokens > 0) lora_rows = true;
        }
        /* Rotary position of each row at its own position (draft-head rows rotate at the index
         * they write, not the token they embed). Three planes of n_tok. */
        if (mm_on_) {
            const mm::RopeLayout* rl =
                (e.req && e.req->media && !e.req->media->rope.empty()) ? &e.req->media->rope : nullptr;
            for (int32_t j = 0; j < e.n_tokens; ++j) {
                int32_t v[3];
                const int64_t p = (int64_t)e.ctx_len + j;
                if (rl) rl->at(p, v);
                else v[0] = v[1] = v[2] = (int32_t)p;
                s.h_rope[k + j]             = v[0];
                s.h_rope[n_tok + k + j]     = v[1];
                s.h_rope[2 * n_tok + k + j] = v[2];
                if (v[1] != v[0] || v[2] != v[0]) rope_mixed = 1;
            }
            if (e.req && e.req->media && !e.req->media->items.empty()) {
                /* Wait until the last upload from the staging has run. */
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
            /* checkpoint_tok indexes this step's tokens: the sequence's base plus the offset the
             * scheduler chose, the last token of a checkpoint interval (§7.3). */
            s.h_cktok[n_ck]  = (int32_t)(k + e.ckpt_tok);
            s.h_ckslot[n_ck] = e.ckpt_slot;
            ++n_ck;
        }
        /* Rows the sampler wants logits for. An entry with `hidden_row` names its own rows
         * (StepEntry::hidden_row), consecutive from it. Otherwise a `produces_token` entry gives
         * 1 + n_spec rows (every drafted position needs a distribution), and a mid-prefill chunk
         * gives none. */
        if (e.hidden_row >= 0) {
            const int32_t nr = e.hidden_n > 0 ? e.hidden_n : e.n_tokens;
            for (int32_t j = 0; j < nr; ++j) s.h_out[n_out++] = e.hidden_row + j;
        } else if (e.produces_token) {
            const int32_t first = (int32_t)(k + e.n_tokens) - 1 - e.n_spec;
            for (int32_t j = 0; j <= e.n_spec; ++j) s.h_out[n_out++] = first + j;
        }
        if (e.draft_out) s.h_dout[n_dout++] = (int32_t)(k + e.n_tokens - 1);
        /* RADIANCE_DFLASH_TRACE logs each drafter pass's kind, positions and token ids. A wrong
         * anchor, mask id or position base shows only as near-zero acceptance with correct
         * output. */
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

    /* Scored rows (Request::score): one per token of a scoring entry's prefill chunk, in entry
     * order (the order Engine::score_rows maps back). Appended after the sampled rows so the
     * sampler and the card's acceptance (dev_out) never see them. Trunk passes only. */
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

    /* Up to a step of the last sequence's following tokens, after this step's own
     * (RadBatch::n_ahead). Only on a trunk pass whose last entry is a prefill chunk, bounded by
     * the tokens the request has; like token_at, a re-prefill after preemption continues into its
     * own output. */
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

    /* build_groups consumes the manager's change marks, so any failure before every rank's
     * upload is issued must forget_tables(). */
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
    if (n_rng > 0) std::memcpy(rng, ranges_.data(), (size_t)n_rng * sizeof(RadCopyRange));
    const int64_t used = align_up(rng_at + n_rng * (int64_t)sizeof(RadCopyRange), k_carve_align);

    /* ---------------------------------------------------------- one copy per rank
     *
     * Below bt_base the device slab mirrors the host span byte for byte (carve() aligns offsets,
     * not pointers), so a step's whole staging crosses in one transfer, with changed block-table
     * entries as ranges in the same dispatch. Copying per array would move fewer bytes, but a
     * decode step has about 2 KB live across dozens of arrays, where per-transfer dispatch cost
     * dominates; the unused tail costs a few microseconds of DMA. Since everything crosses every
     * step, `num_accepted` needs no gate.
     *
     * The copy stays on the compute stream. A separate staging stream is slower at every
     * concurrency: the copy must wait on an event for the step that last read the slab, adding
     * cross-stream round-trips, while on the compute stream it issues in line. Do not move it. */
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const int st = rad_memcpy_ranges_async(s.dev[r], ring_.host + off, s.bt_base, rng, n_rng,
                                               ranks_[r].stream);
        if (st < 0) {
            forget_tables();
            RAD_ERR("step batch: rank %zu's upload -> %s", r, rad_strerror(st));
            return nullptr;
        }
    }

    /* Derivations read the host staging: after the fills, before the event. */
    RAD_TRY_NULL(run_derivations(plan, s));

    /* ---------------------------------------------------------- encoder rows
     * Behind the slab on every rank's stream. Rank 0 gets the staged rows, the others zeros, so the
     * vocabulary reduction counts each row once. */
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
        /* Passed whenever the deployment speculates, not only on steps that drafted: a step
         * without drafts may follow one that had drafts, and the linear kernels need its accepted
         * count instead of their absent-operand default of 1. */
        B.num_accepted = cfg_.n_spec > 0 ? on_rank(base, b0, s.d_acc) : nullptr;
        B.spec_parent = nullptr;      /* a linear chain; the tree case is refused above */
        B.rope_pos = mm_on_ ? on_rank(base, b0, s.d_rope) : nullptr;
        B.rope_mixed = rope_mixed;
        B.n_mm_rows = n_mm;
        B.mm_rows = n_mm ? on_rank(base, b0, s.d_mmrow) : nullptr;
        B.mm_embd = n_mm ? mm_dev_[r] : nullptr;
        /* Null on a step with no adapter row: the core then issues no adapter op, and the step
         * runs the same pass as an engine serving no adapters. */
        B.lora_slot = lora_rows ? on_rank(base, b0, s.d_lora) : nullptr;
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
    /* Record the flight behind everything reading this span: the upload and the derivation
     * copies. */
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
    a.sid_off  = off(s.d_sid);
    a.cu_off   = off(s.d_cu);
    a.q_off    = off(s.d_q);
    a.row_off  = off(s.d_rows);
    a.row_pitch = (int32_t)max_seqs_;
    a.grp_off  = off(s.d_grp);
    if (mm_on_) {
        a.rope_off   = off(s.d_rope);
        a.rope_pitch = (int32_t)last_n_tok_;
    }
    return a;
}

/* ------------------------------------------------------------------ the encoder pass */
const RadBatch* BatchBuilder::build_encoder(const std::vector<EncoderPart>& parts, int step,
                                            const RadEncoderDecl& enc) {
    if (!mm_on_ || parts.empty() || enc.modalities == 0) return nullptr;
    const int64_t enc_dim = enc.patch_dim;
    const int64_t max_segs = enc.max_patches / (enc.merge > 0 ? enc.merge : 1) + 1;
    int64_t n_patch = 0, n_seg = 0, max_seg = 0;
    enc_cu_host_[0] = 0;
    for (const EncoderPart& p : parts) {
        const mm::Item& it = *p.item;
        if (p.seg0 < 0 || p.n_seg <= 0 || p.seg0 + p.n_seg > it.n_seg) {
            RAD_ERR("encoder pass: segments [%d, %d) of an item of %d", p.seg0, p.seg0 + p.n_seg,
                    it.n_seg);
            return nullptr;
        }
        const int64_t np = it.patch_at(p.seg0 + p.n_seg) - it.patch_at(p.seg0);
        if (it.patch_dim != enc_dim || n_patch + np > enc.max_patches ||
            n_seg + p.n_seg > max_segs) {
            RAD_ERR("encoder pass: %lld patches in %lld segments do not fit the declared pass of "
                    "%lld patches of %lld elements", (long long)(n_patch + np),
                    (long long)(n_seg + p.n_seg), (long long)enc.max_patches, (long long)enc_dim);
            return nullptr;
        }
        std::memcpy(enc_pix_host_ + n_patch * enc_dim,
                    it.pixels.data() + it.patch_at(p.seg0) * enc_dim,
                    (size_t)(np * enc_dim) * sizeof(uint16_t));
        n_patch += np;
        for (int32_t q = p.seg0; q < p.seg0 + p.n_seg; ++q) {
            const int64_t seg_p = it.patch_at(q + 1) - it.patch_at(q);
            enc_cu_host_[n_seg + 1] = enc_cu_host_[n_seg] + (int32_t)seg_p;
            ++n_seg;
            if (seg_p > max_seg) max_seg = seg_p;
        }
    }
    /* Coordinates in planes of n_patch, so planes 0 and 1 form a dense [2, n_patch]. */
    {
        std::vector<int32_t>& tmp = enc_coord_tmp_;
        int64_t at = 0;
        for (const EncoderPart& p : parts) {
            const mm::Item& it = *p.item;
            const int64_t np = it.patch_at(p.seg0 + p.n_seg) - it.patch_at(p.seg0);
            tmp.resize((size_t)(4 * np));
            it.coords(it.patch_at(p.seg0), np, tmp.data());
            for (int c = 0; c < 4; ++c)
                std::memcpy(enc_coord_host_ + c * n_patch + at, tmp.data() + c * np,
                            (size_t)np * sizeof(int32_t));
            at += np;
        }
    }
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadStream st = ranks_[r].stream;
        if (rad_memcpy_async(enc_pix_dev_[r], enc_pix_host_,
                             n_patch * enc_dim * (int64_t)sizeof(uint16_t), st) < 0 ||
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
        B.enc = (int32_t)enc.modalities;
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
