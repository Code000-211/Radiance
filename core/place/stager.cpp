/* stager.cpp -- see stager.h. */
#include "place/stager.h"

#include <algorithm>

namespace rad {

namespace {

/* The units `take` accepts, grouped by layer, in first-op order. Sets `*clash` to the first layer
 * whose span starts inside the one before it, or 0 when no spans interleave. */
template <class Take>
std::vector<StagedLayer> routed_layers(const Plan& plan, Take take, size_t* clash) {
    std::map<int32_t, StagedLayer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.layer < 0 || mu.first_use_op < 0 || mu.last_use_op < 0 || !take(mu)) continue;
        StagedLayer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    std::vector<StagedLayer> out;
    for (auto& kv : by_layer) out.push_back(std::move(kv.second));
    std::sort(out.begin(), out.end(),
              [](const StagedLayer& a, const StagedLayer& b) { return a.first_op < b.first_op; });
    *clash = 0;
    for (size_t i = 1; i < out.size() && *clash == 0; ++i)
        if (out[i].first_op <= out[i - 1].last_op) *clash = i;
    return out;
}

}  /* namespace */

PrefillStager::~PrefillStager() {
    for (Place& p : place_) {
        if (p.done) rad_event_destroy(p.done);
        if (p.free) rad_event_destroy(p.free);
    }
    for (RadEvent& g : gate_)
        if (g) rad_event_destroy(g);
    if (moved_t_) rad_event_destroy(moved_t_);
    if (moved_c_) rad_event_destroy(moved_c_);
}

int PrefillStager::init(const Program& prog, const Plan& plan, Mover* mover, char* region,
                        char* low, int64_t bytes, int64_t min_rows) {
    layers_.clear();
    if (!mover || !region || bytes <= 0) return RAD_OK;

    /* A routed layer is its expert units; its op span is their declared uses. The first op must
     * find the copies; after the last (once the block's last pass of rows issued it) nothing reads
     * them. A draft head's layer, numbered past the trunk's, is excluded (see stager.h). */
    const int32_t trunk = prog.drafter.kind != RAD_DRAFT_NONE && prog.meta.n_layers > 0
                              ? (int32_t)prog.meta.n_layers : INT32_MAX;
    size_t c = 0;
    layers_ = routed_layers(
        plan, [&](const MoveUnit& mu) { return mu.expert >= 0 && mu.layer < trunk; }, &c);
    /* Two layers' spans must not overlap: a layer's copies are given back at the next layer's
     * first op, which must be past every op that reads them. */
    if (c > 0) {
        RAD_WARN("prefill staging: routed layers %d and %d are read by interleaved ops (%d..%d "
                 "and %d..%d); staging is off", layers_[c - 1].layer, layers_[c].layer,
                 layers_[c - 1].first_op, layers_[c - 1].last_op, layers_[c].first_op,
                 layers_[c].last_op);
        layers_.clear();
        return RAD_OK;
    }
    if (layers_.empty()) return RAD_OK;

    first_of_.assign(prog.ops.size(), -1);
    last_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i) {
        if ((size_t)layers_[i].first_op < first_of_.size()) first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
        if ((size_t)layers_[i].last_op < last_of_.size()) last_of_[(size_t)layers_[i].last_op] = (int32_t)i;
    }

    mover_ = mover;
    region_ = region;
    low_ = low ? low : region;
    base_ = region;
    want_ = nullptr;
    rebase_ = false;
    ring_ = bytes / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT * 2;
    head_ = 0;
    min_rows_ = min_rows;
    n_early_ = 0;
    ring_order_.clear();
    spare_.clear();
    for (int i = kPlaces - 1; i >= 0; --i) {
        Place& p = place_[i];
        if (rad_event_create_local(&p.done) < 0 || rad_event_create_local(&p.free) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        spare_.push_back(i);
    }
    for (int i = 0; i < kGates; ++i) {
        if (rad_event_create_local(&gate_[i]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        gate_li_[i] = -1;
    }
    /* The move's two waits, only where the ring has a second place to move to (rebase). */
    if (low_ != region_ &&
        ((!moved_t_ && rad_event_create_local(&moved_t_) < 0) ||
         (!moved_c_ && rad_event_create_local(&moved_c_) < 0))) {
        layers_.clear();
        return RAD_E_DEVICE;
    }
    /* An empty list makes the backend open the card's DMA queue now, at bring-up, rather than in
     * the first long prompt's first step. */
    return rad_memcpy_list_async(nullptr, 0, 0, mover_->transfer_stream());
}

/* Arms only for a trunk pass whose level has a ring and with at least min_rows tokens (see
 * stager.h). Layers the previous pass staged early are kept if it arms with the ring where they
 * are, and withdrawn if not. */
bool PrefillStager::begin_pass(const RadBatch* b) {
    if (!b || b->draft_pass != 0) return false;
    armed_ = want_ != nullptr && enabled() && b->n_tok >= min_rows_;
    staged_ = armed_;
    if (!armed_) {
        withdraw_early();
        return false;
    }
    if (want_ != base_) {
        withdraw_early();
        rebase_ = true;
    }
    first_op_ = true;
    wrapped_ = false;
    next_ = 0;
    first_li_ = -1;
    for (int32_t& g : gate_li_) g = -1;
    return true;
}

/* Withdraw the early layers' copies. Nothing has read them, so there is no `free` to record. */
void PrefillStager::withdraw_early() {
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired) retire(p);
    }
    n_early_ = 0;
}

/* The host half of giving a place back: its units are published at their pool slots again, for
 * the next residency sync. The device half is `free`, which the copy that laps the place waits
 * on. */
void PrefillStager::retire(Place& p) {
    mover_->unstage_units(p.staged);
    p.staged.clear();
    p.retired = true;
}

/* The oldest place leaves the ring; the mover's stream waits on its last reader, so later copies
 * into it are ordered behind that read. */
int PrefillStager::recycle_front() {
    const int32_t i = ring_order_.front();
    Place& p = place_[i];
    if (p.free_rec) RAD_TRY(rad_event_wait(mover_->transfer_stream(), p.free));
    p.li = -1;
    p.free_rec = p.read = p.retired = p.early = false;
    p.free_on = nullptr;
    ring_order_.pop_front();
    spare_.push_back(i);
    return RAD_OK;
}

/* Moves the ring (see stager.h) at the first op of a pass whose ring is not where the last armed
 * pass left it. Nothing in it is published any more (begin_pass withdrew it), so it starts over,
 * empty, at its new place. Either place is the other level's buffers, hence two waits: the compute
 * stream does not reach the new step before the copies still going into the old place land, and
 * the first copy into the new place waits for the compute work already queued, which used it. */
int PrefillStager::rebase(RadStream s) {
    rebase_ = false;
    const RadStream t = mover_->transfer_stream();
    RAD_TRY(rad_event_record(moved_t_, t));
    RAD_TRY(rad_event_wait(s, moved_t_));
    RAD_TRY(rad_event_record(moved_c_, s));
    RAD_TRY(rad_event_wait(t, moved_c_));
    while (!ring_order_.empty()) RAD_TRY(recycle_front());
    head_ = 0;
    base_ = want_;
    return RAD_OK;
}

/* A place of `want` bytes at the head of the ring or, with `partial`, all the room the head can
 * reach when `want` does not fit. The head may only overwrite retired places: it stays within one
 * ring of the oldest place still published. *out is -1 when nothing fits. */
int PrefillStager::take(int64_t want, bool partial, int32_t* out) {
    *out = -1;
    int64_t limit = head_ + ring_;
    for (int32_t i : ring_order_)
        if (!place_[i].retired) { limit = place_[i].beg + ring_; break; }
    int64_t len = std::min(want, limit - head_);
    if (len < want && !partial) return RAD_OK;
    if (len < RAD_ALIGN_UNIT) return RAD_OK;
    const int64_t end = head_ + len;
    /* Every place the new one laps leaves the ring first; then, if the table is full, the oldest
     * retired ones (their waits are on reads long finished). */
    while (!ring_order_.empty() && place_[ring_order_.front()].beg + ring_ < end)
        RAD_TRY(recycle_front());
    while (spare_.empty() && !ring_order_.empty() && place_[ring_order_.front()].retired)
        RAD_TRY(recycle_front());
    if (spare_.empty()) return RAD_OK;
    const int32_t i = spare_.back();
    spare_.pop_back();
    Place& p = place_[i];
    p.beg = head_;
    p.end = end;
    *out = i;
    return RAD_OK;
}

/* Stage one layer into the ring, behind `gate` when not null. The layer the run reaches next
 * (`now`) takes whatever room there is up to its need; any other goes only whole and within half
 * the ring, so the next layer always has at least half the ring. `early` is a layer staged for the
 * next trunk pass, which end_pass leaves published. *got is kStaged, kNone when nothing of the
 * layer is pooled, or kLater when the ring has no room yet. */
int PrefillStager::stage_layer(int32_t li, bool now, bool early, RadEvent gate, int* got) {
    *got = kLater;
    const Layer& L = layers_[(size_t)li];
    int64_t unit = 0;
    const int64_t need = mover_->stage_need(L.units, &unit);
    if (need <= 0) { *got = kNone; return RAD_OK; }
    if (!now && need > ring_ / 2) return RAD_OK;
    /* A place that wraps past the region's end is two stretches, and the first may end with less
     * room than a unit, so it asks for one unit extra. */
    const int64_t want = head_ % ring_ + need > ring_ ? need + unit : need;
    int32_t i = -1;
    RAD_TRY(take(want, now, &i));
    if (i < 0) return RAD_OK;
    Place& p = place_[i];
    p.li = li;
    p.early = early;
    if (gate) RAD_TRY(rad_event_wait(mover_->transfer_stream(), gate));
    /* A wrapping place continues at the region's start. */
    const int64_t off = p.beg % ring_, len = p.end - p.beg;
    Mover::StageSpan sp[2] = { { base_ + off, std::min(len, ring_ - off) },
                               { base_, len - std::min(len, ring_ - off) } };
    RAD_TRY(mover_->stage_units(L.units, sp, sp[1].cap > 0 ? 2 : 1, nullptr, p.done, &p.staged));
    if (p.staged.empty()) {
        p.li = -1;
        p.early = false;
        spare_.push_back(i);
        *got = kNone;
        return RAD_OK;
    }
    head_ = p.end;
    ring_order_.push_back(i);
    *got = kStaged;
    return RAD_OK;
}

/* Stage the layers after `cur` as far as the pace and the ring allow. A layer with nothing pooled
 * is skipped; one without room yet, or whose pace layer the run has not reached, stops the walk
 * until a later op. */
int PrefillStager::stage_ahead(int32_t cur, bool* resync) {
    while ((size_t)next_ < layers_.size()) {
        const int32_t g = next_ - kLead;
        RadEvent gate = nullptr;
        if (g >= first_li_) {
            if (gate_li_[g % kGates] != g) break;
            gate = gate_[g % kGates];
        }
        int got = kLater;
        RAD_TRY(stage_layer(next_, next_ <= cur + 1, /*early=*/false, gate, &got));
        if (got == kLater) break;
        *resync = *resync || got == kStaged;
        ++next_;
    }
    return RAD_OK;
}

int PrefillStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;

    /* At the pass's first op of any kind, stage its first routed layer and the ones after it as
     * far as the pace and ring allow. Layers the previous pass staged early are adopted here. */
    if (first_op_) {
        first_op_ = false;
        if (rebase_) RAD_TRY(rebase(s));
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            next_ = (int32_t)(it - layers_.begin());
            first_li_ = next_;
            if (n_early_ > 0 && next_ != 0) {
                withdraw_early();
                *resync = true;
            }
            next_ = std::max(next_, n_early_);
            n_early_ = 0;
            /* The places staged for this pass are this pass's now. Left `early`, the wrap and
             * end_pass would skip them for good, and a pass with as many routed layers as lead()
             * has no later layer to give them back at: the ring would fill and staging stop
             * without a word. */
            for (int32_t i : ring_order_) place_[i].early = false;
            RAD_TRY(stage_ahead(first_li_ - 1, resync));
        }
    }

    /* The first op past this pass's last routed layer is the earliest point that proves the chunk
     * has read everything the ring holds, so the next chunk's first layers go out here rather than
     * at the pass's end (stager.h). */
    if (next_prefill_ && !wrapped_ && (int32_t)h > layers_.back().last_op)
        RAD_TRY(wrap_early(s, resync));

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    /* The run's arrival at this layer paces the copies kLead layers on. */
    RAD_TRY(rad_event_record(gate_[li % kGates], s));
    gate_li_[li % kGates] = li;
    /* This layer's copies were published when issued; this one wait orders its ops behind them. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li == li) RAD_TRY(rad_event_wait(s, p.done));
    }
    /* Earlier layers are done: every pass of rows issued their last op and recorded `free`, and
     * layers do not interleave. They are retired here, not at their last op, which may repeat for
     * another pass of rows. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li < li && p.read) retire(p);
    }
    /* Stage further ahead into the freed room. */
    if (next_ <= li) next_ = li + 1;
    const bool was = *resync;
    RAD_TRY(stage_ahead(li, resync));
    *resync = *resync || was;
    return RAD_OK;
}

/* A layer's last op: record `free` behind it so a lapping copy can start as early as possible.
 * The op may repeat for another pass of rows, moving `free` again, which is why the copies are
 * retired only when the next layer starts. */
int PrefillStager::after_op(rad_op h, RadStream s) {
    if ((size_t)h >= last_of_.size()) return RAD_OK;
    const int32_t li = last_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired || p.li != li) continue;
        RAD_TRY(rad_event_record(p.free, s));
        p.free_rec = true;
        p.read = true;
        p.free_on = s;
    }
    return RAD_OK;
}

/* The wrap (see stager.h). The op about to be issued is, by stream order, behind every op issued on
 * this stream before it, so a place whose last read was recorded here is behind that read: give it
 * back now, so a copy that laps it waits on the read and not on the end of the pass, and so the
 * next chunk's bytes start crossing while this pass still has its tail to run. A place last read on
 * another lane stays published: only the join before end_pass says what is behind it. */
int PrefillStager::wrap_early(RadStream s, bool* resync) {
    wrapped_ = true;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired || !p.read || p.early || p.free_on != s) continue;
        retire(p);
        *resync = true;
    }
    return stage_early(resync);
}

/* The next trunk pass's first layers, layers [n_early_, kLead), paced on this pass's layer
 * n - kLead + k as if the two passes were one. Stops at the first layer whose pace layer the run
 * never reached or whose room the ring has not freed yet; end_pass calls this again after it has
 * retired the pass. A layer with nothing pooled counts as settled either way: the next pass has
 * nothing of it to wait for. */
int PrefillStager::stage_early(bool* resync) {
    const int32_t n = (int32_t)layers_.size();
    while (n_early_ < kLead && n_early_ < n) {
        const int32_t g = n - kLead + n_early_;
        if (g < first_li_ || gate_li_[g % kGates] != g) break;
        int got = kLater;
        RAD_TRY(stage_layer(n_early_, /*now=*/false, /*early=*/true, gate_[g % kGates], &got));
        if (got == kLater) break;
        if (got == kStaged && resync) *resync = true;
        ++n_early_;
    }
    return RAD_OK;
}

/* Retire everything this pass staged, except what it staged ahead for the next one. Then finish
 * the wrap: whatever its pacing or the ring's room left short goes out here, now that the places
 * are back.
 *
 * The `free` of a place already read is recorded again, behind the pass's last op. That is later
 * than the read it stands for, and the wrap is what avoids paying for it; here it is the point,
 * because this stream is behind the lane join, so a place whose last reader ran on the other lane
 * is behind everything this records over. */
int PrefillStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    int rc = RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired || p.early) continue;
        const int r = rad_event_record(p.free, s);
        if (r < 0 && rc >= 0) rc = r;
        p.free_rec = r >= 0 || p.free_rec;
        retire(p);
    }
    if (rc < 0 || !next_prefill_) return rc;
    return stage_early(nullptr);
}

/* ================================================================== the file stager */
FileStager::~FileStager() {
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
}

int FileStager::init(const Program& prog, const Plan& plan, Mover* mover) {
    layers_.clear();
    if (!mover) return RAD_OK;

    /* A unit declared and never read has no op span and is left out. */
    size_t c = 0;
    layers_ = routed_layers(plan, [](const MoveUnit& mu) {
        return mu.tier == Tier::SSD && mu.access == RAD_ACCESS_CONDITIONAL;
    }, &c);
    if (layers_.empty()) return RAD_OK;
    /* A layer's copies stay published until the next one starts, so interleaved layers would need
     * a third buffer for the read-ahead; with no fallback address, such a plan is refused. */
    if (c > 0) {
        RAD_ERR("file stager: routed layers %d and %d are read by interleaved ops (%d..%d and "
                "%d..%d), and a routed layer read from the container is published one layer at a "
                "time", layers_[c - 1].layer, layers_[c].layer, layers_[c - 1].first_op,
                layers_[c - 1].last_op, layers_[c].first_op, layers_[c].last_op);
        layers_.clear();
        return RAD_E_UNSUPPORTED;
    }

    buf_[0] = mover->file_buffer(0);
    buf_[1] = mover->file_buffer(1);
    cap_ = mover->file_buffer_bytes();
    if (!buf_[0] || !buf_[1] || cap_ < plan.file_layer_max) {
        RAD_ERR("file stager: the plan reads %zu routed layers from the container and the mover "
                "holds no buffers for them", layers_.size());
        layers_.clear();
        return RAD_E_STATE;
    }
    first_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i)
        if ((size_t)layers_[i].first_op < first_of_.size())
            first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
    mover_ = mover;
    for (int b = 0; b < 2; ++b)
        if (rad_event_create_local(&done_[b]) < 0 || rad_event_create_local(&free_[b]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
    return RAD_OK;
}

/* Arms for every pass: a file-tier unit has no other address, so an unarmed pass would reach its
 * first routed layer with null expert pointers. */
bool FileStager::begin_pass(const RadBatch* b) {
    armed_ = enabled() && b;
    if (!armed_) return false;
    holds_[0] = holds_[1] = -1;
    cur_ = -1;
    first_op_ = true;
    pass_kind_ = b->draft_pass;
    last_issued_ = -1;
    return true;
}

bool FileStager::will_issue(int32_t li) const {
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int FileStager::start(int32_t li) {
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_E_STATE;     /* released before the next read, by construction */
    const int rc = mover_->stage_file_units(layers_[(size_t)li].units, buf_[b], cap_,
                                            free_rec_[b] ? free_[b] : nullptr, done_[b],
                                            &staged_[b]);
    if (rc < 0) return rc;
    if (!staged_[b].empty()) holds_[b] = li;
    return RAD_OK;
}

int FileStager::finish(int32_t b, RadStream s) {
    RAD_TRY(rad_event_record(free_[b], s));
    free_rec_[b] = true;
    mover_->unstage_units(staged_[b]);
    staged_[b].clear();
    holds_[b] = -1;
    return RAD_OK;
}

int FileStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;
    if ((int32_t)h > last_issued_) last_issued_ = (int32_t)h;

    /* Read the pass's first routed layer at its first op of any kind. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (will_issue(li)) {
                RAD_TRY(start(li));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0 || li == cur_) return RAD_OK;

    /* Entering layer `li`: the layer before is behind every op issued so far on this stream. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0 && holds_[b] != li) { RAD_TRY(finish(b, s)); *resync = true; }
    /* A layer the read-ahead skipped (the learned bound said the pass ends before it) is read
     * here rather than issued without addresses. */
    if (holds_[li & 1] != li) {
        RAD_TRY(start(li));
        *resync = true;
    }
    if (holds_[li & 1] == li) RAD_TRY(rad_event_wait(s, done_[li & 1]));
    cur_ = li;

    const int32_t next = li + 1;
    if ((size_t)next < layers_.size() && will_issue(next)) {
        RAD_TRY(start(next));
        *resync = true;
    }
    return RAD_OK;
}

int FileStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    pass_last_[pass_kind_] = last_issued_;
    int rc = RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    cur_ = -1;
    return rc;
}

}  /* namespace rad */
