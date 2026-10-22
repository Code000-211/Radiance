/* stager.h -- the prefill stager: routed layers' non-resident experts copied onto the card ahead
 * of the layers that read them.
 *
 * ---- why prefill needs it ---------------------------------------------------------------------
 *
 * An expert the slab has no room for lives in the pinned host pool, and a GEMM that routes to it
 * reads it across the link in place (Site::DeviceZeroCopy). That suits decode, which reads a few
 * experts per step from the host. A prefill chunk routes to nearly every expert of every layer, so
 * each layer's whole pooled part (~85 MB per card on Qwen3.8-Flash-Next) crosses the link inside
 * its two GEMMs, on the critical path, while the rest of the layer leaves the link idle. With
 * r4d_selftest --perfmoe, a 2048-token gate_up GEMM takes 1.53 ms with every expert on the card
 * and 4.34 ms with the coldest 13% in host memory.
 *
 * So for large enough steps the stager copies the pooled experts of the layers ahead into a VRAM
 * ring while earlier layers run, publishes the copies for each layer's ops, and publishes the host
 * addresses again once those ops are issued. The bytes are the same, so output is bit-identical.
 *
 * ---- the region --------------------------------------------------------------------------------
 *
 * One region, two layers' share of the experts the budget leaves off the card, at the top of the
 * activation arena (Program::arena_stage_off, sized by vram_budget_resolve). A step smaller than
 * the arena's full plan lends that top to the expert slab, so the region belongs to the stager
 * only in full-plan steps (the ones worth staging for). The scheduler says which, per pass
 * (set_ring).
 *
 * Under --prefill-span the ring has a second place. The full plan is sized for a spanned step,
 * and a step of one interval (a chunk that shares its step, a short prompt) runs at the level of
 * that size (Engine::probe_arena_levels). That level keeps the ring right above its own buffers,
 * so everything over it, the span's extra rows included, is lent to the slab while such steps
 * run, and the step is still staged. A pass whose ring is in the other place than the last one's
 * moves it first (rebase).
 *
 * ---- why a ring, not two buffers ---------------------------------------------------------------
 *
 * Layers leave unequal parts off the card: at depth, with the KV cache's loan taken back, the last
 * third of a Qwen3.8-Flash-Next stack pools about twice the median layer. With two fixed buffers,
 * a light layer's copy finishes early and the engine idles, and a heavy layer's copy delays every
 * layer after it. The ring packs each layer at its own size, so copies run ahead through light
 * layers and heavy ones start early. A layer staged beyond the next one takes at most half the
 * ring, so the next layer always has the room two buffers would give it.
 *
 * ---- order and safety --------------------------------------------------------------------------
 *
 *   the pass's first op     stage the first routed layer the pass issues, and the ones after it
 *                           as far as the pace and the ring allow
 *   a layer's first op      wait on its copies; publish the host addresses of earlier layers
 *                           again; stage further layers into the freed room
 *   a layer's last op       record "free" behind it on the op's stream
 *   the first op past the   give back the places that read is behind, and stage the first layers
 *   last routed layer       of the next trunk pass (the wrap, below)
 *   the pass's end          the same for anything still staged, and the wrap's leftovers
 *
 * A layer's place in the ring is counted in bytes handed out since bring-up, so a copy finds the
 * places it overwrites from the count alone. Before a copy is issued, the mover's stream waits on
 * the "free" event of every earlier place it laps (recorded behind that place's last reader, in
 * this pass or an earlier one). The layer's ops wait on the copies' event through one stream wait
 * before its first op: copies are published settled (see Mover::stage_units), and only the
 * layer's own ops resolve them.
 *
 * ---- how far ahead -----------------------------------------------------------------------------
 *
 * A layer's copies are released once the compute stream reaches the MoE of the layer kLead before
 * it (an event recorded at that layer's first op), not as soon as the ring has room: copies that
 * land long before use slow the layers they overlap. On Qwen3.8-Flash-Next TP2, unpaced copies
 * cost a 26K-token prompt 0.6% and one layer ahead cost a 160K-token prompt 2.4% (its heavy late
 * layers waited); three ahead covered the late layers at the least cost to short prompts.
 *
 * ---- the next chunk's first layers -------------------------------------------------------------
 *
 * A chunk is issued while the chunk before it is still queued, so its first layer would otherwise
 * wait for all its own copies with the link idle. When the next step continues a prompt
 * (set_next_prefill), this pass stages the next trunk pass's first layers, paced on this pass's
 * last layers as if the passes were one, and publishes them at once. The next trunk pass adopts
 * them when armed; an unarmed trunk pass publishes their host addresses again in begin_pass, which
 * the run phase calls before updating the weight tables. A draft pass in between reads none of the
 * trunk's layers and leaves them.
 *
 * Where the pass issues an op past its last routed layer -- a prompt's tail has the gather and the
 * head, and every trunk pass of a hybrid model ends at the mixer's state read -- the wrap goes out
 * there rather than at the pass's end (wrap_early), which gives back the places that read is
 * behind. Two things follow: a copy that laps one of those places waits on the read that finished
 * it, not on the end of the pass, and the copies are in flight while the pass still has its tail
 * to run. The point has to be an op and cannot be after_op of the last routed layer's last one:
 * a step longer than the MoE block's row pass issues that op again, and only the next op says the
 * repeats are over. Staged there on a gate a later repeat moves, the copies would land in the
 * middle of this chunk, where the file header's pacing says they cost the layers they overlap; and
 * a place could only be given back once its last reader was provably behind, which after_op of an
 * op that may repeat cannot say. A place whose read was recorded on another lane is left for
 * end_pass, whose stream is behind the join. With no op past the last routed layer, or with a
 * place that lane has still to give back, end_pass stages the same layers, and finishes whatever
 * the wrap's pacing or the ring's room left short.
 *
 * What the wrap does not cross is a decode step between two chunks, whether --prefill-interleave
 * or another sequence's step: set_next_prefill is false for the chunk before it, and the decode
 * pass runs at a level that lends the ring's place to the slab, so it would withdraw what the
 * chunk staged ahead. Overlap across such a step wants the region held through the small step,
 * not the copies moved earlier; that is the half of the overlap still missing (issue #7).
 *
 * ---- trunk only --------------------------------------------------------------------------------
 *
 * A draft head's routed layer is not staged. The head runs in its own passes; over a prefill chunk
 * (the history pass) it needs its experts only on a prompt's last chunk, for the few rows that
 * draft, so its pooled experts are read in place as at decode. The head's layer is numbered past
 * the trunk's (RadModelMeta::n_layers) and left out at bring-up; a draft pass is never armed.
 */
#pragma once
#include "../runtime/stager.h"
#include "mover.h"
#include "planner.h"

#include <deque>
#include <map>
#include <vector>

namespace rad {

/* One routed layer a stager serves: its units and the span of ops that read them. */
struct StagedLayer {
    int32_t              layer = -1;
    std::vector<int32_t> units;
    int32_t              first_op = -1, last_op = -1;
};

class PrefillStager final : public OpStager {
public:
    ~PrefillStager() override;

    /* Binds the routed layers of `plan` and the ops that read them, with `bytes` of device memory
     * at `region` as the ring -- or at `low`, the ring's place under the span's shared level, when
     * there is one (null otherwise). `min_rows` is the smallest pass it arms for. Leaves enabled()
     * false when no routed layer has a pooled unit to stage. */
    int init(const Program& prog, const Plan& plan, Mover* mover, char* region, char* low,
             int64_t bytes, int64_t min_rows);

    /* Per pass, before release: where the ring is for the step -- the region at the full plan,
     * `low` at the span's shared level -- or null when the step's level lends both. */
    void set_ring(char* base) { want_ = base; }
    /* The ring's two places: at the full plan, and at the span's shared level (the same one when
     * there is no such level). */
    char*   ring_full() const { return region_; }
    char*   ring_low() const { return low_; }
    /* Per step, before release: whether the next step continues a prompt in a pass this stager
     * arms for, so this step's trunk pass stages that pass's first layers. */
    void set_next_prefill(bool more) { next_prefill_ = more; }

    bool    enabled() const { return !layers_.empty(); }
    int64_t ring_bytes() const { return ring_; }
    /* The smallest pass, in tokens, the stager arms for; smaller steps read pooled experts in
     * place. The scheduler keeps a held-back prefill chunk at or above it
     * (Scheduler::set_stage_rows). */
    int64_t min_rows() const { return min_rows_; }
    size_t  n_layers() const { return layers_.size(); }
    /* Whether the last trunk pass armed (copied its pooled experts through the ring). A draft
     * pass leaves it unchanged. */
    bool    staged() const { return staged_; }
    static constexpr int32_t lead() { return kLead; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    using Layer = StagedLayer;
    std::vector<Layer>   layers_;      /* the routed layers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */
    std::vector<int32_t> last_of_;     /* by op: the layer whose last op it is, or -1 */

    /* One layer's place in the ring. [beg, end) is in bytes handed out since bring-up; the region
     * offset is beg modulo its size, and a place never straddles the region's end. */
    struct Place {
        int32_t  li = -1;                      /* index into layers_ */
        int64_t  beg = 0, end = 0;
        RadEvent done = nullptr;               /* behind its copies, on the mover stream */
        RadEvent free = nullptr;               /* behind its last reader */
        bool     free_rec = false;
        /* The stream `free` was last recorded on. A place goes back early only to a wrap issued on
         * that stream, where the op being issued now is behind that read (stager.h). */
        RadStream free_on = nullptr;
        /* Its last op has been issued at least once and `free` is recorded behind it. A step
         * longer than the MoE block's pass of rows issues a layer's ops once per pass, so the last
         * op can repeat: each repeat moves `free` behind itself, and the copies are published away
         * only at the next layer's first op. */
        bool     read = false;
        bool     retired = false;              /* its units are published at their pool slots */
        bool     early = false;                /* staged for the next trunk pass, not this one */
        std::vector<int32_t> staged;
    };
    /* More places than a two-share ring can hold at the smallest per-layer share; a full table
     * only stops copies running further ahead. */
    static constexpr int kPlaces = 16;
    Place    place_[kPlaces];
    std::deque<int32_t>  ring_order_;  /* places in the ring, oldest first; retired ones stay until lapped */
    std::vector<int32_t> spare_;       /* places not in the ring */

    Mover*   mover_ = nullptr;
    char*    region_ = nullptr;
    char*    low_ = nullptr;           /* the ring's place at the span's shared level, or null */
    char*    base_ = nullptr;          /* where the places in the ring are */
    char*    want_ = nullptr;          /* where this pass wants it (set_ring) */
    bool     rebase_ = false;          /* the pass moves the ring before its first copy */
    RadEvent moved_t_ = nullptr;       /* the move's two waits: behind the mover's stream, */
    RadEvent moved_c_ = nullptr;       /* and behind the compute stream */
    int64_t  ring_ = 0;
    int64_t  head_ = 0;                /* bytes handed out since bring-up */
    int64_t  min_rows_ = 0;

    /* The pace (see the file header): layer k's copies wait on the compute-stream event recorded
     * at the first op of layer k - kLead. A layer within kLead of the pass's first routed layer
     * goes at once. */
    static constexpr int32_t kLead = 3;
    static constexpr int kGates = 64;
    RadEvent gate_[kGates] = {};
    int32_t  gate_li_[kGates] = {};    /* the layer each gate was recorded at this pass, or -1 */
    int32_t  first_li_ = -1;           /* the pass's first routed layer */

    bool     armed_ = false;
    /* armed_ as the last trunk pass's begin_pass set it; end_pass leaves it, since the engine
     * asks after the pass has ended. */
    bool     staged_ = false;
    bool     next_prefill_ = false;
    bool     first_op_ = false;        /* the pass has issued nothing yet */
    bool     wrapped_ = false;         /* this pass has tried the wrap at its last layer (stager.h) */
    int32_t  next_ = 0;                /* the next layer to stage this pass */
    /* The next trunk pass's first layers, staged ahead of it: layers [0, n_early_). */
    int32_t  n_early_ = 0;

    enum { kStaged, kNone, kLater };
    int  stage_ahead(int32_t cur, bool* resync);
    int  stage_layer(int32_t li, bool now, bool early, RadEvent gate, int* got);
    /* The next trunk pass's first layers, layers [n_early_, lead()), paced on this pass's last
     * layers as if the two passes were one (stager.h). `resync` reports a table change to the run
     * phase, and is null from end_pass, which has no op left to resolve. */
    int  stage_early(bool* resync);
    /* The wrap: the op about to be issued is past every read of a staged layer, so give back the
     * places this stream's read is behind and stage the next pass's first layers. */
    int  wrap_early(RadStream s, bool* resync);
    void withdraw_early();
    int  take(int64_t want, bool partial, int32_t* out);
    int  recycle_front();
    int  rebase(RadStream s);
    void retire(Place& p);
};

/* ---- the file stager: routed layers read from the container -------------------------------------
 *
 * A routed unit on the file tier (--weights-disk-tier) has no address of its own, and a grouped
 * GEMM is handed every expert's pointer at issue. So on every pass, decode included, each routed
 * layer's file-tier units are read into one of the mover's two file buffers before the layer's
 * first expert op and published there, then published as null when the next routed layer starts.
 * The router's picks are not known on the host without a round trip, so the layer's whole
 * file-tier part is read; decode is then paced by the drive, which is the trade the flag makes.
 *
 *   the pass's first op      read the first routed layer the pass issues
 *   a layer's first op       release the layer before (record its buffer free on the stream, publish
 *                            null), wait on this layer's copies, and read the next layer into the
 *                            buffer just released -- its copies queue behind that free event
 *   the pass's end           release whatever is still held
 *
 * A layer is released when the next one starts, not at its own last op: a routed block that slices
 * a large step into passes of `rows` tokens issues its expert ops once per slice, so a release at
 * the last op would leave the next slice reading null. Repeats of the current layer's first op do
 * nothing.
 *
 * The read blocks the issuing thread while the card runs what was issued before it. A layer's
 * file-tier part is gigabytes against milliseconds of compute, so the step is drive-bound either
 * way; a reader thread would gain little and add a second writer of the residency table.
 *
 * A layer is read ahead only when the pass will issue it (a bound learned per draft_pass from the
 * last pass of that kind), so a trunk pass does not read the MTP head's layer. */
class FileStager final : public OpStager {
public:
    ~FileStager() override;

    /* Binds the routed layers with file-tier units and the ops that read them, over the mover's
     * two file buffers. Leaves enabled() false when no routed unit is on the file tier. Refuses a
     * plan whose routed layers are read by interleaved ops, since a file-tier unit has no other
     * address to fall back on. */
    int init(const Program& prog, const Plan& plan, Mover* mover);

    bool    enabled() const { return !layers_.empty(); }
    size_t  n_layers() const { return layers_.size(); }
    int64_t buffer_bytes() const { return cap_; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  end_pass(RadStream s) override;

private:
    using Layer = StagedLayer;
    std::vector<Layer>   layers_;      /* the routed layers with file-tier units, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */

    Mover*   mover_ = nullptr;
    char*    buf_[2] = { nullptr, nullptr };
    int64_t  cap_ = 0;
    RadEvent done_[2] = { nullptr, nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[2] = { nullptr, nullptr };   /* behind the ops that read a buffer's layer */
    bool     free_rec_[2] = { false, false };
    std::vector<int32_t> staged_[2];
    int32_t  holds_[2] = { -1, -1 };             /* the layer a buffer holds, or -1 */
    int32_t  cur_ = -1;                          /* the layer whose ops are being issued */

    bool     armed_ = false;
    bool     first_op_ = false;
    int32_t  pass_kind_ = 0;
    int32_t  last_issued_ = -1;
    std::map<int32_t, int32_t> pass_last_;   /* by draft_pass: the last op such a pass issued */

    bool will_issue(int32_t li) const;
    int  start(int32_t li);
    int  finish(int32_t b, RadStream s);
};

}  /* namespace rad */
