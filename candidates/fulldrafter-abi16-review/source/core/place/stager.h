/* stager.h -- the prefill stager: a routed layer's non-resident experts copied onto the card one
 * layer ahead of the layer that reads them.
 *
 * ---- WHY A PREFILL NEEDS IT -------------------------------------------------------------------
 *
 * An expert the slab has no room for lives in the pinned host pool, and a GEMM that routes to it
 * reads it across the link in place (Site::DeviceZeroCopy). At decode that is the right trade: a
 * step routes a few dozen rows to a few dozen experts and reads a few of them from the host. A
 * prefill chunk routes tens of thousands of rows to nearly EVERY expert of every layer, so the
 * whole pooled part of each layer -- ~85 MB a card on Qwen3.8-Flash-Next -- crosses the link inside
 * the layer's two GEMMs, at the link's rate and on the critical path, while the rest of the layer's
 * work leaves the link idle. Measured with r4d_selftest --perfmoe, the gate_up GEMM of a 2048-token
 * chunk is 1.53 ms with every expert on the card and 4.34 ms with the coldest 13% in host memory.
 *
 * So at a step large enough, the stager copies layer L+1's pooled experts into a VRAM buffer while
 * layer L runs, publishes the copies for layer L+1's ops, and publishes the host addresses again
 * once those ops are behind it. The copy is the same bytes the GEMM would have read, so the output
 * is bit-identical; only when the link carries them moves.
 *
 * ---- WHERE THE BUFFERS ARE, AND WHEN THEY ARE ITS ---------------------------------------------
 *
 * Two buffers, each one layer's share of the experts the budget leaves off the card, at the top of
 * the activation arena (Program::arena_stage_off, sized by vram_budget_resolve). A step smaller
 * than the arena's full plan LENDS that top to the expert slab, so the buffers are the stager's
 * only in a step at the full plan -- which is exactly the steps worth staging for -- and the rest of
 * the time they hold resident experts like any other lent slot. The scheduler says which it is,
 * pass by pass (set_full_arena).
 *
 * ---- THE ORDER, AND WHAT KEEPS IT SAFE ---------------------------------------------------------
 *
 *   the pass's first op     stage the first routed layer the pass issues, into buffer 0 or 1
 *   a layer's first op      stage the NEXT layer into the other buffer; re-read the residency
 *                           table so this layer's ops resolve the copies made one layer ago
 *   a layer's last op       record "buffer free" on the op's stream; publish the host addresses
 *   the pass's end          the same for anything still staged: nothing outlives its pass
 *
 * A copy into a buffer waits on that buffer's "free" event, recorded behind the last op that read
 * the layer it held -- in this pass or an earlier one. The layer's ops wait on the copies' event
 * through ONE stream wait, issued before the layer's first op: the copies are published settled
 * (Mover::stage_units says why), and nothing but the layer's own ops ever resolves them.
 *
 * A LAYER IS STAGED AHEAD ONLY WHEN THE PASS WILL ISSUE IT. A step issues its routed layers across
 * more than one pass -- the trunk in one, the MTP head's in the next -- and the last layer a trunk
 * pass stages ahead would otherwise be the head's, copied for a pass that never reads it and thrown
 * away at its end. The stager learns the last op each kind of pass issues (by draft_pass) and stages
 * a layer ahead only when its first op is at or below it.
 */
#pragma once
#include "../runtime/stager.h"
#include "mover.h"
#include "planner.h"

#include <map>
#include <vector>

namespace rad {

class PrefillStager final : public OpStager {
public:
    ~PrefillStager() override;

    /* The routed layers of `plan` and the ops that read them, and `bytes` of device memory at
     * `region` as two buffers. `min_rows` is the smallest pass it arms for. Leaves it disabled --
     * enabled() false -- when the plan has no routed layer with a pooled unit to stage. */
    int init(const Program& prog, const Plan& plan, Mover* mover, char* region, int64_t bytes,
             int64_t min_rows);

    /* Per pass, before it is released: whether the step runs at the arena's full plan, which is
     * when the buffers are not lent. */
    void set_full_arena(bool full) { full_ = full; }

    bool    enabled() const { return !layers_.empty(); }
    static constexpr int kMaxBufs = 4;
    int     n_bufs() const { return n_bufs_; }
    int64_t buffer_bytes() const { return cap_; }
    size_t  n_layers() const { return layers_.size(); }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    struct Layer {
        int32_t              layer = -1;
        std::vector<int32_t> units;
        int32_t              first_op = -1, last_op = -1;
    };
    std::vector<Layer>   layers_;      /* the routed layers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */
    std::vector<int32_t> last_of_;     /* by op: the layer whose last op it is, or -1 */

    Mover*   mover_ = nullptr;
    int      n_bufs_ = 2;
    char*    buf_[kMaxBufs] = { nullptr };
    int64_t  cap_ = 0;
    int64_t  min_rows_ = 0;
    RadEvent done_[kMaxBufs] = { nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[kMaxBufs] = { nullptr };   /* behind a buffer's last reader */
    bool     free_rec_[kMaxBufs] = { false };
    std::vector<int32_t> staged_[kMaxBufs];
    int32_t  holds_[kMaxBufs] = { -1, -1, -1, -1 }; /* the layer a buffer holds, or -1 */
    bool     read_[kMaxBufs] = { false };

    bool     full_ = false;
    bool     armed_ = false;
    bool     first_op_ = false;                  /* the pass has issued nothing yet */
    std::vector<uint8_t> started_;               /* by layer: staged (or tried) this pass */
    int32_t  pass_kind_ = 0;
    int32_t  last_issued_ = -1;
    std::map<int32_t, int32_t> pass_last_;       /* by draft_pass: the last op such a pass issued */

    bool will_issue(int32_t li) const;
    int  start(int32_t li);
    int  finish(int32_t b, RadStream s);
    void release(int32_t b);
};

/* ---- THE FILE STAGER: routed layers read from the container -------------------------------------
 *
 * A routed unit the plan put on the file tier (--weights-disk-tier) has no address of its own, and
 * a grouped GEMM is handed every expert's pointer at issue. So on EVERY pass -- a decode step as
 * much as a prefill chunk -- each routed layer's file-tier units are read into one of the mover's
 * two file buffers before the layer's first expert op and published there, and published as null
 * again when the next routed layer starts. Which experts the router picks cannot be known on the
 * host without a round trip, so the whole layer's file-tier part is read; a prefill chunk routes to
 * nearly all of it anyway, and a decode step is paced by the drive -- the trade the flag names.
 *
 *   the pass's first op      read the first routed layer the pass issues
 *   a layer's first op       release the layer before (record its buffer free on the stream, publish
 *                            null), wait on this layer's copies, and read the NEXT layer into the
 *                            buffer just released -- its copies queue behind that free event
 *   the pass's end           release whatever is still held
 *
 * A LAYER IS RELEASED WHEN THE NEXT ONE STARTS, not at its own last op: a routed block that slices a
 * large step into passes of `rows` tokens issues its expert ops once a slice, so the last declared
 * op of the layer comes round again, and a release there would leave the second slice reading null.
 * Repeats of the current layer's first op do nothing.
 *
 * THE READ IS ON THE ISSUING THREAD and blocks it; the card meanwhile runs whatever was issued
 * before it. A routed layer's file-tier part is gigabytes and its compute milliseconds, so the step
 * is the drive's either way, and a reader thread would buy the few percent the layer's own compute
 * is worth at the cost of a second writer of the residency table.
 *
 * A layer is read ahead only when the pass will issue it -- the same learned bound PrefillStager
 * keeps (by draft_pass), so a trunk pass does not read the MTP head's layer for nothing. */
class FileStager final : public OpStager {
public:
    ~FileStager() override;

    /* The routed layers of `plan` that hold a unit on the file tier and the ops that read them,
     * over the mover's two file buffers. Leaves it disabled -- enabled() false -- when the plan
     * puts no routed unit there; refuses a plan whose routed layers are read by interleaved ops,
     * because a file-tier unit has no other address to fall back on. */
    int init(const Program& prog, const Plan& plan, Mover* mover);

    bool    enabled() const { return !layers_.empty(); }
    size_t  n_layers() const { return layers_.size(); }
    int64_t buffer_bytes() const { return cap_; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    struct Layer {
        int32_t              layer = -1;
        std::vector<int32_t> units;
        int32_t              first_op = -1, last_op = -1;
    };
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
    std::map<int32_t, int32_t> pass_last_;

    bool will_issue(int32_t li) const;
    int  start(int32_t li);
    int  finish(int32_t b, RadStream s);
};

}  /* namespace rad */
