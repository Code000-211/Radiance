/* steplog.cpp -- the ring behind steplog.h.
 *
 * Three functions, and the shape of all three comes from one constraint: `push` runs on the step
 * path, inside the bracket Scheduler::step_finished already takes. So the storage is allocated once
 * in `enable`, the ring never grows, and a push is a copy into a slot the writer owns. The reader
 * drains between phases, off that path.
 *
 * The ring is not a queue and nothing here blocks: when it is full the oldest record is overwritten
 * and `dropped_` counts it, so a run longer than the ring reports its distributions over what
 * survived and says how much did not (core/diagnose.h). */
#include "steplog.h"

#include <algorithm>

namespace rad {

void StepLog::enable(size_t cap) {
    /* One allocation, here, and none after: a ring sized for the run rather than grown during it.
     * Re-enabling larger keeps the records already held, because a reader that drains between
     * points never depends on the capacity. */
    if (cap == 0) cap = 1;
    if (ring_.size() < cap) ring_.resize(cap);
    head_ = std::min(head_, ring_.size());
}

void StepLog::disable() {
    ring_.clear();
    head_ = count_ = 0;
    dropped_ = 0;
}

void StepLog::push(const StepRecord& r) {
    if (ring_.empty()) return;                 /* never enabled, or switched off again */
    ring_[head_] = r;
    head_ = head_ + 1 == ring_.size() ? 0 : head_ + 1;
    if (count_ < ring_.size()) ++count_;
    else ++dropped_;                           /* the slot at head_ was the oldest unread */
}

uint64_t StepLog::drain(std::vector<StepRecord>* out) {
    out->clear();
    if (ring_.empty() || count_ == 0) {
        const uint64_t d = dropped_;
        dropped_ = 0;
        return d;
    }
    /* Oldest first: the writer's slot is one past the last record, so with the ring full the oldest
     * sits at head_, and otherwise the run started at 0. */
    const size_t n = ring_.size();
    size_t first = count_ < n ? 0 : head_;
    out->reserve(count_);
    for (size_t i = 0; i < count_; ++i) out->push_back(ring_[(first + i) % n]);
    head_ = 0;
    count_ = 0;
    const uint64_t d = dropped_;
    dropped_ = 0;
    return d;
}

}  /* namespace rad */
