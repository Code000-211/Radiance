/* draft_confidence.h -- the head's own confidence in each draft it published, joined with what the
 * verify then made of that draft (RADIANCE_DRAFT_CONFIDENCE, a diagnostic).
 *
 * The question it answers is whether a verify window could be cut by the draft's own probability:
 * verify rows are most of what a deeper window costs on a mixture-of-experts target, the serial
 * draft rounds little. Under RADIANCE_MTP_COUPLED every round leaves Q -- the FP64 softmax of the
 * head's reranked top eight -- and the token x it drew from Q in its RadMtpCoupleState, so q(x) is
 * already on the host when the engine publishes the round. Nothing here changes a plan or a draft:
 * the scheduler only remembers q per published draft and, at the commit of the step that verified
 * that publication, writes one record per window. A greedy draw (status 0) carries no Q and is not
 * recorded, nor is a prompt-lookup window.
 */
#pragma once

#include "rad_spechub.h"

#include <cstdint>

namespace rad {

/* q(x) and the largest q of one round's state. False for a state that is not a valid sampled draw
 * (rad_mtp_couple_state_valid): no Q to read. */
inline bool mtp_draft_confidence(const RadMtpCoupleState& s, int64_t vocab, float* q_x,
                                 float* q_max) {
    if (!rad_mtp_couple_state_valid(s, vocab)) return false;
    double qx = 0.0, qm = 0.0;
    for (int i = 0; i < s.count; ++i) {
        if (s.ids[i] == s.x) qx = s.q[i];
        if (s.q[i] > qm) qm = s.q[i];
    }
    *q_x = (float)qx;
    *q_max = (float)qm;
    return true;
}

/* One verified head window: the step's row count, the drafts the target verified, the drafts it
 * committed, the raw match (above `accepted` when the end-of-generation cut stopped a matching
 * run), and q(x) and max q of each verified draft in order. A coupled head runs at most
 * RAD_MTP_COUPLE_MAX_ROUNDS rounds, so that bounds a window that has a q for every draft. */
struct DraftConfRecord {
    uint8_t rows = 0;
    uint8_t verified = 0;
    uint8_t accepted = 0;
    uint8_t raw = 0;
    float   q[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
    float   q_max[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
};

}  // namespace rad
