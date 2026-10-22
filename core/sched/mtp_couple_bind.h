/* mtp_couple_bind.h -- the engine's post-commit binding step for a serial drafting row.
 *
 * Engine::finish_serial_drafter calls this after the step's commit, once the card's proposal states
 * have been read back. It answers two questions the position contract turns on:
 *
 *   - WHICH LENGTH the next build will stage the row at. That is the request's committed output
 *     length AFTER this commit -- the value the next build_step reads -- and nothing else. It is
 *     deliberately not rebuilt from the draw's pre-commit base plus the slot counter (n_acc): the
 *     two agree only while the commit emits as many tokens as the card accepted, and a max_tokens
 *     or context cut emits fewer.
 *   - WHETHER the row may bind at all. A sequence the plan no longer names, or whose request is no
 *     longer Running, will not sample again: its rows are terminal work (a carried ahead step, a
 *     request that finished on this commit). Binding them would put a coupled verify on the card
 *     for a request that has already left the sampler, which reads as a live coupled round in any
 *     log and is not one.
 *
 * The answer comes from Scheduler::committed_seq(), which commit fills under the scheduler lock.
 * It is deliberately NOT read off the plan's StepEntry::req: after commit that Request may already
 * have been reaped (release_locked destroys it), and a serial bind that dereferenced it would read
 * freed memory.
 *
 * Here rather than in engine_draft.cpp so the rule can be driven against a real scheduler commit
 * and the sampler's own build (tests/sched_test.cpp, tests/sample_test.cpp) without an Engine.
 */
#pragma once

#include "advance.h"
#include "rad_spechub.h"

#include <cstring>
#include <string>

namespace rad {

/* Where the next build will sample this sequence, after the current step's commit. */
struct MtpCoupleBindPoint {
    bool     found = false;   /* the committed step carried the sequence */
    bool     live  = false;   /* and its request was still Running (not finished/cancelled/failed) */
    uint64_t n_out = 0;       /* its committed output length: what the next build reads as n_out */
};

/* Bind `done` rounds of this row's draw to the positions the next build samples. `rb` is round 1
 * of this row in the engine's readback (stride `rb_stride` states), `tok[rd]` the card's token for
 * round rd, `S` the card's slot. Fills `out[0, done)` and sets `*n_bound = done`; a sequence that
 * is not live binds nothing (`*n_bound = 0`, RAD_OK) and leaves `out` untouched. RAD_E_STATE with
 * `why` for a readback that is not this draw's, or is not a state the proposer could have written.
 * `base` is never read here: it is the draft's coordinate, not the verify's. */
inline int mtp_couple_bind_rounds(const MtpCoupleBindPoint& p, const RadMtpCoupleDraw& d,
                                  const RadMtpCoupleState* rb, int rb_stride, int done,
                                  const StepSlot& S, int64_t vocab, RadMtpCoupleState* out,
                                  int* n_bound, std::string* why) {
    *n_bound = 0;
    if (!p.live) return RAD_OK;
    const bool sampled = d.mode == 1;
    for (int rd = 1; rd <= done; ++rd) {
        const RadMtpCoupleState& st = rb[(size_t)(rd - 1) * (size_t)rb_stride];
        if (std::memcmp(&st.draw, &d, sizeof d) || st.round != rd || st.x != S.tok[rd] ||
            st.status != (sampled ? 1 : 0) ||
            (sampled && !rad_mtp_couple_state_valid(st, vocab))) {
            *why = "MTP coupled: sequence " + std::to_string(d.seq) + " round " +
                   std::to_string(rd) + " returned status " + std::to_string(st.status) +
                   " round " + std::to_string(st.round) + " x " + std::to_string(st.x) +
                   " for card token " + std::to_string(S.tok[rd]) + " (draw " +
                   (std::memcmp(&st.draw, &d, sizeof d) ? "stale" : "current") +
                   "); refusing the step";
            return RAD_E_STATE;
        }
        RadMtpCoupleState b = st;
        b.pos = rad_mtp_couple_verify_pos(p.n_out, rd);
        out[rd - 1] = b;
    }
    *n_bound = done;
    return RAD_OK;
}

}  // namespace rad
