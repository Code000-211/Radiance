/* mtp_couple_evidence.h -- what the completed-verify line may claim.
 *
 * The line is evidence only if all of it holds: the build that staged bound rows belongs to the
 * step whose wait just returned, at least one of its sequences is still live at that point, and
 * the op that ran is a DEVICE op. Each of these was wrong in the first version of the line:
 *
 *   - It gated on Sampler::staged_any_couple(), which the drafter's build_greedy had already
 *     cleared between the build and the wait -- so the line never appeared in the coupled path.
 *     It reads Sampler::couple_build() now, the snapshot build_step froze.
 *   - It listed the rows of whatever build was last staged. The snapshot carries its step, and
 *     the line is written only for the step being collected.
 *   - It printed a kernel name without the domain, so a host reference row read like a device
 *     verify. The domain is decided here and printed on the line.
 *   - It listed sequences that were live at build time. `live` is asked after the wait (under the
 *     scheduler's lock), and only those rows count.
 *
 * A pure function of the snapshot, a liveness predicate and the op's domain, so the decision can
 * be driven without an Engine (tests/sample_test.cpp).
 */
#pragma once

#include "sampler.h"

#include <string>

namespace rad {

struct MtpCoupleEvidence {
    int64_t     rows   = 0;     /* bound rows of sequences still live after the wait */
    bool        device = false; /* the op that ran is a device op */
    std::string seqs;           /* those sequences, comma separated, in build order */
};

/* `live(seq)` answers whether the sequence is still Running; the engine passes a predicate over
 * Scheduler::running(), the tests a set. Empty (rows == 0) when the snapshot is not this step's,
 * or nothing it staged is still live. */
template <class Live>
MtpCoupleEvidence mtp_couple_evidence(const Sampler::CoupleBuild& b, int64_t step, int domain,
                                      Live&& live) {
    MtpCoupleEvidence e;
    if (b.step != step) return e;
    e.device = domain == RAD_DOMAIN_DEVICE;
    for (const auto& s : b.seqs)
        if (live(s.first)) {
            e.rows += s.second;
            if (!e.seqs.empty()) e.seqs += ",";
            e.seqs += std::to_string(s.first);
        }
    return e;
}

}  // namespace rad
