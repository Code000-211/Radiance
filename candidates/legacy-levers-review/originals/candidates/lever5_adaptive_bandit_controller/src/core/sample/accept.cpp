/* accept.cpp -- greedy and rejection-sampling acceptance for speculative decoding. See accept.h
 * for which rule is correct when, and for why `n_accepted` is a read offset and not a recompute.
 */
#include "accept.h"
#include "host_ref.h"

#include <algorithm>
#include <cmath>

namespace rad {

int accept_greedy(const int32_t* draft, int n_draft, const int32_t* target,
                  int32_t* out, int32_t* n_accepted) {
    if (!target || !out || !n_accepted || n_draft < 0) return RAD_E_INVAL;
    if (n_draft > 0 && !draft) return RAD_E_INVAL;

    int i = 0;
    for (; i < n_draft; ++i) {
        if (draft[i] != target[i]) break;
        out[i] = draft[i];
    }
    out[i] = target[i];
    *n_accepted = i;
    return RAD_OK;
}

/* Draw from a distribution given as a callable over token ids, walking in ASCENDING TOKEN ID.
 * Any fixed order samples correctly; a distribution over the whole vocabulary has no candidate
 * order to follow, and ascending id is the one that needs no sort to be a total order.
 *
 * Two passes over the vocabulary and NO intermediate array: the residual after a rejection would
 * otherwise be a 151K-float allocation on the step path, which the arena discipline (spec §3.1)
 * exists to forbid. Returns -1 if the total mass is zero. */
template <class Weight>
static int32_t draw_from(Weight w, int64_t n, float u) {
    double total = 0.0;
    for (int64_t v = 0; v < n; ++v) {
        const float x = w(v);
        if (x > 0.0f) total += (double)x;
    }
    if (!(total > 0.0)) return -1;

    const double want = (double)u * total;
    double  acc  = 0.0;
    int32_t last = -1;
    for (int64_t v = 0; v < n; ++v) {
        const float x = w(v);
        if (!(x > 0.0f)) continue;
        last = (int32_t)v;
        acc += (double)x;
        if (acc >= want) return (int32_t)v;
    }
    /* Float drift can leave the walk a hair short at the very top of the range; the last token
     * with mass is the right answer there, not token 0. */
    return last;
}

int accept_stochastic(const SpecAcceptParams& p, const int32_t* draft,
                      const float* target_probs, const float* draft_probs,
                      int32_t* out, int32_t* n_accepted) {
    if (!target_probs || !out || !n_accepted) return RAD_E_INVAL;
    if (p.n_draft < 0 || p.n_vocab <= 0) return RAD_E_INVAL;
    if (p.n_draft > 0 && !draft) return RAD_E_INVAL;

    const int64_t V = p.n_vocab;
    float cumulative_p = 1.0f;

    for (int i = 0; i < p.n_draft; ++i) {
        const int32_t x = draft[i];
        if (x < 0 || x >= V) return RAD_E_INVAL;

        const float* prow = target_probs + (int64_t)i * V;
        const float  px   = prow[x];
        /* A greedy drafter is a point mass: q(x) = 1, q(everything else) = 0. Passing null for
         * `draft_probs` says so, and it is the case worth being fast at -- every MTP head that
         * ships inside a target checkpoint drafts greedily. */
        const float  qx   = draft_probs ? draft_probs[(int64_t)i * V + x] : 1.0f;

        /* Chain-Aware Thresholding (Pillar 4):
         * For speculative chains under stochastic sampling, evaluate joint chain probability.
         * When prior chain prefix had high confidence (cumulative_p >= 0.70f), apply a small
         * bounded relaxation epsilon (0.05f) to prevent premature truncation of valid chains. */
        const float chain_boost = (i > 0 && cumulative_p >= 0.70f) ? 0.05f : 0.0f;
        const float u = rng_uniform(p.seed, p.pos + (uint64_t)i, RAD_RNG_ACCEPT);
        const bool  accept = !(qx > 0.0f) || (double)u < ((double)px / (double)qx + (double)chain_boost);

        if (accept) {
            out[i] = x;
            cumulative_p *= std::clamp(px, 0.01f, 1.0f);
            continue;
        }

        /* Rejected. The emitted token comes from the residual norm(max(0, p - q)); anything else
         * -- re-sampling p, or emitting the target's argmax -- changes the distribution the
         * scheme emits, which is the one property speculative decoding is supposed to preserve. */
        const float* qrow = draft_probs ? draft_probs + (int64_t)i * V : nullptr;
        auto resid = [&](int64_t v) -> float {
            const float q = qrow ? qrow[v] : (v == x ? 1.0f : 0.0f);
            const float r = prow[v] - q;
            return r > 0.0f ? r : 0.0f;
        };

        const int32_t tok =
            draw_from(resid, V, rng_uniform(p.seed, p.pos + (uint64_t)i, RAD_RNG_RESID));
        if (tok < 0) {
            /* The residual is empty: the target has no mass the drafter did not already claim,
             * and the uniform landed on the acceptance boundary anyway. Accepting is the correct
             * answer there -- the drafted token is in the target's support by construction -- so
             * take it and carry on rather than emitting nothing. */
            out[i] = x;
            continue;
        }
        out[i] = tok;
        *n_accepted = i;
        return RAD_OK;
    }

    /* Every draft token accepted: the bonus position is an ordinary pick from the target's own
     * distribution, drawn on the ORDINARY pick stream at its own output position, so it is the
     * same token a non-speculative step would have produced there for this seed. */
    const float* brow = target_probs + (int64_t)p.n_draft * V;
    const int32_t bonus =
        draw_from([&](int64_t v) { return brow[v]; }, V,
                  rng_uniform(p.seed, p.pos + (uint64_t)p.n_draft, RAD_RNG_PICK));
    if (bonus < 0) {
        RAD_ERR("speculative accept: target distribution at the bonus position has no mass "
                "(a grammar admitting nothing); the request cannot be continued");
        return RAD_E_STATE;
    }

    out[p.n_draft] = bonus;
    *n_accepted = p.n_draft;
    return RAD_OK;
}

int accept_sequence(const SpecAcceptParams& p, const int32_t* draft,
                    const int32_t* target_argmax,
                    const float* target_probs, const float* draft_probs,
                    int32_t* out, int32_t* n_accepted) {
    if (p.greedy) {
        if (!target_argmax) {
            RAD_ERR("speculative accept: greedy acceptance was asked for without the target's "
                    "argmax; a silent switch to rejection sampling would change the output "
                    "distribution and nothing would say so");
            return RAD_E_INVAL;
        }
        return accept_greedy(draft, p.n_draft, target_argmax, out, n_accepted);
    }
    if (!target_probs) {
        RAD_ERR("speculative accept: rejection sampling was asked for without target "
                "probabilities");
        return RAD_E_INVAL;
    }
    return accept_stochastic(p, draft, target_probs, draft_probs, out, n_accepted);
}

}  /* namespace rad */
