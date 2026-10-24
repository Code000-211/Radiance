/* accept.h -- speculative acceptance. Verification is a declared op; ACCEPTANCE IS CORE SAMPLER
 * LOGIC (spec §10), which is why it lives here beside the chain that produced the distributions
 * it compares.
 *
 * Three rules, and the cheapest one is exact everywhere.
 *
 * AGREEMENT WITH THE TARGET'S OWN TOKEN -- what accept_greedy below actually implements, and what
 * this engine uses. A drafted token is accepted when it equals the token the target produced at
 * that position. At temperature 0 that token is the argmax, which is where the name came from.
 * ABOVE ZERO IT IS THE TARGET'S SAMPLED DRAW, AND ACCEPTING ON IT IS STILL EXACT -- provided each
 * verified position draws from its own RNG coordinate, which core/sample/sampler.cpp guarantees
 * (`p.pos = n_out + k`). Then the target's draw at position k is the one its own distribution
 * specifies there, the first emitted token is that draw whether the draft agreed or not, and every
 * later position is conditioned on a prefix accepted only because it matched. The emitted sequence
 * is a draw from the target, not an approximation of one.
 *
 * Exact given the same logits -- which is not the same as byte-identical to a non-speculative run,
 * because a verify step's logits are computed at a different M than a decode step's. The text
 * matches exactly at temperature 0 and mostly above it. See Engine::collect_step.
 *
 * What it costs is acceptance RATE: P(y == x) rather than the optimal min(1, p/q). What it costs
 * in machinery is nothing, which is why the engine computes no probability planes for it.
 *
 * ACCEPTING ON ARGMAX AT A POSITIVE TEMPERATURE is the rule that is NOT exact -- it replaces the
 * target's distribution with its mode. That is a different thing from the above, and the two are
 * easy to conflate.
 *
 * REJECTION SAMPLING. At temperature > 0 the drafted token x is accepted with probability
 * min(1, p(x)/q(x)) where p is the target's filtered distribution and q the drafter's. On
 * rejection the emitted token is drawn from the residual, norm(max(0, p - q)), which is what
 * makes the whole scheme emit exactly the target's distribution. With a GREEDY DRAFTER q is a
 * point mass, so the accept probability collapses to p(x) and the residual is p with x removed
 * and the rest renormalised -- which is the case the fast path is written for.
 *
 * Worth stating plainly because it looks like a regression and is not: acceptance FALLS relative
 * to greedy-against-greedy, because a proposal the target agrees with is still only accepted with
 * probability p(x) rather than always. That is the price of emitting the target's true
 * distribution.
 *
 * WHAT THE ANSWER IS FOR. `n_accepted` is consumed by two components that do different things
 * with it. The scheduler frees paged KV blocks past the accepted position -- a slot-table edit.
 * The linear-attention kernels do NOT recompute: the recurrent state and the conv window have
 * already absorbed the rejected tokens, and libr4d handles that by treating the conv state cache
 * as a rolling window of `conv_width-1 + n_spec` entries and READING AT THE SLOT THE LAST
 * ACCEPTED TOKEN LEFT. A rejection is a change of read offset, not a recompute, and `num_accepted`
 * is a required operand of `gdn_conv_update` and `gdn_recurrent_update` for exactly that reason
 * (docs/OPS.md, spec §10).
 */
#pragma once
#include "../rad_internal.h"

#include <cstdint>

namespace radiance { class CandidateTree; }

namespace rad {

struct SpecAcceptParams {
    int      n_draft   = 0;      /* draft tokens proposed for this sequence */
    int64_t  n_vocab   = 0;      /* row stride of the probability planes */
    uint64_t seed      = 0;      /* the request's seed */
    uint64_t pos       = 0;      /* output position of draft token 0 */
    bool     greedy    = false;  /* temperature 0: argmax acceptance instead of rejection */
};

/* Acceptance by agreement, over precomputed target tokens. `target` is [n_draft+1] -- the token
 * the target model produced at each verified position, including the bonus position past the last
 * draft token, which is why the buffer is one longer than the draft. That token is the argmax at
 * temperature 0 and the target's own sampled draw above it; both are exact, for the reason at the
 * top of this file. The name describes the temperature-0 case; the rule is correct above it too.
 *
 * Writes n_accepted+1 tokens into `out` (capacity n_draft+1) and sets *n_accepted. Returns
 * RAD_OK. Always emits at least one token: the position after the last accepted one is the
 * target's own answer, which is always correct to emit. */
int accept_greedy(const int32_t* draft, int n_draft, const int32_t* target,
                  int32_t* out, int32_t* n_accepted);

/* DARTree tree-structured candidate acceptance by agreement at temperature 0.0.
 * `target_argmax` contains target model predictions following each candidate node.
 * Writes accepted draft tokens into `out_accepted` [0..n_accepted-1] and bonus token at out_accepted[n_accepted].
 * Sets *n_accepted and *out_bonus_token. Optionally writes best path node IDs into `out_best_nodes`. */
int accept_tree_greedy(const radiance::CandidateTree& tree,
                       const int32_t* target_argmax,
                       int n_target,
                       int32_t* out_accepted,
                       int32_t* n_accepted,
                       int32_t* out_bonus_token,
                       int32_t* out_best_nodes = nullptr);

/* Rejection sampling. `target_probs` is [(n_draft+1) * n_vocab], each row a NORMALISED
 * distribution as the sampler chain left it -- filtered, renormalised over the survivors, zero
 * elsewhere. `draft_probs` is [n_draft * n_vocab] on the same terms, or null for a greedy drafter
 * (a point mass on the drafted token), which is the common case and the cheap one.
 *
 * Writes n_accepted+1 tokens into `out` and sets *n_accepted. Returns RAD_OK, or RAD_E_INVAL on
 * a malformed argument. A target row that sums to zero -- everything masked out by a grammar --
 * is RAD_E_STATE: there is no token to emit and the request must fail rather than receive one. */
int accept_stochastic(const SpecAcceptParams& p, const int32_t* draft,
                      const float* target_probs, const float* draft_probs,
                      int32_t* out, int32_t* n_accepted);

/* The one a caller reaches for: dispatches on p.greedy. The greedy path needs `target_argmax`
 * (which the `sample_argmax` op already produced) and ignores the probability planes; the
 * stochastic path needs the planes and ignores the argmax. Passing the wrong one for the mode is
 * RAD_E_INVAL rather than a quiet fallback, because a verify step that silently switched
 * acceptance rules would change the output distribution and nothing would say so. */
int accept_sequence(const SpecAcceptParams& p, const int32_t* draft,
                    const int32_t* target_argmax,
                    const float* target_probs, const float* draft_probs,
                    int32_t* out, int32_t* n_accepted);

}  /* namespace rad */
