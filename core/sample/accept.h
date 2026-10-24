/* accept.h -- speculative acceptance. Verification is a declared op; acceptance is core sampler
 * logic (spec §10), so it lives beside the chain that produced the distributions it compares.
 *
 * Agreement with the target's own token. A drafted token is accepted when it equals the token the
 * target produced at that position: the argmax at temperature 0, the target's sampled draw above
 * it. Both are exact, provided each verified position draws from its own RNG coordinate
 * (`p.pos = n_out + k` in sampler.cpp): the first emitted token is the target's draw whether the
 * draft agreed or not, and every later position is conditioned on a prefix accepted only because
 * it matched. Exact given the same logits; a verify step computes logits at a different M than a
 * decode step, so text matches a non-speculative run exactly at temperature 0 and mostly above it
 * (see Engine::collect_step). The cost is acceptance rate, P(y == x) instead of rejection
 * sampling's min(1, p/q), and no probability planes are needed.
 *
 * Accepting on argmax at a positive temperature is not exact: it replaces the target's
 * distribution with its mode. Do not conflate it with the rule above.
 *
 * Consumers of `n_accepted`: the scheduler frees paged KV blocks past the accepted position. The
 * linear-attention kernels do not recompute: the recurrent state and conv window have absorbed the
 * rejected tokens, and libr4d keeps the conv state as a rolling window of `conv_width-1 + n_spec`
 * entries read at the slot the last accepted token left. A rejection is a read-offset change, which
 * is why `num_accepted` is an operand of `gdn_conv_update` and `gdn_recurrent_update`
 * (docs/OPS.md, spec §10).
 */
#pragma once
#include "../rad_internal.h"

#include <cstdint>

namespace rad {

/* Acceptance by agreement over precomputed target tokens. `target` is [n_draft+1]: the token the
 * target produced at each verified position plus the bonus position past the last draft token.
 * It is the argmax at temperature 0 and the sampled draw above it; the name describes the
 * temperature-0 case, but the rule is exact for both (see the file header).
 *
 * Writes n_accepted+1 tokens into `out` (capacity n_draft+1) and sets *n_accepted. Returns
 * RAD_OK. Always emits at least one token: the target's own token after the last accepted one. */
int accept_greedy(const int32_t* draft, int n_draft, const int32_t* target,
                  int32_t* out, int32_t* n_accepted);

class CandidateTree;

/* The same rule over a candidate tree (arch/common/rad_tree_candidate.h): the verified rows are
 * nodes, and a node is accepted only when its token equals the target's own pick at its PARENT's
 * row. `pick` is the target's token at each node's row, [tree.node_count()] -- the argmax at
 * temperature 0, the sampled draw above it. Each row draws once at its own position, so walking
 * matching children emits exactly what a serial verify of that path would, and a sibling never
 * gets a second draw after a mismatch. Ties between children carrying the same token go to the
 * lower node id, which is what makes the result independent of row order.
 *
 * Writes the accepted draft tokens into `out[0..accepted-1]` and the target's own token past the
 * last accepted node into `out[accepted]` -- the bonus, `accepted + 1` tokens, the same shape
 * accept_greedy produces. `out_path`, when given, receives the node id of each accepted token
 * and of the root, [accepted + 1]: the rows the passes after the verify read. `accept_cut_at_eog`
 * applies to the emitted tokens unchanged, so a stop token inside the tree ends the step where it
 * would end a chain.
 *
 * Returns RAD_E_INVAL for a malformed tree or a `pick` shorter than the tree (every visited row
 * needs its pick), and RAD_E_STATE when the walk ends without a bonus token. */
int accept_tree_greedy(const CandidateTree& tree, const int32_t* pick, int32_t n_pick,
                       int32_t* out, int32_t* n_accepted, int32_t* out_path);

/* Where a verify step's emission ends: at the first end-of-generation token among `out`'s
 * n_accepted + 1 tokens (accept_greedy's output), which is emitted, or after all of them. Returns
 * the number of accepted drafts to commit, at most `n_accepted`. A drafter that predicts the stop
 * token agrees with the target, so without the cut the run would continue into the next turn.
 * The card's acceptance (sched/advance_impl.h, ADV_ACCEPT) applies the same rule, and the engine
 * refuses a step where the two disagree. */
template <class IsEog>
inline int32_t accept_cut_at_eog(const int32_t* out, int32_t n_accepted, IsEog&& is_eog) {
    for (int32_t j = 0; j <= n_accepted; ++j)
        if (is_eog(out[j])) return j;
    return n_accepted;
}

}  /* namespace rad */
