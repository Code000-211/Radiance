/* accept.cpp -- acceptance for speculative decoding. See accept.h for why the rule is exact.
 */
#include "accept.h"
#include "../../arch/common/rad_tree_candidate.h"

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
    /* The target's token at position i is correct to emit either way: the bonus token on a full
     * run, the correction on a rejection. */
    out[i] = target[i];
    *n_accepted = i;
    return RAD_OK;
}

int accept_tree_greedy(const CandidateTree& tree, const int32_t* pick, int32_t n_pick,
                       int32_t* out, int32_t* n_accepted, int32_t* out_path) {
    if (!out || !n_accepted) return RAD_E_INVAL;

    TreeWinner w;
    const int rc = verify_tree_greedy(tree, pick, n_pick, &w);
    if (rc != RAD_OK) return rc;

    /* Same shape as accept_greedy: accepted draft tokens, then the target's own token past the
     * last one. On a full walk that token is the bonus speculation buys; on a rejection it is the
     * correction, and one path for both is what keeps the bonus from being an off-by-one. */
    for (int32_t j = 0; j < w.accepted; ++j) out[j] = static_cast<int32_t>(w.tokens[j]);
    out[w.accepted] = static_cast<int32_t>(w.bonus);
    if (out_path)
        for (int32_t j = 0; j <= w.accepted; ++j) out_path[j] = w.path[j];
    *n_accepted = w.accepted;
    return RAD_OK;
}

}  /* namespace rad */
