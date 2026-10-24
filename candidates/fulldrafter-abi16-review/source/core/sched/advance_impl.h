/* advance_impl.h -- one row of an advance, shared by the kernel (advance.hip) and the host backend
 * (advance.cpp). See advance.h for what the modes are and why the card computes them.
 *
 * Every row is independent: it reads its own slot's state and writes only its own positions,
 * tokens and slots, so the kernel runs one thread a row and the host a loop.
 */
#pragma once
#include "advance.h"

#if defined(__HIPCC__)
#define RAD_ADV_FN __host__ __device__ inline
#else
#define RAD_ADV_FN inline
#endif

namespace rad {

RAD_ADV_FN int32_t* adv_i32(const AdvanceArgs& a, int64_t off) {
    return reinterpret_cast<int32_t*>(a.slab + off);
}

RAD_ADV_FN bool adv_is_eog(const AdvanceArgs& a, int32_t t) {
    int32_t lo = 0, hi = a.n_eog;
    while (lo < hi) {
        const int32_t mid = (lo + hi) / 2;
        if (a.eog[mid] < t) lo = mid + 1; else hi = mid;
    }
    return lo < a.n_eog && a.eog[lo] == t;
}

/* Every group's slots for the row's `n` positions, read from the positions just written, and its
 * length from `base`. A position its block table does not reach gets slot -1, which the kernels
 * read as "no slot" -- the builder sized the tables for the furthest position a pass can reach,
 * so it only happens to a row that is being computed for nothing. */
RAD_ADV_FN void adv_groups(const AdvanceArgs& a, int i, int32_t k, int32_t base, int32_t n) {
    const AdvGroup* G = reinterpret_cast<const AdvGroup*>(a.slab + a.grp_off);
    const int32_t* pos = adv_i32(a, a.pos_off);
    for (int g = 0; g < a.n_groups; ++g) {
        const AdvGroup& e = G[g];
        const int32_t first = adv_i32(a, e.first_off)[i];
        adv_i32(a, e.used_off)[i] = base + n - first;
        if (!e.paged) continue;
        const int32_t* bt = adv_i32(a, e.bt_off) + (int64_t)i * e.pitch;
        int32_t* slot = adv_i32(a, e.slot_off);
        for (int32_t j = 0; j < n; ++j) {
            const int32_t p = pos[k + j];
            const int32_t bi = (p - first) / e.block_size;
            slot[k + j] = (p >= first && bi < e.pitch && bt[bi] >= 0)
                              ? bt[bi] * e.block_size + p % e.block_size : -1;
        }
    }
}

/* A row's position, written, with its rotary components moved by the same amount -- only when it
 * moved, so a row the host placed exactly keeps whatever components the host gave it. */
RAD_ADV_FN void adv_pos(const AdvanceArgs& a, int32_t* pos, int32_t row, int32_t p) {
    const int32_t was = pos[row];
    pos[row] = p;
    if (a.rope_off < 0 || was == p) return;
    int32_t* r = adv_i32(a, a.rope_off);
    const int32_t shift = r[row] - was;
    r[row] = p + shift;
    r[a.rope_pitch + row] = p + shift;
    r[2 * a.rope_pitch + row] = p + shift;
}

RAD_ADV_FN void adv_row(const AdvanceArgs& a, int i) {
    const int32_t* rows = adv_i32(a, a.row_off);
    const int32_t rtok = rows[a.row_pitch + i];
    const int32_t rout = rows[2 * a.row_pitch + i];
    const int32_t aux  = rows[3 * a.row_pitch + i];
    /* Acceptance runs over every row that sampled, whoever staged it; every other mode over the
     * rows the card owns. */
    if (a.mode == ADV_ACCEPT ? rout < 0 : !rows[i]) return;

    int32_t* tok = adv_i32(a, a.tok_off);
    int32_t* pos = adv_i32(a, a.pos_off);
    int32_t* ctx = adv_i32(a, a.ctx_off);
    const int32_t slot = adv_i32(a, a.sid_off)[i];
    const int32_t k = adv_i32(a, a.cu_off)[i];
    const int32_t n = adv_i32(a, a.q_off)[i];
    StepSlot& S = a.st[slot];

    switch (a.mode) {
    case ADV_ACCEPT: {
        const int32_t* y = a.src + rout;
        const int32_t* d = tok + rtok + 1;
        const int32_t* par = a.spec_parent ? a.spec_parent + rtok : nullptr;
        /* A TREE IS THE SAME RULE WITH A PARENT ARRAY. `spec_parent` is [n_tok] over the whole step,
         * so this row's node j carries `spec_parent[rtok + j]`; node 0 is the anchor and every other
         * node's parent is a LOWER id, which makes each node's depth reachable by walking parents.
         * Acceptance takes, at each depth, the child of the node just accepted whose token equals
         * the target's own pick AT THAT NODE -- `accept_tree_greedy`'s sample-and-match, which is
         * what the host computes for the same proposal.
         *
         * TWO COORDINATES LIVE HERE AND THEY ARE NOT THE SAME ONE ON A TREE:
         *
         *   `cur`   the node id the walk reached. Every sampled row is one NODE, so the pick that
         *           continues from it is `y[cur]` and the row the next pass starts at is `cur` --
         *           NOT `acc`, which is only the number of tokens taken.
         *   `depth` how far along the text that node sits, and therefore the position the sequence
         *           reaches. A two-token path can end on node 9.
         *
         * On a chain the two coincide, which is exactly why using `acc` for either would be
         * invisible until a branched window is served. */
        int32_t cur = 0;
        int32_t acc = 0;
        S.path[0] = 0;
        for (;;) {
            int32_t next = -1;
            if (!par) {
                /* A CHAIN: node acc+1 is the only candidate, and `d` is one entry a token. */
                if (acc >= aux || d[acc] != y[acc]) break;
                next = acc + 1;
            } else {
                if (acc >= aux) break;
                const int32_t want = y[cur];
                for (int32_t j = 1; j <= aux; ++j) {
                    if (par[j] != cur) continue;
                    /* the candidate's depth: a parent walk, bounded by the node count */
                    int32_t dep = 0, p = j;
                    while (p > 0 && dep <= acc) { p = par[p]; ++dep; }
                    if (dep == acc + 1 && p == 0 && d[j - 1] == want) { next = j; break; }
                }
                if (next < 0) break;
            }
            /* THE END-OF-GENERATION CUT, BEFORE THE TOKEN IS COUNTED.
             *
             * A stop token the target picks here IS the step's committed token -- `bonus` below
             * becomes it and the sink emits it -- but it is not an accepted DRAFT, so `acc` does
             * not advance past it and the position the sequence reaches stays where the last real
             * draft left it. Counting it would advance the next step one position past a token
             * that ended the turn, which is text the model never emitted.
             *
             * The rule matches what accept_greedy did before trees existed: it accepted the run,
             * then truncated at the first stop token WITHOUT removing it from the emitted set. */
            const int32_t took = d[next - 1];
            if (adv_is_eog(a, took)) break;
            cur = next;
            S.path[acc + 1] = next;
            ++acc;
        }
        /* THE BONUS IS THE TARGET'S PICK AT THE LAST ACCEPTED NODE'S OWN ROW. */
        const int32_t bonus = y[cur];
        int32_t dep = 0;
        if (par) for (int32_t p = cur; p > 0; p = par[p]) ++dep;
        const int32_t head = ctx[i] + (rtok - k) + (par ? dep : acc);
        S.acc = acc;
        S.head = head;
        S.ctx = head + 1;
        S.n_acc = acc + 1;
        S.tok[0] = bonus;
        /* `S.path` was filled as the walk went: node id a depth, root at 0. On a chain that is
         * 0,1,2,... ; on a tree it follows the parents the proposal was built with, which is what
         * lets the pass that fills the drafter's own cache gather the ACCEPTED rows rather than
         * whichever node happens to sit at an index. */
        return;
    }
    case ADV_TRUNK: {
        const int32_t base = S.ctx;
        for (int32_t j = 0; j < n; ++j) { tok[k + j] = S.tok[j]; adv_pos(a, pos, k + j, base + j); }
        ctx[i] = base;
        adv_i32(a, a.acc_off)[i] = S.n_acc;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_HIST: {
        /* `aux` positions the host already knew the tokens of, then the verify step's own
         * positions up to the last one it committed, each with the token committed after it --
         * the sampler's pick there, since the drafts before it agreed with it. The pass has a
         * row for every position the step COULD have committed, and the ones past the last
         * committed repeat it: a position past it would take a rejected draft's hidden state
         * into the head's cache. The repeat is also what makes the run's LAST row draft round
         * 1's -- the committed position, its hidden state and the token after it -- whatever
         * the acceptance was (RadBatch::draft_out_ids). */
        const int32_t base = S.head - S.acc - aux;
        const int32_t last = aux + S.acc;
        int32_t* out = adv_i32(a, a.out_off);
        /* TWO BASES, AND THEY ARE NOT THE SAME NUMBER.
         *
         * `row0` is where the LEAD starts -- the trunk rows this step computed before the verified
         * set, contiguous by construction.
         *
         * The verified rows start somewhere else in the tap plane: at `out[k + aux]`, the row the
         * host staged for the first VERIFIED position, which is the first node of the accepted
         * path. It has to be READ BEFORE THIS LOOP WRITES, because the loop overwrites the very
         * entries it is reading -- with `aux > 0` the staged entry at `k + aux` is consumed on the
         * first verified iteration and every later one would read a row this loop already wrote.
         *
         * Using `row0` for the verified rows instead drops the prefix: at `aux = 2` a path starting
         * at row 40 would be written as 7, 7+2, ... against a lead that began at 7, so the head
         * gathers two trunk rows that are not the ones it verified. */
        const int32_t row0 = out[k];
        const int32_t vbase = n > aux ? out[k + (aux < last ? aux : last)] : row0;
        for (int32_t j = 0; j < n; ++j) {
            const int32_t jj = j < last ? j : last;
            /* THE ROW IS THE NODE, FOR THE TOKENS AND FOR THE ROW LIST BOTH.
             *
             * `out_ids` is this pass's tap-row list: the host stages the rows the pass must read
             * and the card writes back the ones it actually used. On a tree the verified rows are
             * the accepted path's NODE ids, so a flat run would replace the host's staged path and
             * the gather would read whichever sibling sits at each depth. On a chain `S.path` is
             * the identity and this is the run it always was. */
            const int32_t node = jj >= aux && S.path[jj - aux] >= 0 ? S.path[jj - aux] : (jj - aux);
            if (jj >= aux) tok[k + j] = a.src[rout + node];
            adv_pos(a, pos, k + j, base + jj);
            out[k + j] = jj >= aux ? vbase + node : row0 + jj;
        }
        ctx[i] = base;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_ROUND: {
        /* Round 2 onward: round 1 is the history pass's last row (ADV_HIST above), so every
         * round embeds the proposal of the round before it. */
        const int32_t base = S.head + a.round - 1;
        const int32_t t = a.src[(int64_t)i * a.src_pitch];
        S.tok[a.round - 1] = t;
        tok[k] = t;
        adv_pos(a, pos, k, base);
        ctx[i] = base;
        adv_groups(a, i, k, base, 1);
        return;
    }
    case ADV_TAKE:
        S.tok[a.round] = a.src[(int64_t)i * a.src_pitch];
        return;
    case ADV_BLOCK:
        tok[k] = S.tok[0];
        adv_i32(a, a.dout_off)[i] = rtok + S.acc;
        return;
    default:
        return;
    }
}

}  /* namespace rad */
