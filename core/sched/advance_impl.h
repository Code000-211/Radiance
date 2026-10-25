/* advance_impl.h -- one row of an advance, shared by the kernel (advance.hip) and the host backend
 * (advance.cpp). See advance.h for the modes.
 *
 * Rows are independent: each reads its own slot's state and writes only its own positions, tokens
 * and slots, so the kernel runs one thread per row and the host a loop.
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

/* Every group's slots for the row's `n` positions (from the positions just written) and its
 * length from `base`. A position past the block table gets slot -1 ("no slot"); the builder sizes
 * tables for the furthest reachable position, so only a row whose output is unused hits this. */
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

/* Writes a row's position and moves its rotary components by the same amount. A row whose
 * position does not change keeps the host's components. */
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
    if ((a.mode == ADV_ACCEPT || a.mode == ADV_TREE_ACCEPT) ? rout < 0 : !rows[i]) return;

    int32_t* tok = adv_i32(a, a.tok_off);
    int32_t* pos = adv_i32(a, a.pos_off);
    int32_t* ctx = adv_i32(a, a.ctx_off);
    const int32_t slot = adv_i32(a, a.sid_off)[i];
    const int32_t k = adv_i32(a, a.cu_off)[i];
    const int32_t n = adv_i32(a, a.q_off)[i];
    StepSlot& S = a.st[slot];

    switch (a.mode) {
    case ADV_ACCEPT: {
        /* accept_greedy's rule, then collect_step's end-of-generation cut: a draft is accepted
         * while it equals the target's pick at its position, and the run stops at an
         * end-of-generation token, not after it. */
        const int32_t* y = a.src + rout;
        const int32_t* d = tok + rtok + 1;
        int32_t acc = 0;
        while (acc < aux && d[acc] == y[acc]) ++acc;
        for (int32_t j = 0; j <= acc; ++j)
            if (adv_is_eog(a, y[j])) { acc = j; break; }
        const int32_t head = ctx[i] + (rtok - k) + acc;
        S.acc = acc;
        S.head = head;
        S.ctx = head + 1;
        S.n_acc = acc + 1;
        S.tok[0] = y[acc];
        /* A new cycle: a row that drafts nothing this time must not publish the last cycle's
         * proposals, which the slots still hold. */
        S.gen = 0;
        return;
    }
    case ADV_TREE_ACCEPT: {
        /* verify_tree_greedy's walk over the row's nodes, then collect_step's end-of-generation
         * cut (accept_cut_at_eog's rule). Node 0 is the root at the row's first position; a node is accepted when a lower-id
         * child of the current node carries the target's pick at the current node, lowest id
         * first, and the first node without such a child contributes the step's own token. The
         * row's tokens are the nodes' tokens in id order, its parents at parent_off.
         *
         * Prepared, not wired: nothing in this build issues ADV_TREE_ACCEPT, because the batch
         * cannot stage a tree without an ancestor-mask attention operand (sched/tree_dispatch.h).
         * A node without a pick, a tree with no node or one past the row's capacity is not a tree
         * the host admitted: acc = -1 is what the host's cross-check (tree_dispatch_confirm) will
         * refuse the step on, and the other fields keep the position the slot was staged at. */
        if (n <= 0 || n > kAdvTreeMaxNodes) {
            S.acc = -1;
            S.n_acc = 0;
            S.head = ctx[i];
            S.ctx = ctx[i];
            S.tok[0] = 0;
            return;
        }
        const int32_t* par = adv_i32(a, a.parent_off) + k;
        const int32_t* pick = a.src + rout;
        int32_t emitted[kAdvTreeMaxNodes + 1];
        int32_t node = 0, acc = 0;
        for (;;) {
            const int32_t g = pick[node];
            if (g < 0) {
                S.acc = -1;
                S.n_acc = 0;
                S.head = ctx[i];
                S.ctx = ctx[i];
                S.tok[0] = 0;
                return;
            }
            int32_t match = -1;
            for (int32_t j = node + 1; j < n; ++j)
                if (par[j] == node && tok[k + j] == g) { match = j; break; }
            emitted[acc] = g;
            if (match < 0) break;
            ++acc;
            node = match;
        }
        for (int32_t j = 0; j <= acc; ++j)
            if (adv_is_eog(a, emitted[j])) { acc = j; break; }
        S.acc = acc;
        S.head = ctx[i] + acc;
        S.ctx = S.head + 1;
        S.n_acc = acc + 1;
        S.tok[0] = emitted[acc];
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
        if (a.shared_kv) {
            /* A head that reads the trunk's cache and keeps none (rad_drafter_shared_kv). With no
             * history to fill, the pass is round 1 alone: one row at the committed position, with
             * the trunk's row there and the token chosen after it. Its context ends at the
             * committed position, not the verify step's whole run, because the cache past it
             * holds K/V of rejected drafts. The host staged the run's first trunk row; the card
             * adds the accepted drafts. */
            int32_t* out = adv_i32(a, a.out_off);
            tok[k] = a.src[rout + S.acc];
            adv_pos(a, pos, k, S.head + a.rot_shift);
            out[k] += S.acc;
            ctx[i] = S.head;
            adv_groups(a, i, k, S.head, 1);
            return;
        }
        /* `aux` host-known positions, then the verify step's positions up to the last committed
         * one, each with the token committed after it (the sampler's pick there). The pass has a
         * row for every position the step could have committed; rows past the last committed
         * repeat it, so no rejected draft's hidden state enters the head's cache. The repeat also
         * makes the run's last row draft round 1's input whatever the acceptance was
         * (RadBatch::draft_out_ids). */
        const int32_t base = S.head - S.acc - aux;
        const int32_t last = aux + S.acc;
        int32_t* out = adv_i32(a, a.out_off);
        const int32_t row0 = out[k];
        for (int32_t j = 0; j < n; ++j) {
            const int32_t jj = j < last ? j : last;
            if (jj >= aux) tok[k + j] = a.src[rout + jj - aux];
            adv_pos(a, pos, k + j, base + jj);
            out[k + j] = row0 + jj;
        }
        ctx[i] = base;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_ROUND: {
        /* Round 2 onward (round 1 is the history pass's last row); each round embeds the previous
         * round's proposal. A head with no cache of its own stays at the committed position over
         * the committed context; its proposal reaches the next round through its hidden state. */
        const int32_t base = a.shared_kv ? S.head : S.head + a.round - 1;
        const int32_t t = a.src[(int64_t)i * a.src_pitch];
        S.tok[a.round - 1] = t;
        tok[k] = t;
        adv_pos(a, pos, k, a.shared_kv ? base + a.rot_shift : base);
        ctx[i] = base;
        adv_groups(a, i, k, base, 1);
        return;
    }
    case ADV_TAKE:
        /* Row i's proposal from the pass just issued, at the buffer's own row pitch (a head's
         * proposal buffer, else the sampler's id buffer). `round` is that pass's round, so a row
         * that leaves the shrinking prefix keeps its last proposal at tok[its own K]. */
        S.tok[a.round] = a.src[(int64_t)i * a.src_pitch];
        S.gen = a.round;
        return;
    case ADV_BLOCK: {
        /* A block drafter's query row: the anchor (the token the verify just chose) at the
         * committed end, where the next verify starts, then `aux`, the mask id, at every position
         * after it. The host staged the furthest the anchor can be. */
        const int32_t base = S.ctx;
        tok[k] = S.tok[0];
        for (int32_t j = 1; j < n; ++j) tok[k + j] = aux;
        for (int32_t j = 0; j < n; ++j) adv_pos(a, pos, k + j, base + j);
        ctx[i] = base;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_BTAKE:
        /* Row i is the query pass's i-th block, which is the i-th drafting sequence. */
        for (int32_t j = 0; j < a.round; ++j)
            S.tok[1 + j] = a.src[(int64_t)i * a.src_pitch + j];
        return;
    default:
        return;
    }
}

}  /* namespace rad */
