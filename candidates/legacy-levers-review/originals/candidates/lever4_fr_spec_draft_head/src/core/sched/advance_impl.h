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
        /* THE RULE accept_greedy states, and then the end-of-generation cut collect_step applies:
         * a draft is accepted while it equals the target's own pick at its position, and the run
         * stops AT an end-of-generation token rather than after it. */
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
        /* Round 2 onward: round 1 is the history pass's last row (ADV_HIST above), so every
         * round embeds the proposal of the round before it. */
        const int32_t base = S.head + a.round - 1;
        int32_t t = a.src[(int64_t)i * a.src_pitch];
        if (a.vocab_ids && t >= 0 && t < a.n_vocab_ids) t = a.vocab_ids[t];
        S.tok[a.round - 1] = t;
        tok[k] = t;
        adv_pos(a, pos, k, base);
        ctx[i] = base;
        adv_groups(a, i, k, base, 1);
        return;
    }
    case ADV_TAKE: {
        int32_t t = a.src[(int64_t)i * a.src_pitch];
        if (a.vocab_ids && t >= 0 && t < a.n_vocab_ids) t = a.vocab_ids[t];
        S.tok[a.round] = t;
        return;
    }
    default:
        return;
    }
}

}  /* namespace rad */
