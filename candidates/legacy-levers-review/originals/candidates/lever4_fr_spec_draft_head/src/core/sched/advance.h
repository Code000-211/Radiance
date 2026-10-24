/* advance.h -- the card's own bookkeeping between passes: which drafts a verify step accepted, and
 * the inputs of every pass that follows from it.
 *
 * WHY THE CARD DOES THIS AND NOT THE HOST. A speculative step is a verify pass and the draft passes
 * behind it, and every pass after the verify depends on how many drafts it accepted: the draft
 * head continues from the accepted position, and the next verify starts after it. Deciding that on
 * the host means reading the sampled tokens back, waiting for them, and only then staging the next
 * pass -- a synchronisation per step during which the card has nothing queued. Deciding it here
 * lets the host stage and issue every pass of a step, and the next step, before any of them has
 * run: the rule is a compare of a few integers a sequence, and the card already holds all of them.
 *
 * THE STATE IS PER SEQUENCE SLOT (StepSlot), in device memory on every rank, and it carries exactly
 * what a pass needs that the host cannot know in advance: where the next verify starts, what it
 * rolls back, which tokens it verifies, and where the draft head continues from. Every rank keeps
 * its own copy and computes the same numbers from the same sampled tokens, so no rank ever reads
 * another's memory.
 *
 * A PASS NAMES ITS DEVICE ROWS. A batch can mix rows whose inputs the host staged -- a prefill
 * chunk, a history row for positions the host already knows -- with rows whose positions come from
 * the card; `rows[0][i]` says which, and the advance leaves a host row exactly as it was uploaded.
 *
 * The same code runs on the host backend, where device memory is host memory: the rows run on the
 * issuing thread once the stream has drained, which is their place in stream order
 * (advance_impl.h is shared by both builds).
 */
#pragma once
#include <cstdint>

#include "rad_device.h"
#include "draft.h"

namespace rad {

/* One sequence slot's state between passes. The accept step writes it; every later pass reads it. */
struct StepSlot {
    int32_t ctx;      /* the next verify step's first position: tokens committed so far */
    int32_t n_acc;    /* its one-based num_accepted: tokens the last verify step committed */
    int32_t head;     /* the last committed index, which the draft head's first round extends */
    int32_t acc;      /* drafts the last verify step accepted */
    /* The next verify step's input tokens: the bonus token, then the draft head's proposals. */
    int32_t tok[1 + RAD_SCHED_MAX_SPEC];
    int32_t pad[3];
};

enum AdvMode : int32_t {
    ADV_ACCEPT = 1,   /* after a verify step's sampler: acceptance, and the state it leaves */
    ADV_TRUNK  = 2,   /* before a verify step: its tokens, positions and rollback count */
    ADV_HIST   = 3,   /* before the draft head's history pass: the verified positions */
    ADV_ROUND  = 4,   /* before draft round `round` >= 2: the next position, the last proposal */
    ADV_TAKE   = 5,   /* after the last round: its proposal into the state */
};

/* One KV group, as the builder stages it for this build. Offsets are bytes into the rank's slab,
 * which every rank carves identically. */
struct AdvGroup {
    int32_t block_size;
    int32_t paged;          /* 0: a state group, whose slots do not depend on the position */
    int32_t pitch;          /* this build's block-table pitch */
    int32_t pad;
    int64_t slot_off;       /* [n_tok] slot_mapping */
    int64_t used_off;       /* [n_seq] seqused */
    int64_t bt_off;         /* [n_seq][pitch] block table */
    int64_t first_off;      /* [n_seq] the position each row's block table starts at */
};

/* One advance: a mode, the build it patches, and where its inputs are. Passed by value as the
 * kernel's argument. */
struct AdvanceArgs {
    int32_t mode = 0;
    int32_t round = 0;        /* ADV_ROUND: the draft round, from 2; ADV_TAKE: the last, from 1 */
    int32_t n_seq = 0;
    int32_t n_groups = 0;
    char*   slab = nullptr;   /* this rank's device slab */
    int64_t tok_off = 0, pos_off = 0, ctx_off = 0, acc_off = 0, out_off = 0;
    int64_t sid_off = 0, cu_off = 0, q_off = 0;
    /* [4][row_pitch] int32: device-row flag; the verify step's token row of the row's first
     * verified position; its first sampled row, or -1 for a row that sampled nothing; and one
     * number the mode reads -- ADV_ACCEPT the row's draft count, ADV_HIST how many of the row's
     * positions come before the verified ones, whose tokens the host staged. */
    int64_t row_off = 0;
    int32_t row_pitch = 0;
    int64_t grp_off = 0;      /* [n_groups] AdvGroup */
    StepSlot* st = nullptr;   /* [max_seqs], by scheduler slot */
    /* ADV_ACCEPT and ADV_HIST: the verify step's sampled tokens, one a row. ADV_ROUND and
     * ADV_TAKE: the last round's proposals, row i at i * src_pitch. */
    const int32_t* src = nullptr;
    int64_t src_pitch = 1;
    const int32_t* eog = nullptr;   /* end-of-generation ids, sorted */
    int32_t n_eog = 0;
    const int32_t* vocab_ids = nullptr; /* reduced vocabulary id mapping table */
    int32_t n_vocab_ids = 0;
    /* THE ROTARY PLANES (RadBatch::rope_pos), [3][rope_pitch] at rope_off, or rope_off -1 when the
     * build carries none. A row whose position the card moves carries its rotary position along
     * by the offset the host staged for it: the rows the card owns are decode and draft rows, all
     * text, and text rotates at its index plus one per-sequence shift. A row the card leaves at
     * the host's position keeps the host's three components, which is what a media token needs. */
    int64_t rope_off = -1;
    int32_t rope_pitch = 0;
};

/* Issue one advance on `s`. The rows are independent, so the card runs them in parallel. */
int step_advance(const AdvanceArgs& a, RadStream s);

}  /* namespace rad */

/* The launcher in advance.hip: one kernel onto `stream` through hipLaunchKernel. */
extern "C" int rad_step_advance_launch(const rad::AdvanceArgs* args, void* stream);
