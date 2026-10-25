/* rad_runtime.h -- the run phase. Ordinary C++ with one rule: it may only issue handles obtained
 * during declare. There is no capture discipline: the device backend records a pass's dispatch
 * packets at the queue as they are issued and replays them (core/device/aql.cpp,
 * core/runtime/ctx.h), so the step function needs no graph-safe form. See spec.md §3.2.
 */
#ifndef RAD_RUNTIME_H
#define RAD_RUNTIME_H

#include "rad_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadCtx RadCtx;

/* ================================================================== the step batch */
/* Built once per step by the core and shared by every layer and KV group; rebuilding it per group
 * and per layer would make a large share of a decode step host work with the device idle (spec §8).
 *
 * Every pointer is device memory unless the field name says _h. The host mirrors give the
 * scheduler and the run phase the bounds without a device read. */
enum { RAD_PHASE_PREFILL = 0, RAD_PHASE_DECODE = 1, RAD_PHASE_MIXED = 2 };

typedef struct RadKVGroupBatch {
    rad_kvgroup group;
    int32_t*    slot_mapping;    /* [n_tok] flat slot index per token, device */
    int32_t*    block_table;     /* [n_seq, max_blocks] device */
    int64_t     block_table_pitch;
    int32_t*    seqused;         /* [n_seq] context length in this group, device */
    int32_t*    state_index;     /* [n_seq, state_index_pitch] state slots per sequence, device */
    /* State slots per sequence: 1 unless the group's kernel asked for scratch beside the state.
     * A linear kernel serving speculation cannot commit the state until acceptance is known, so it
     * keeps the committed state in column 0 and replay scratch in the later columns. The engine
     * zeroes all of them at admission and otherwise does not look inside. Zero means 1. */
    int64_t     state_index_pitch;
    int64_t     block_size;
    int64_t     max_blocks;
} RadKVGroupBatch;

typedef struct RadBatch {
    int      phase;              /* RAD_PHASE_* -- the run phase uses it to index bucket tables */
    int      step;               /* monotonic, for the heat engine and for logs */

    int64_t  n_tok;              /* total tokens across all sequences in this step */
    int64_t  n_seq;

    const int32_t* token_ids;    /* [n_tok] device */
    const int32_t* positions;    /* [n_tok] device */
    const int32_t* cu_seqlens;   /* [n_seq+1] device: sequence boundaries within this step */
    const int32_t* seq_ids;      /* [n_seq] device: engine-side sequence identity */
    /* [n_tok] device: the runtime LoRA adapter each token's row runs with, as a slot of the
     * adapter pools the core declared (docs/OPS.md, lora_shrink / lora_expand). A row whose slot is
     * < 0 or >= the pools' slot count S is a base row, which an adapter op leaves untouched. Every
     * row of a sequence carries its request's slot, fixed when the request was admitted, so a step
     * that mixes adapters and the base needs nothing else.
     *
     * Null on a step with no adapter row, so always when the engine serves no adapters (no --lora
     * and no --max-loras). The core issues the adapter ops itself, only when this is set; a block
     * with an adapter epilogue of its own reads it too. Like every array above it sits at one
     * address per rank for the engine's life and is rewritten every step, so a recorded pass
     * replays it. */
    const int32_t* lora_slot;

    /* The tokens the sampler wants logits for: indices into this step's tokens, [n_out].
     *
     * A prefill chunk computes hidden states for every token, but the sampler wants only each
     * sequence's last. The plugin issues gather_rows over these and runs the lm_head only on the
     * result (docs/OPS.md, "Vocabulary edges"); over all n_tok rows at max_tok 8192 and a 248K
     * vocab the logits buffer would be 8 GiB instead of 63 MiB.
     *
     * n_out is 0 for a chunk that finishes no sequence (the middle of a long prefill): no logits
     * are produced and the lm_head is not run. */
    int64_t        n_out;
    const int32_t* out_ids;            /* [n_out] device */

    const int32_t* q_lens;       /* [n_seq] device: query length this step */
    const int32_t* ctx_lens;     /* [n_seq] device: context length before this step */
    int32_t        max_q_len;    /* host-side bounds -- these select kernel bands */
    int32_t        max_ctx_len;

    int              n_kv_groups;
    const RadKVGroupBatch* kv;   /* [n_kv_groups] */

    /* Speculative window (spec §10). n_spec == 0 for an ordinary decode step. */
    int            n_spec;             /* draft tokens per sequence this step */
    const int32_t* num_accepted;       /* [n_seq] device, from the previous verify */
    const int32_t* spec_parent;        /* [n_tok] device, tree layout; null for a linear chain */

    /* The draft pass. 0 is an ordinary step that runs the trunk; non-zero runs only the draft
     * head, and the sign gives the kind:
     *
     *    r < 0   a history pass: one row per token, gathered from the trunk step's rows
     *            (out_ids). The head attends to its own K and V, which nothing else writes (a
     *            prefill fills only the trunk's layers); this fills them for the positions the
     *            step just committed. When `n_draft_out` is non-zero the pass is also draft round
     *            1 and ends in the lm_head over the rows `draft_out_ids` names.
     *    r > 1   draft round r, one-based. One row per sequence, reading the head's output from
     *            the round before and ending in the lm_head; it samples the r-th draft. There is
     *            no pass with r == 1: the history pass carries that round.
     *
     *    r == 1  a block query, not a draft round: one sequence's anchor plus its mask rows,
     *            carrying `n_draft` block rows whose last `hidden_n` rows produce the drafts. The
     *            block drafter (sched/batch.h, block_query) issues it; the serial drafter never
     *            does, because its round 1 is the history pass above. A plugin must read
     *            `n_draft`/`hidden_row` to tell the two apart, not the value 1 alone.
     *
     * Zero is the ordinary step so that a zeroed RadBatch runs the model, the same reason
     * `num_accepted` is one-based (docs/OPS.md).
     *
     * On any draft pass `token_ids` is already shifted (the head at index i reads the token at
     * i + 1) by the batch builder, so no kernel or plugin applies the shift. Every other field
     * keeps its usual meaning. */
    int32_t        draft_pass;

    /* Linear-state checkpointing (spec §7.3): tokens at which the scheduler wants a checkpoint
     * written, already aligned to the checkpoint interval by the chunk split. */
    int            n_checkpoints;
    const int32_t* checkpoint_tok;     /* [n_checkpoints] device: index into this step's tokens */
    const int32_t* checkpoint_slot;    /* [n_checkpoints] device: destination snapshot slot */

    /* The decode / prefill split within a mixed step.
     *
     * Sequences [0, n_seq_decode) carry decode rows and the rest prefill chunks; the scheduler
     * orders every step that way. A block that treats the two differently issues over each as a
     * contiguous sub-range of the per-sequence arrays (`cu_seqlens + n_seq_decode`,
     * `state_index + n_seq_decode * pitch`, ...) with the token operands left at their base. This
     * works because cu_seqlens holds absolute row offsets, so a kernel reading cu[n] and cu[n+1]
     * finds the right rows in either sub-array.
     *
     * Without the split a mixed step is prefill-shaped for every sequence, so its decode rows
     * cannot verify a draft and produce one token instead of several; under interactive load many
     * steps are mixed.
     *
     * n_seq_decode == n_seq on a pure decode step and 0 on a pure prefill, so a block that does
     * not care can ignore these fields, and one that does needs no phase test. */
    int64_t n_seq_decode;
    /* The decode rows are tokens [0, n_tok_decode) and the prefill chunks the rest. A per-token
     * op cannot derive this, since each decode row is 1 + its own draft count. */
    int64_t n_tok_decode;
    /* The longest query in each half. max_q_len over a mixed step is the prefill chunk's, and
     * banding the decode half with it would give eight rows a thousand-row GEMM's band.
     *
     * Only the decode value is currently read (by the GDN blocks). The prefill value is for a
     * block whose prefill path bands on a per-sequence query length; the GDN prefill ops band on
     * the step's total tokens, taken from their operands. */
    int32_t max_q_len_decode;
    int32_t max_q_len_prefill;

    /* Draft round 1, carried by the history pass: [n_draft_out] device, indices into this pass's
     * rows, one per drafting sequence, in the order the later rounds' rows take.
     *
     * Round 1 continues from the last committed position with the token the trunk chose after it.
     * The history pass already has a row with exactly those inputs: it covers every position the
     * verify step could have committed and repeats the last committed one past it
     * (core/sched/advance.h, ADV_HIST). So each drafting sequence's last history row is its
     * round-1 row, saving a separate pass; the pass ends in the lm_head over those rows, and the
     * head's wide state there is what round 2 continues from.
     *
     * 0 on a history pass that drafts nothing, and on every other pass. */
    int64_t        n_draft_out;
    const int32_t* draft_out_ids;

    /* The prompt after this step: `n_ahead` more tokens of the last sequence, stored in
     * `token_ids` right after its n_tok (the next prefill chunk); 0 on every other step (decode,
     * draft pass, a prefill's last chunk).
     *
     * Nothing is computed on them. They let an op that fetches per-token data from a slow tier
     * start the next step's reads during this one: the PLE gather reads its rows from the
     * container file, tens of thousands of random reads per chunk (docs/OPS.md, ngram_ids'
     * `ahead_ids` and embed_lookup_q's `ahead`). */
    int64_t        n_ahead;

    /* ================================================ multimodal (spec §11) */

    /* The rotary position of every token, which differs from `positions`. `positions` is a
     * token's index in its sequence (the KV slot, the QSA block and other bookkeeping count in
     * it). Qwen-VL style M-RoPE gives each token three components (temporal, height, width): a
     * text token has all three equal, an image token has the image's start in the first and its
     * row and column offsets in the others, and text after an image continues from the image's
     * largest component, so from the first image on the rotary position runs behind the index.
     *
     * [3, n_tok] device, component-major: plane c is rope_pos + c * n_tok. Plane 0 alone is a
     * valid one-component position for every token whose components agree, i.e. every token when
     * `rope_mixed` is 0, so a kernel reading one position per token can use plane 0 then.
     *
     * Null when the program declares no encoder; `positions` is then the rotary position. */
    const int32_t* rope_pos;
    /* Host-side: non-zero when some token on this pass has differing components (an image or
     * video span is present); a block with a one-component fast path must take the
     * multi-component path. */
    int32_t        rope_mixed;

    /* Rows that take an encoder's output in place of a token embedding: row i of `mm_embd`
     * replaces the embedding of this pass's token `mm_rows[i]`. The placeholder ids a media span
     * holds in `token_ids` are still real ids (a hashed n-gram embedding reads them).
     *
     * Rank 0 holds the rows and every other rank zeros, matching a vocabulary-sharded embedding
     * lookup where non-owning ranks contribute zero to the following reduction. So a plugin
     * overwrites its lookup output with these rows before that reduction, whether it is an
     * all-reduce or folded into the first layer's norm.
     *
     * On a pass that shifts its tokens (a draft history pass) the rows are shifted with them. 0
     * rows on every pass without media. */
    int64_t        n_mm_rows;
    const int32_t* mm_rows;            /* [n_mm_rows] device: index into this pass's tokens */
    const void*    mm_embd;            /* [n_mm_rows, n_embd] bf16 device */

    /* The encoder pass. Non-zero `enc` runs a declared encoder (rad_declare_encoder) and nothing
     * else: no token rows, no KV, no logits. Its value is that encoder's `modalities`
     * (RAD_MM_AUDIO, or the image/video mask), selecting between two declared encoders. Input is a
     * list of patches already resized, normalised and cut by the processor; output is the
     * encoder's declared `out` buffer, one row per `merge` patches, read back after the pass.
     *
     * An audio pass's patches are feature frames with 1-D coordinates: plane 0 the frame's index
     * in its clip, plane 1 whether it takes part in the convolution (1) or is staging the encoder
     * reads as zero (0), plane 2 the clip's real frames and plane 3 its staged ones. The zero
     * frames pad a clip staged whole to whole output rows, or, for a clip staged a second at a
     * time (Qwen3-ASR), align each second to whole rows.
     *
     * Patches are grouped into segments (one per image, one per temporal group of a video), and
     * attention runs within a segment. `enc_coord` gives each patch's place in its segment, which
     * is all a learned position table or a 2-D rotary embedding needs. */
    int32_t        enc;
    int64_t        enc_n_patch;
    const void*    enc_pixels;         /* [enc_n_patch, patch_dim] bf16 device */
    /* [4, enc_n_patch] device, component-major like rope_pos: row, column, the segment's height
     * and its width, all in patches. Planes 0 and 1 together are a [2, enc_n_patch] position. */
    const int32_t* enc_coord;
    int64_t        enc_n_seg;
    const int32_t* enc_cu;             /* [enc_n_seg + 1] device: segment boundaries in patches */
    int32_t        enc_max_seg;        /* host: the longest segment, which bands the attention */

    /* ================================================ image generation */

    /* A generator pass (rad_declare_generator): RAD_GEN_PASS_*, with `gen_b` its payload. 0 on
     * every step of a program that declared no generator. */
    int32_t                   gen;
    const struct RadGenBatch* gen_b;

    /* ================================================ audio generation */

    /* An audio generator's pass (rad_declare_audio_generator; `gen` is one of its
     * RAD_GEN_PASS_*): its payload. Null on every other step. */
    const struct RadAudioGenBatch* agen_b;

    /* ================================================ video generation */

    /* A video generator's pass (rad_declare_video_generator; `gen` is one of its
     * RAD_GEN_PASS_*): its payload. Null on every other step. */
    const struct RadVideoGenBatch* vgen_b;
} RadBatch;

/* What a generator pass carries beside the step batch's own fields, which keep their meanings:
 *
 *   COND     one sequence. `token_ids`, `positions` and `rope_pos` are its conditioning prompt, the
 *            encoder's input (`enc_*`) is the prompt's reference pictures and `mm_rows` the prompt
 *            rows the encoder's output rows replace, as on any media step. What is here lays out
 *            the generator's sequence over those rows and writes it into the KV group.
 *   DENOISE  one entry per (image, conditioning): `cu_seqlens` bounds each entry's latent rows,
 *            and the KV group's `block_table` and `seqused` name its conditioning's sequence.
 *   DECODE   one image.
 *
 * Every pointer is device memory unless the name ends in _h. */
typedef struct RadGenBatch {
    /* ---- COND */
    int64_t        rows;          /* the generator's rows for this sequence */
    /* Where each row comes from: >= 0 the prompt row of that index (the conditioning network's
     * output there), < 0 the reference latent row -(1 + src). [rows] */
    const int32_t* src;
    /* Rotary positions, three components a row. On a COND pass they are laid out segment by
     * segment: segment s's [3, length] component-major block starts at element 3 * seg_h[s], so a
     * segment's positions are one dense operand. On a DENOISE pass, [3, n_tok] component-major,
     * the latent rows'. */
    const int32_t* rope;
    /* The sequence in segments: rows [seg_h[s], seg_h[s+1]), each a picture (`seg_bidi_h` 1, its
     * rows attend to each other both ways) or a run of text (0, causal). A segment is never split
     * and attends to every segment before it. `seg` is the same offsets on the device; `seg_cu`
     * is [n_seg, 2], segment s's own cu_seqlens {0, length}, and `seg_start` [n_seg] its first
     * row -- the keys before it, which is its `seqused` when it is run on its own. */
    int64_t        n_seg;
    const int32_t* seg;
    const int32_t* seg_h;
    const int32_t* seg_bidi_h;
    const int32_t* seg_cu;
    const int32_t* seg_start;
    /* [rows] the KV slot each row is stored at. */
    const int32_t* slot_mapping;
    /* Pixels for the decoder's encoder: `n_px` blocks of `px` [*, pixel_channels] bf16 in
     * [-1, 1], block b at pixel px_h[3b], px_h[3b+1] high and px_h[3b+2] wide. These are the
     * reference pictures in prompt order and, with `inpaint`, a last block holding the picture
     * being repainted at the output size, whose latents go to the lanes below. */
    int64_t        n_px;
    const void*    px;
    const int32_t* px_h;
    /* [n_px, 2] each block's own cu_seqlens over its latent tokens: {0, (h / scale) (w / scale)}. */
    const int32_t* px_cu;
    int32_t        inpaint;
    /* [image tokens] f32: 1 where the image is repainted, 0 where the picture is kept. */
    const float*   inp_mask;
    /* The lanes this conditioning's images run in, which take the repainted picture's latents. */
    int64_t        n_lane;
    const int32_t* lane_h;

    /* ---- DENOISE, one value per entry ([n_seq]); on a COND pass `sigma` is one 0, the time
     * the conditioning is modulated at */
    const float*   sigma;         /* where the entry's latent is */
    const float*   sigma_next;    /* where this step takes it */
    const float*   cfg;           /* guidance: v = neg + cfg * (pos - neg); 1 without a pair */
    /* The entry holding this image's negative conditioning, or -1 for none; -2 marks an entry
     * that is one, whose rows serve another entry's step and which does not move its lane. */
    const int32_t* pair;
    const int32_t* lane;          /* the plugin's lane the image's latent lives in */
    const int32_t* lane_dh;       /* ... the same, host */
    /* RAD_GEN_F_*. */
    const int32_t* flags;
    /* [n_seq, 2] the image's noise seed, low word first. */
    const uint32_t* seed;

    /* ---- DECODE */
    int32_t        dec_lane;
    int32_t        dec_height, dec_width;    /* pixels */
    int32_t        dec_preview;              /* the lane's preview latent, not its state */

    /* ---- the joint layout (RAD_GEN_LAYOUT_JOINT_STEP)
     * COND: `branch` is 0 for the prompt and 1 for the negative prompt, and `n_lane` / `lane_h` /
     * `lane` name every lane the conditioning's images run in; the plugin keeps the
     * conditioning's row r for lane l at its own row (2 l + branch) cond_rows + r. The prompt's
     * last `n_pad` tokens are the padding, and `valid_cu` is {0, the tokens before it}. `rope_f`
     * is the conditioning's positions, [4, rows] f32 component-major.
     * DENOISE: each entry's rows are its conditioning and then its image's latent tokens, so
     * `cu_seqlens` bounds both together. `src` [n_tok] names a row's source: >= 0 the plugin's
     * conditioning row (as above), < 0 the image row -(1 + k) of the compact image order, in
     * which `img_cu` [n_seq + 1] bounds each entry's image and `img_rows` [n_img] is each image
     * row's place in the pass. `mode_cu` [3 n_seq + 1] cuts each entry into its text (padding
     * included), its reference pictures and its image: entry e's text is rows [mode_cu[3e],
     * mode_cu[3e + 1]), its pictures [mode_cu[3e + 1], mode_cu[3e + 2]), its image up to
     * mode_cu[3e + 3]. `rope_f` is [4, n_tok]. Everything sigma, cfg, pair, lane, flags and seed
     * say is per entry as for JOINT_GRID, and the lane ops read `img_cu`. */
    int32_t        branch;
    int32_t        n_pad;
    const int32_t* valid_cu;
    int64_t        n_img;
    const int32_t* img_cu;
    const int32_t* img_rows;
    const int32_t* mode_cu;
    const float*   rope_f;
    /* DENOISE: the rows again in mode-major order -- every entry's text, then every entry's
     * pictures, then every entry's image (the compact image order) -- for a model whose early
     * blocks run each mode on its own. `mode_rows` [n_tok] is each one's place in the pass,
     * `mode_src` [n_tok] its source as `src` names it, and `mode_rope_f` holds the positions mode
     * by mode, mode m's a dense [4, n_m] block at 4 times its start;
     * `n_mode_h` [3] counts each mode's rows, so mode m starts at the sum of the counts before
     * it, and `mode_ccu` [3, n_seq + 1] bounds each entry's rows of each mode from that start
     * (mode m's cu_seqlens at mode_ccu + m (n_seq + 1), each beginning at 0). */
    const int32_t* mode_rows;
    const int32_t* mode_src;
    const int32_t* mode_ccu;
    const int64_t* n_mode_h;
    const float*   mode_rope_f;

    /* ---- appended in ABI 25 (Anima, SDXL) */
    /* COND, for a generator that reads its prompt through a second tokenizer
     * (RadGeneratorDecl::aux_tokens): the user's prompt as that tokenizer encodes it, `n_aux` ids
     * on the device, and `aux_cu` {0, n_aux} on the device. 0 and null otherwise. */
    int64_t        n_aux;
    const int32_t* aux_ids;
    const int32_t* aux_cu;
    /* ---- a discrete-time denoiser (RadGeneratorDecl::train_steps), DENOISE, one value per
     * entry: the schedule's sigma before `sigma` (0 on an image's first step), `sigma` on the
     * model's trained clock (the nearest of its timesteps, as ComfyUI conditions it), the
     * request's sampler (RAD_GEN_SAMPLER_*) and the image's step, from 0. */
    const float*   sigma_prev;
    const float*   timestep;
    const int32_t* sampler;
    const int32_t* step;
    /* DENOISE: [n_seq, 2] each entry's image height and width in pixels, f32 (a model
     * conditioned on its size); `entry_h` [n_seq, 4] the same entries on the host: the
     * conditioning's rows (JOINT_STEP), the height and width in pixels, and the branch (0 the
     * prompt, 1 the negative prompt), for a plugin whose ops take an image's geometry from their
     * operands and so issue each entry on its own. */
    const float*   size_px;
    const int32_t* entry_h;
    /* COND: the pass's token ids on the host ([n_tok]), for a plugin whose issue depends on
     * where a window's end-of-text marker sits. */
    const int32_t* tokens_h;
} RadGenBatch;

/* An entry's flags on a DENOISE pass. */
enum {
    RAD_GEN_F_FIRST   = 1,   /* the image's first step: its latent is the noise its seed draws */
    RAD_GEN_F_INPAINT = 2,   /* repaint: after the step, the kept region goes back to the picture,
                              * noised to sigma_next with the image's own noise */
    RAD_GEN_F_PREVIEW = 4    /* write the step's estimate of the clean latent to the lane's preview */
};

/* What an audio generator's pass carries beside the step batch's own fields (rad_builder.h,
 * RadAudioGenDecl, describes the passes):
 *
 *   PROMPT   an ordinary causal prefill: `token_ids`, `positions`, `cu_seqlens` and the KV group's
 *            slot mapping, block table and seqused, two sequences a lane (2i conditional, 2i + 1
 *            unconditional). A chunk of a longer prompt is the same with positions that start
 *            past zero.
 *   FRAME    one row a sequence, two sequences a lane, ordered as PROMPT's: the frame's embedding
 *            is the plugin's (it samples the frame first), so `token_ids` are zero; `positions`,
 *            the slot mapping and seqused place the row after its sequence's earlier ones.
 *   WINDOW   one entry per (window, branch): `cu_seqlens` bounds each entry's rows, `win_lead`
 *            rows of the plugin's own (RadAudioGenDecl::win_lead) and then the window's latent
 *            rows; positions count from 0 in each entry.
 *   VOCODE   one window, no rows.
 *
 * Every pointer is device memory unless the name ends in _h. */
typedef struct RadAudioGenBatch {
    /* ---- PROMPT and FRAME: the lanes of the pass's requests, in entry order */
    int64_t         n_lane;
    const int32_t*  lane;          /* [n_lane] */
    const int32_t*  lane_h;
    /* PROMPT: 1 where this pass holds the end of the lane's prompt, so the last rows' hidden
     * states are the ones the first frame is sampled from. [n_lane] */
    const int32_t*  last;
    const int32_t*  last_h;
    /* FRAME: the frame each lane makes, from 0; frames before RadAudioGenDecl::first_frame only
     * move the model past the prompt. [n_lane] */
    const int32_t*  frame;
    const int32_t*  frame_h;
    /* PROMPT, FRAME and WINDOW: each lane's (WINDOW: each entry's) request seed, [n, 2] low word
     * first. */
    const uint32_t* seed;
    /* PROMPT and FRAME, one value per entry ([n_seq]): the entry's row in a state of two rows a
     * lane, 2 lane + branch, and the entry's last row in the pass (a prompt chunk's, or a frame's
     * one row). */
    const int32_t*  slot;
    const int32_t*  last_row;
    /* PROMPT: each entry's rows already in the KV before this pass ([n_seq]), the prefix a chunk's
     * rows attend to beside their own. */
    const int32_t*  prefix;

    /* ---- WINDOW, one value per entry ([n_seq]) */
    const float*    t;             /* where the entry's latent is, in [0, 1) */
    const float*    t_next;        /* where this step takes it */
    const float*    cfg;           /* v = u + cfg (c - u); 1 without a pair. FRAME: each lane's
                                    * guidance, [n_lane] */
    /* The entry holding this window's unconditional branch, or -1 for none; -2 marks an entry
     * that is one, whose rows serve another entry's step and which does not move its lane. */
    const int32_t*  pair;
    const int32_t*  win_lane;      /* the lane the window's request runs in */
    const int32_t*  win_lane_h;
    /* RAD_AGEN_F_*, device and host. */
    const int32_t*  flags;
    const int32_t*  flags_h;
    /* The window: its index in the request, its first frame and its frames, host and device. */
    const int32_t*  win_index;
    const int32_t*  win_index_h;
    const int32_t*  win_frame0;
    const int32_t*  win_frame0_h;
    const int32_t*  win_frames;
    const int32_t*  win_frames_h;
    /* The latent rows apart from the lead rows: [n_seq + 1] offsets over them, and for each the
     * row of the pass it is (`lat_rows` of them). */
    int64_t         lat_rows;
    const int32_t*  lat_cu;
    const int32_t*  lat_cu_h;
    const int32_t*  lat_src;

    /* ---- VOCODE */
    int32_t         voc_lane;
    int32_t         voc_window;
    int32_t         voc_latents;   /* the window's latents */
    int32_t         voc_left;      /* latents at its start not played */
    int32_t         voc_right;     /* latents at its end not played */

    /* VOCODE with RadAudioGenDecl::voc_chunk: the lane's first frame the pass reads, counted
     * from the first kept frame. 64 bits so that it starts past the struct's ABI 18 end. */
    int64_t         voc_frame0;
} RadAudioGenBatch;

/* A window entry's flags on a WINDOW pass. FIRST is RAD_GEN_F_FIRST's bit and the others are clear
 * of RAD_GEN_F_*'s, so one flags array serves the image generator's sampler ops as well. */
enum {
    RAD_AGEN_F_FIRST = 1,    /* the window's first step: its latent is the noise its seed draws and
                              * its conditioning is built from the request's frames */
    RAD_AGEN_F_LAST  = 8,    /* the window's last step: after it the latent is final */
    RAD_AGEN_F_JOIN  = 16    /* the window has a predecessor whose end it overlaps */
};

/* What a video generator's pass carries beside the step batch's own fields (rad_builder.h,
 * RadVideoGenDecl, describes the passes). Every pass is one request's, in lane `lane`:
 *
 *   VCOND    `token_ids`, `positions` and `cu_seqlens` are the prompt, one sequence of `n_tok`
 *            rows, attended causally, nothing paged.
 *   VSTEP    `n_tok` rows: the packed sequence, n_text + audio_channels n_audio + the video's
 *            patches. No token ids.
 *   ADECODE  no rows of the pass's own.
 *   VDECODE  `dec_rows` rows of the decoder's own (rad_arch_video_positions, RAD_VPOS_DECODE).
 *   VUPSCALE no rows of the pass's own: the lane's video latent from `geom_from` to `geom`.
 *
 * Every pointer is device memory unless the name ends in _h. */
typedef struct RadVideoGenBatch {
    int32_t         lane;
    RadVideoGeom    geom;          /* the request's geometry */
    /* The geometry again for ops that read it from an operand, i32 [RAD_VGEOM_N] (below). */
    const int32_t*  geom_d;
    /* The same lane, i32 [1], and 1 on a request's first step (0 otherwise), i32 [1]. */
    const int32_t*  lane_d;
    const int32_t*  first_d;
    /* i32 [4]: the packed sequence's segments, {0, n_text, n_text + audio rows, n_tok} -- text,
     * soundtrack, video -- and i32 [4] {0, video rows, 0, audio rows}, the bounds of the lane's two
     * latents counted from each one's first row. */
    const int32_t*  seg_cu;
    const int32_t*  lat_cu;

    /* ---- VCOND: every step the request takes, the video's timestep and the soundtrack's */
    int32_t         n_steps;
    const float*    t_video;       /* [n_steps] */
    const float*    t_audio;       /* [n_steps] */

    /* ---- VSTEP */
    int32_t         step;          /* from 0; the first draws the lane's noise */
    /* f32 [6]: the video's timestep, sigma and next sigma, then the soundtrack's. */
    const float*    sched;
    const float*    pos;           /* f32 [3, n_tok]: RAD_VPOS_STEP */
    /* u32 [2, 2]: the video's noise seed and the soundtrack's, low word first (VSTEP, VUPSCALE and
     * VDECODE passes; a decoder that denoises draws from it). */
    const uint32_t* seed;

    /* ---- VDECODE */
    int32_t         chunk;         /* from 0, of n_chunks */
    int32_t         n_chunks;
    int64_t         dec_rows;      /* RAD_VPOS_DECODE */
    int32_t         n_frames_out;  /* frames the pass writes to frames_out */

    /* ---- ADECODE */
    int64_t         n_samples;     /* samples a channel the pass writes to audio_out */

    /* ---- VUPSCALE (rad_builder.h, "Latent upscaling"). `geom` is the geometry the video latent
     * is enlarged to and `geom_from` the one it has; `geom_d` carries both (RAD_VGEOM_UP_*).
     * `sched` is the first full-size step's, whose sigmas (sched[1] the video's, sched[4] the
     * soundtrack's) the latents are re-noised to; `seed` is a step's. */
    RadVideoGeom    geom_from;
    const float*    upscale;       /* f32 [1]: the factor the network is told */

    /* ---- appended in ABI 24 */
    /* VCOND with a negative prompt: `n_neg` of the pass's tokens, after the prompt's. */
    int32_t         n_neg;
    /* The request's guidance, f32 [8] host: the video's scale, its spatio-temporal scale, its
     * modality scale and its rescale, then the soundtrack's four; a model's own defaults where
     * the request named none. */
    const float*    guidance_h;
    /* VCOND with a first frame (RadVideoGenDecl::image_cond): `image` [height, width, 3] bf16 in
     * [-1, 1], device; null without one. Every later pass carries `has_image` 1 for such a
     * request. */
    const uint16_t* image;
    int32_t         has_image;
    /* guidance_h on the device, f32 [8], for ops that read the terms. */
    const float*    guidance;
    /* VDECODE: the decoder the request named (RadVideoGenDecl::decoders), its index; 0 the
     * default. */
    int32_t         decoder;
} RadVideoGenBatch;

/* RadVideoGenBatch::geom_d. The request's latent geometry, then a VDECODE pass's chunk: the latent
 * frames it reads (from latent frame `chunk_lat0`), its latent grid again so that (frames, h, w,
 * first, last) is one run, and the same chunk in patches. */
enum {
    RAD_VGEOM_LAT_FRAMES = 0, RAD_VGEOM_LAT_H = 1, RAD_VGEOM_LAT_W = 2, RAD_VGEOM_N_TEXT = 3,
    RAD_VGEOM_N_AUDIO = 4, RAD_VGEOM_CHUNK_LAT0 = 5,
    RAD_VGEOM_CHUNK_FRAMES = 6, RAD_VGEOM_CHUNK_H = 7, RAD_VGEOM_CHUNK_W = 8,
    RAD_VGEOM_CHUNK_FIRST = 9, RAD_VGEOM_CHUNK_LAST = 10,
    RAD_VGEOM_PATCH_FRAMES = 11, RAD_VGEOM_PATCH_H = 12, RAD_VGEOM_PATCH_W = 13,
    /* A VUPSCALE pass's: (frames, h, w, H, W), the latent's grid before and after, of which the
     * first three are also the source volume; (frames, H, W) the target volume; then both in
     * patches, (frames, h / patch, w / patch) and (frames, H / patch, W / patch). */
    RAD_VGEOM_UP_SRC = 16, RAD_VGEOM_UP_DST = 21, RAD_VGEOM_UP_PSRC = 24, RAD_VGEOM_UP_PDST = 27,
    /* The soundtrack's latents as a volume of single latents, (n_audio, 1, 1); and the video's
     * rows cut at its first latent frame, {0, that frame's patches, every patch} (a first frame
     * a model conditions on at its own timestep). */
    RAD_VGEOM_AUDIO_VOL = 32, RAD_VGEOM_FIRST_CU = 35,
    RAD_VGEOM_N = 40
};

/* ================================================================== operands */
/* RAD_OPK_KV names a KV group's pool rather than a buffer: attention and kv_store take the cache
 * as an operand, and RadKVGroupBatch carries the slot mapping and block table but not the pool
 * base, which belongs to the block manager. handle is the rad_kvgroup and offset is the layer,
 * since a group's pool holds one cache per bound layer. */
/* RAD_OPK_WTAB names a run of consecutively declared weights. A routed MoE layer has no single
 * base to stride from: its experts are separate movement units (spec §4.2), and after placement
 * one may be in VRAM, the next streamed over the link and a third on disk. The core resolves each
 * handle like any weight (residency table, mover event, use note) and hands the kernel a device
 * array of pointers, one per expert, in declaration order. See RAD_WTAB below for the RadTensor
 * shape rule. */
enum { RAD_OPK_BUF = 0, RAD_OPK_WEIGHT = 1, RAD_OPK_RAW = 2, RAD_OPK_NONE = 3, RAD_OPK_KV = 4,
       RAD_OPK_WTAB = 5 };

typedef struct RadOperand {
    uint8_t  kind;
    uint32_t dtype;      /* RAD_OPK_RAW only: what the pointer points at (buffers and weights
                          * carry their dtype in the declaration). RAD_DT_INVALID means the
                          * kernel knows it from the schema position, which a specialised
                          * kernel does and libref does not. */
    uint32_t handle;     /* rad_buf or rad_weight */
    void*    raw;        /* RAD_OPK_RAW: a pointer the core did not allocate (a batch field) */
    int64_t  offset;     /* element offset into the buffer or weight */
    int64_t  rows;       /* 0 = the declared extent; >0 = only this many rows of dim 0 */
    int64_t  cols;       /* 0 = the declared extent; >0 = a column slice of that width, with the
                          * declared width kept as the stride. Lets q, k and v be views of one
                          * fused qkv buffer instead of three buffers and a copy. */
} RadOperand;

#define RAD_B(h)            ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, 0, 0, 0 })
#define RAD_B_AT(h, o)      ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (o), 0, 0 })
#define RAD_B_N(h, o, n)    ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (o), (n), 0 })
/* A column slice: `w` columns starting at column `c`, keeping the buffer's declared width as the
 * stride. RAD_B_COL(qkv, 0, n_q) is the q view of a fused projection. */
#define RAD_B_COL(h, c, w)  ((RadOperand){ RAD_OPK_BUF,    RAD_DT_INVALID, (h), 0, (c), 0, (w) })
#define RAD_W(h)            ((RadOperand){ RAD_OPK_WEIGHT, RAD_DT_INVALID, (h), 0, 0, 0, 0 })
/* A weight table: the `n` weights declared at handles h, h+1, ... h+n-1, as a declaration loop
 * over experts produces (rad_decl_weight hands out consecutive indices).
 *
 * The kernel receives `RadTensor::data` as a device array of `n` `void*`, entry e pointing at
 * expert e's first byte. `shape[0]` is n and `stride[0]` is 0; the zero leading stride marks an
 * axis that must be indexed through the table, not by arithmetic. `dtype` and `shape[1..]` /
 * `stride[1..]` are one entry's; every weight in the run must agree on them, or the core refuses
 * the issue by name.
 *
 * The table is rebuilt only when a pointer changes, so a resident model pays one comparison per
 * expert per issue and no copy. */
#define RAD_WTAB(h, n)      ((RadOperand){ RAD_OPK_WTAB,   RAD_DT_INVALID, (h), 0, 0, (n), 0 })
#define RAD_KV(g, layer)    ((RadOperand){ RAD_OPK_KV,     RAD_DT_INVALID, (g), 0, (layer), 0, 0 })
#define RAD_P(ptr)          ((RadOperand){ RAD_OPK_RAW,    RAD_DT_INVALID, 0, (void*)(ptr), 0, 0, 0 })
#define RAD_P_T(ptr, dt)    ((RadOperand){ RAD_OPK_RAW,    (dt), 0, (void*)(ptr), 0, 0, 0 })
/* A raw pointer to a dense 2-D array, [rows, cols], packed. On a raw operand `cols` is the second
 * extent, not a column slice (there is no declared width to slice). This is how a two-dimensional
 * batch field such as the block table, [n_seq, max_blocks], keeps its width. */
#define RAD_P_T2(ptr, dt, r, c) \
    ((RadOperand){ RAD_OPK_RAW, (dt), 0, (void*)(ptr), 0, (r), (c) })
#define RAD_NONE            ((RadOperand){ RAD_OPK_NONE,   RAD_DT_INVALID, 0, 0, 0, 0, 0 })

#define RAD_OPD(...)  ((const RadOperand[]){ __VA_ARGS__ }), \
                      (int)(sizeof((const RadOperand[]){ __VA_ARGS__ }) / sizeof(RadOperand))

/* ================================================================== issue */
/* The core resolves each weight handle to a pointer, waits on the mover's event if the weight is
 * in flight, indexes the op's bucket table with `n`, and dispatches to the device or host
 * implementation for the current execution site.
 *
 * `n` is the value of the op's ranged parameter (at most one per op, so this is a single integer
 * on the hot path). RAD_N_BATCH means the batch's token count, which almost every op wants. An op
 * with no ranged parameter ignores it.
 *
 * Ops take handles, not pointers, so a weight's location is not fixed at declare; placement
 * depends on this (spec §3.2). */
#define RAD_N_BATCH ((int64_t)-1)

int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n);

/* The second lane, for overlapping a collective (or an arm no shard divides) with compute.
 * `rad_lane` selects the stream later issues launch on (0 or 1); `rad_lane_join(c, from, to)`
 * orders `to` behind everything already issued on `from` without blocking the host. The runtime
 * joins lane 1 into lane 0 at the end of every step, so a missing join costs speed, not
 * correctness.
 *
 * Three rules (reasons at Ctx::lane1_ in core): every collective must use the same lane, since
 * the all-reduce numbers its scratch slots per launch and the ranks must agree; a kernel with a
 * device-wide arrival counter must key it on the stream (libr4d's split-K GEMM does), or it may
 * not run here; and every buffer the second lane touches must be named to rad_buf_concurrent,
 * because the planner packs transients from one linear order. */
int rad_lane(RadCtx* c, int lane);
int rad_lane_join(RadCtx* c, int from, int to);

/* Same, with the ranged value taken from the batch. */
#define RAD_ISSUE(c, op, ...) rad_issue((c), (op), RAD_OPD(__VA_ARGS__), RAD_N_BATCH)
#define RAD_ISSUE_N(c, op, n, ...) rad_issue((c), (op), RAD_OPD(__VA_ARGS__), (n))

/* Fail the step with a reason. An architecture block's step() returns void, so this is how it
 * refuses a batch shape it does not serve; the engine stops with `what` in the message. Returns
 * the status it set, so a helper can `return rad_step_fail(...)`. */
int rad_step_fail(RadCtx* c, const char* what);

/* ================================================================== run-phase queries */
const RadBatch* rad_batch(RadCtx* c);
int             rad_rank(RadCtx* c);
int             rad_world_size(RadCtx* c);
RadStream       rad_stream(RadCtx* c);

/* The routing result for a MoE layer, resolved on device against the current residency table.
 * Routing never blocks the compute stream: the histogram is copied back asynchronously and the
 * heat engine consumes it on the next step (spec §5.5). These are device-side buffers the plugin
 * issues its expert ops against, not host-visible expert ids. */
typedef struct RadRouting {
    const int32_t* expert_ids;    /* [n_tok, top_k] device */
    const float*   expert_w;      /* [n_tok, top_k] device */
    const int32_t* expert_count;  /* [n_expert] device: tokens routed to each */
    const int32_t* sorted_tok;    /* [n_tok * top_k] device: tokens grouped by expert */
    const int32_t* expert_offset; /* [n_expert+1] device */
    int64_t        top_k;
    int64_t        n_expert;
} RadRouting;

/* Record that this layer's routing lives in these buffers, so the heat engine can read the
 * histogram back and the mover can be told what was wanted. The plugin computed them with
 * ordinary declared ops; this is the handoff, not a computation. */
int rad_route_report(RadCtx* c, int layer, const RadRouting* r);

/* A core-owned device row of at least `n_expert` int32 where a layer's expert counts can be
 * written in place; the core copies it to the host once at the end of the pass. A router that
 * writes here and reports this pointer as `expert_count` hands the counts over for free; any other
 * pointer costs one copy dispatch per routed layer on the compute stream. Null when the core keeps
 * no histogram for `layer` or its row is narrower than `n_expert`; the router then writes its own
 * buffer and reports that. */
int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert);

/* Next-layer expert prefetch, at decode. At its router input a routed layer names the next routed
 * layer; the core applies that layer's router to this layer's input rows, picks the experts it is
 * likeliest to route that sit in the pinned host pool, and copies them to the card while the rest
 * of this layer runs. Just before the next layer's routed GEMMs it points their weight-table
 * entries at the copies that have landed, where the entries still name the pool slots copied from.
 * Results are unchanged: a copy holds its slot's bytes. Active only when placement keeps experts
 * in a pinned pool and the device backend can spare a DMA engine; a pass of more than 16 rows
 * makes no prediction. The three calls are no-ops otherwise.
 *
 * `parts` names, for each of the next layer's experts, the weights its unit is made of (at most four,
 * e.g. the gate-up matrix and its scales and the down matrix and its scales), as the routed GEMMs
 * read them through weight tables; 0 where an expert has fewer or none. An expert is prefetched only
 * where each of its weights is in exactly one table and all of them sit in one pool slot. The arrays
 * must outlive the context; the core reads the routing counts only to rank and count, never to choose
 * an expert's bytes. `layer` is the next layer's index as rad_route_report and
 * rad_expert_prefetch_step name it; `router` is its router, [n_expert, n_embd] bf16. */
typedef struct RadPrefetchNext {
    int               layer;
    rad_weight        router;
    int64_t           n_expert, top_k;
    int64_t           n_parts;
    const rad_weight* parts;      /* [n_expert * n_parts] */
    const int32_t*    count_at;   /* [n_expert]: where the expert's count is in the layer's routing
                                     counts (rad_route_counts), -1 for nowhere; null when at its index */
} RadPrefetchNext;

/* At this layer's router input: `h` rows [r0, r0 + rows) of n_embd bf16, read in place on another
 * stream -- the block must not overwrite them before this layer's routed GEMMs. */
int rad_expert_prefetch(RadCtx* c, const RadPrefetchNext* next, rad_buf h, int64_t r0, int64_t rows,
                        int64_t n_embd);
/* After this layer's routed down GEMM: the copies cross the link from here, overlapping the next
 * layer's attention rather than this layer's own streamed experts. */
int rad_expert_prefetch_mark(RadCtx* c);
/* Ahead of `layer`'s routed GEMMs, on every routed layer of every pass. */
int rad_expert_prefetch_step(RadCtx* c, int layer);

/* A weight's pointer, for the rare plugin-private kernel invoked outside the op vocabulary. Waits
 * on the mover. The weight may move afterwards, so the pointer is valid only within one issue. */
void* rad_weight_ptr(RadCtx* c, rad_weight w);
void* rad_buf_ptr(RadCtx* c, rad_buf b);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_RUNTIME_H */
