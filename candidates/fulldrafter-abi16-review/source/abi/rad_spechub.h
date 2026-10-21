#ifndef RAD_SPECHUB_H
#define RAD_SPECHUB_H
#include <stddef.h>
#include <stdint.h>

/* Wire rows are U32 planes, with eight-byte aligned bases and row strides. */
typedef struct RadSpechubDraw { uint64_t seed, pos, seq; } RadSpechubDraw;
typedef struct RadSpechubState {
    RadSpechubDraw draw;
    int32_t status, root, count, pair[2], ids[8];
    uint32_t reserved;
    double q[8];
} RadSpechubState;
#define RAD_SPECHUB_DRAW_WORDS (sizeof(RadSpechubDraw)/4)
#define RAD_SPECHUB_STATE_WORDS (sizeof(RadSpechubState)/4)
#ifdef __cplusplus
static_assert(sizeof(RadSpechubDraw) == 24, "SpecHub draw wire");
static_assert(sizeof(RadSpechubState) == 144, "SpecHub state wire");
#endif
#define RAD_SPECHUB_KEY_FIRST 0x5348554246495253ull
#define RAD_SPECHUB_KEY_TAIL  0x534855425441494cull
#define RAD_SPECHUB_KEY_MATCH 0x534855424d415443ull
#define RAD_SPECHUB_KEY_RESID 0x5348554252455349ull

/* Linear MTP coupling (RADIANCE_MTP_COUPLED=1). One draw per drafting row, staged before the
 * serial passes; one state per (temporal round, row). `base` is the request's PRE-commit output
 * size, so (base, round) names one draft stream that no later step reuses. `card` is the step
 * slot the advance writes, checked against RadBatch::seq_ids. `mode` 1 samples x~Q; 0 keeps the
 * greedy proposal (temperature 0 or top_k 1). */
typedef struct RadMtpCoupleDraw {
    uint64_t seed, base, seq;
    int32_t  card;
    uint32_t mode;
} RadMtpCoupleDraw;
/* status 1: x sampled from the exact q[] stored here; 0: greedy proposal kept; -1: malformed
 * shortlist or row identity, greedy proposal kept. `pos` is the verify row's output position,
 * bound by the host after commit; the device writes 0. ids[count..7] = -1, q[count..7] = 0. */
typedef struct RadMtpCoupleState {
    RadMtpCoupleDraw draw;
    uint64_t pos;
    int32_t  status, round, x, count;
    int32_t  ids[8];
    double   q[8];
} RadMtpCoupleState;
#define RAD_MTP_COUPLE_DRAW_WORDS (sizeof(RadMtpCoupleDraw)/4)
#define RAD_MTP_COUPLE_STATE_WORDS (sizeof(RadMtpCoupleState)/4)
#define RAD_MTP_COUPLE_MAX_ROUNDS 8
/* Draft stream of round r: seed ^ (KEY_DRAFT + r) at coordinate `base`. Acceptance and the
 * residual draw use RAD_SAMPLE_KEY_ACCEPT / RAD_SAMPLE_KEY_RESID at the verify position. */
#define RAD_MTP_COUPLE_KEY_DRAFT 0x4d5450434f55504cull
#ifdef __cplusplus
static_assert(sizeof(RadMtpCoupleDraw) == 32, "MTP coupling draw wire");
static_assert(sizeof(RadMtpCoupleState) == 152, "MTP coupling state wire");
static_assert(offsetof(RadMtpCoupleState, q) % 8 == 0, "MTP coupling q alignment");
#endif
#endif
