/* plugin/spechub/spechub.h -- the plugin-private contract between its host half and its kernels.
 *
 * Plugin-private ON PURPOSE: core/ must never include this, and everything the CORE needs (the wire
 * row and the coupling arithmetic) already lives in abi/rad_spechub.h. Only the packed call structs
 * and the argument readers are here, shared by the host reference and the device launch.
 */
#ifndef RAD_PLUGIN_SPECHUB_H
#define RAD_PLUGIN_SPECHUB_H

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_param_geti/getdim: the pure readers, inline and total */
#include "rad_spechub.h"

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#define RAD_SPECHUB_HD __host__ __device__
#else
#define RAD_SPECHUB_HD
#endif

/* A draft round's operands, packed once for both the device launch and the host reference. */
struct RadMtpCoupleProposeCall {
    int32_t*       idx;    /* [M, 8] INOUT: this rank's shortlist; column 0 is the proposal */
    const float*   val;    /* [M, 8] the shortlist's scores */
    const int32_t* card;   /* [batch] RadBatch::seq_ids: the slot the advance writes */
    const uint32_t* draw;  /* [M, DRAW_WORDS] RadMtpCoupleDraw */
    uint32_t*      state;  /* [rounds, M, STATE_WORDS] RadMtpCoupleState */
    int64_t        M;
    int64_t        vocab;
    int64_t        ld_idx, ld_val, ld_draw, st_round, st_row;
    int            round;
};

/* The verify's operands: the sampler's own candidate plane, its parameter rows, the bound state
 * rows and the token plane the pick already wrote. */
struct RadMtpCoupleVerifyCall {
    const int32_t*  idx;    /* [M, n_cand] */
    const float*    val;    /* [M, n_cand] */
    const uint32_t* params; /* [M, 24 words] RadSampleParams */
    const uint32_t* state;  /* [M, STATE_WORDS] RadMtpCoupleState */
    int32_t*        token;  /* [M] or [M, 1] */
    int64_t         M, n_cand, vocab, ld[5];
};

int rad_mtp_couple_propose_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_verify_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_propose_args(const RadArgs*, RadMtpCoupleProposeCall*);
int rad_mtp_couple_verify_args(const RadArgs*, RadMtpCoupleVerifyCall*);

/* The geometry/validation the shape hooks and the arg readers share. `op` 0 proposes, 1 verifies. */
int rad_couple_validate(const RadArgs*, int op, int64_t* M, int64_t* vocab, int64_t* n_cand);

/* The op set. Two schemas, two device rows and two host-reference rows. */
int rad_spechub_schema_count();
const RadOpSchema* rad_spechub_schema_at(int i);
RadKernelInfo rad_spechub_row(int i, RadLaunchFn launch);

/* The launches. */
int rad_mine_mtp_couple_propose_launch(const RadArgs*, RadStream);
int rad_mine_mtp_couple_verify_launch(const RadArgs*, RadStream);
int rad_ref_mtp_couple_propose(const RadArgs*, RadStream);
int rad_ref_mtp_couple_verify(const RadArgs*, RadStream);

/* THE PROPOSE'S HOST ARITHMETIC, host and device. One definition of "x ~ Q over the final eight":
 * column 0 is what ADV_ROUND/ADV_TAKE take, and it becomes x only for a valid sampled row. */
static inline RAD_SPECHUB_HD void rad_mtp_couple_propose_row(const RadMtpCoupleProposeCall& c,
                                                             int64_t row) {
    RadMtpCoupleState s{};
    s.draw = *reinterpret_cast<const RadMtpCoupleDraw*>(c.draw + row * c.ld_draw);
    int32_t* idx = c.idx + row * c.ld_idx;
    s.round = c.round;
    s.x = idx[0];
    for (int i = 0; i < 8; ++i) s.ids[i] = -1;
    if (s.draw.mode == 1) {
        s.status = -1;
        if (s.draw.card == c.card[row] &&
            rad_mtp_couple_build_q(idx, c.val + row * c.ld_val, c.vocab, &s)) {
            const double u = rad_spechub_uniform(s.draw.seed, s.draw.base,
                                                 RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)c.round);
            s.x = s.ids[rad_mtp_couple_pick(s, u)];
            s.status = 1;
            idx[0] = s.x;
        } else {
            s.count = 0;
            for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
        }
    }
    *reinterpret_cast<RadMtpCoupleState*>(c.state + (c.round - 1) * c.st_round + row * c.st_row) = s;
}

/* THE VERIFY'S, after sample_pick. Rows without a valid bound state keep the ordinary token. */
static inline RAD_SPECHUB_HD void rad_mtp_couple_verify_row(const RadMtpCoupleVerifyCall& c,
                                                            int64_t row) {
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    if (s.status != 1) return;
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    if ((p.flags & RAD_SP_GREEDY) || s.draw.seed != p.seed || s.pos != p.pos ||
        !rad_mtp_couple_state_valid(s, c.vocab))
        return;
    int32_t out = -1;
    if (rad_mtp_couple_decide(s, c.idx + row * c.ld[0], c.val + row * c.ld[1], c.n_cand, c.vocab,
                              rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT),
                              rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID), &out))
        c.token[row * c.ld[4]] = out;
}

#undef RAD_SPECHUB_HD
#endif  /* RAD_PLUGIN_SPECHUB_H */
