/* bole_layout.h -- the plain C++ half of the Bole plugin: the flattened kernel argument bundle and
 * the operand indices, shared by the host row table and the device source.
 *
 * Nothing here is device code, so a host translation unit can include it. The device helpers that
 * need gfx1201 builtins live in bole_device.h, which only the .hip includes: including that from a
 * host compile fails on the very first __builtin_amdgcn_*.
 */
#ifndef BOLE_LAYOUT_H
#define BOLE_LAYOUT_H

#include <stdint.h>

/* Operand positions, in the schema's order (bole_rows.cpp). One enum, used by both halves, so a
 * reordering cannot silently mean a different tensor on the two sides. It must be kept in step with
 * kBoleOperands: a single transposed entry makes the host read one tensor's dtype where another's
 * belongs, which is how a valid call gets refused and an invalid one gets served. */
enum {
    BX = 0, BCW, BCB, BCST, BCIDX, BA, BB, BALOG, BDTB, BSTATE, BSIDX, BNACC, BCU, BPARENT, BCTX,
    BZ, BNW, BFACT, BO, BOQ, BOS, BOLE_N_OPS
};

/* One struct by value: the ABI hands a launcher a RadArgs and nothing else, and hip's
 * kernel-argument path copies slot bytes verbatim, so the host flattens the descriptors and the
 * device sees plain offsets. */
struct BoleCall {
    void*   t[BOLE_N_OPS];
    int64_t s[BOLE_N_OPS][4];
    int     nseq, tokens, q_len, sigmoid, conv_w;
    /* Widths and counts are separate fields on purpose. `kw`/`vw` are the head WIDTHS (both 128, the
     * compiled row, and what the `head_k`/`head_v` parameters mean); `nkg`/`nhv` are the number of key
     * and value heads, which is a property of the state cache. Collapsing the two -- reading the
     * grid and every per-head offset off a width -- is a device memory fault, not a wrong number. */
    int     kw, vw, nkg, nhv;
    /* Slot counts, so the metadata gate can refuse an index the caches cannot back. */
    int     slots, cslots, cdepth, cdim;
    float   scale, eps, l2_eps, softplus_thr;
};

/* libr4d's conv width: fixed there by r4d_gdn_conv_tap.h:8, and by both rows' constraints. */
#define BOLE_CONV_W 4
#define BOLE_CONV_HIST (BOLE_CONV_W - 1)

/* The factor width the verify row writes and the commit row reads:
 * [normalized k(128) | conv v(128) | gamma | beta], all F32.
 *
 * This is the vb-exact-commit variant's contract, and the reason it is the variant ported. It does
 * NOT carry U: a verifier whose U bits differ from the serial path leaves a residual in the
 * committed anchor when the commit replays that U -- the archived adjoint's round-2 zero-group
 * failure. The commit recomputes u from the original update data with the engine's own FP32
 * instructions instead, which is what makes its output byte-identical. */
#define BOLE_FACTOR_W (2 * 128 + 2)

/* Tree node budget: a root plus fifteen proposals, the source's RAD_GDN_TREE_CAP. Both rows refuse a
 * wider batch rather than truncate it. */
#define BOLE_NODES 16

/* Nodes whose factors the commit stages in LDS at a time. */
#define BOLE_COMMIT_TILE 8

/* The conv history workgroups the commit launches beside its value heads: one per 512 channels of a
 * packed [q|k|v] row, and R4D's largest GDN width is 512 channels (2*128 + 2*128 for four value
 * heads of a two-key-head group is the engine's own 384; four is the headroom the launcher checks
 * against the real channel count before using it). */
#define BOLE_COMMIT_HIST_WG 4

/* libr4d's recurrent-block shape: 128 value rows, two k halves. Declared here because the host row
 * uses it to bound the projection width the history arm can cover, not only the kernel. */
#define BOLE_THR 256

/* The launches, defined beside the kernels in bole_tree.hip and called from bole_rows.cpp with the
 * flattened call the host built. A failure is a RAD_E_* status, never a partial success. */
int bole_verify_launch(const BoleCall* f, void* stream);
int bole_commit_launch(const BoleCall* f, void* stream);

#endif /* BOLE_LAYOUT_H */
