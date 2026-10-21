// The QSA indexer's TAIL: the raw index keys of the tokens that have not yet completed a block.
//
// A block key is the mean of `ratio` consecutive raw keys, so a block cannot be built until its
// last token arrives -- and at decode three of the four are from earlier steps. This is where they
// wait.
//
// IT IS A LINEAR KV STATE AND NOT A PLANE INDEXED BY THE BATCH ROW. A sequence owns one state per
// bound layer, the engine keeps it for that sequence's life, zeroes it when it hands one out and
// checkpoints it with the rest of the linear state; the batch row is step-local and names a
// different sequence from one step to the next. arch/common/rad_qsa.h's Wire is the argument, and
// it is the same one that makes this per LAYER: the keys are `index_qk_proj`'s output.
//
// `state` is the group's cache for ONE layer, `[slots, 1, ring, n]`, and `sidx` names this
// sequence's slot in it. A token at position `p` lands at row `p % ring`.
//
// THE RING IS `ratio + n_spec`, NOT `ratio`, AND SPECULATION IS THE WHOLE OF IT. A verify step
// runs 1 + n_spec positions and the next step restarts at the first REJECTED one, so a position
// this step has already overwritten is owed to the next. At ratio 4 depth 3 the step covering
// positions 9..12 writes rows 1,2,3,0 -- and row 0 is the one position 8 is still owed, because
// the block holding 8..11 is rebuilt as soon as the step that follows starts at 10. The widest
// window a step can still owe is [p0 - ratio + 1, p0 + n_spec]; a ring that long means a row is
// only ever overwritten by a position past the end of any step that could ask for it.
//
// WHICH SEQUENCE A TOKEN BELONGS TO IS `cu_seqlens`, and that is the whole reason this op takes a
// boundary list. The batch's other per-token index is `slot_mapping`, which is a FLAT KV slot and
// says nothing about the sequence; the boundary list does, and with at most a handful of sequences
// a linear scan of it is free beside the copy it guards.
//
// ONLY THE LAST `ring` TOKENS OF A SEQUENCE'S CHUNK WRITE, AND THAT IS NOT AN OPTIMISATION. A
// prefill chunk carries many tokens per sequence and positions `p` and `p + ring` land on the
// SAME row, so a launch that wrote every token would have two workgroups racing for one row and
// the winner would depend on the schedule. Restricting the write to the last `ring` tokens makes
// every row the target of at most one token -- their positions differ by less than `ring`, so
// their rows differ -- and the result is what a token-by-token loop would have left. The older
// tokens of the chunk are not lost: the blocks they complete are built from the chunk's own keys,
// which are still in the projection's output.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

__global__ __launch_bounds__(kThreads) void qsa_tail_store_kernel(
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ sidx,
    uint16_t* __restrict__ state,
    int T, int n, int ring, int seqs, int k_ld, int s_slot, int s_row, int sidx_ld) {

    const int r = (int)blockIdx.x;
    const int s = (int)blockIdx.y;
    if (s >= seqs || r >= ring) return;

    /* Only the last `ring` tokens of this sequence's chunk -- see the header.
     * CTAs are launched on a (ring, seqs) 2D grid, indexing directly into the sequence's
     * tail window [cu[s + 1] - ring, cu[s + 1]) instead of launching T CTAs and discarding
     * non-writing prefill tokens via a linear scan. */
    const int s_start = cu[s];
    const int s_end = cu[s + 1];
    const int t = s_end - ring + r;
    if (t < s_start || t >= s_end || t >= T) return;

    /* COLUMN 0 IS THE COMMITTED SLOT. -1 is the pad a sequence with no state carries, and every
     * kernel that reads a state row already tests for it. */
    const int si = sidx[(size_t)s * sidx_ld];
    if (si < 0) return;

    const int row = pos[t] % ring;
    const size_t src = (size_t)t * k_ld;
    const size_t dst = (size_t)si * s_slot + (size_t)row * s_row;
    for (int d = (int)threadIdx.x; d < n; d += kThreads) state[dst + d] = k[src + d];
}

}  // namespace

//   k      [T][n]                bf16, `k_ld` elements a row (0 = n)
//   pos    [T]                   i32, each token's position in its sequence
//   cu     [seqs + 1]            i32, cu_seqlens
//   sidx   [seqs][sidx_ld]       i32, the tail group's state row; column 0 is the committed slot
//   state  [slots][ring][n]      bf16, this layer's tail cache -- READ NOWHERE HERE, written in
//                                place, and rows nothing targets keep what they held
static int r4d_qsa_tail_store_bf16_emit(long k, long pos, long cu, long sidx, long state, int T,
                                       int n, int ring, int seqs, int k_ld, int s_slot, int s_row,
                                       int sidx_ld, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!k || !pos || !cu || !sidx || !state) return -1;
    if (T <= 0 || n <= 0 || ring <= 0 || seqs <= 0) return -2;
    if (!k_ld) k_ld = n;
    if (!s_row) s_row = n;
    if (!sidx_ld) sidx_ld = 1;
    emit_rc = R4D_EMIT(sk, qsa_tail_store_kernel, dim3((unsigned)ring, (unsigned)seqs), dim3(kThreads), 0, (const uint16_t*)k, (const int32_t*)pos,
                       (const int32_t*)cu, (const int32_t*)sidx, (uint16_t*)state, T, n, ring,
                       seqs, k_ld, s_slot, s_row, sidx_ld);
    return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
}

extern "C" int r4d_qsa_tail_store_bf16_emit(long k, long pos, long cu, long sidx, long state, int T,
                                       int n, int ring, int seqs, int k_ld, int s_slot, int s_row,
                                       int sidx_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return r4d_qsa_tail_store_bf16_emit(k, pos, cu, sidx, state, T, n, ring, seqs, k_ld, s_slot, s_row, sidx_ld, sk);
}

// ============================================================== the plugin ABI entry point

enum { QT_K = 0, QT_POS = 1, QT_CU = 2, QT_STATE = 3, QT_SIDX = 4 };

static int r4d_rad_qsa_tail_store_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* k  = r4d_opt(a, QT_K);
    const RadTensor* po = r4d_opt(a, QT_POS);
    const RadTensor* cu = r4d_opt(a, QT_CU);
    const RadTensor* st = r4d_opt(a, QT_STATE);
    const RadTensor* sx = r4d_opt(a, QT_SIDX);
    if (!k || !po || !cu || !st || !sx) return R4D_NO(RAD_E_INVAL);
    if (k->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* THE STATE ARRIVES AS THE GROUP DECLARED IT: [slots, heads, ring, n], heads 1. The engine
     * composes that shape out of the declaration and the manager's slot count
     * (core/engine_bringup.cpp), so a different rank here means a different group, not a layout
     * this kernel could have coped with. */
    if (st->rank != 4 || st->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (st->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long ring  = r4d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R4D_NO(RAD_E_INVAL);
    const long long T = k->shape[0];
    if (T < 1 || k->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (st->shape[2] < ring || st->shape[3] != n) return R4D_NO(RAD_E_SHAPE);
    /* `cu_seqlens` IS seqs + 1 LONG AND THAT IS WHERE THE SEQUENCE COUNT COMES FROM. Taking it
     * from the state's slot count instead would read the DECLARED max_seqs, and a step with fewer
     * sequences than that would scan boundaries the batch never wrote. */
    const long long seqs = r4d_numel(cu) - 1;
    if (seqs < 1 || r4d_numel(po) < T) return R4D_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k);
    if (kp < 0) return R4D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || st->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_qsa_tail_store_bf16_emit(r4d_addr(k), r4d_addr(po), r4d_addr(cu), r4d_addr(sx),
                                          r4d_addr(st), (int)T, (int)n, (int)ring, (int)seqs,
                                          (int)kp, (int)st->stride[0], (int)st->stride[2],
                                          (int)sxp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

R4D_ENTRY(r4d_rad_qsa_tail_store, r4d_rad_qsa_tail_store_body)
