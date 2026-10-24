#ifndef RAD_MINE_DFLASH_CONV_ADD_H
#define RAD_MINE_DFLASH_CONV_ADD_H

#include "rad_plugin.h"
#include <stdint.h>

/* dflash_conv_add -- DFlash2's closing grouped convolution plus the residual add, in one launch.
 *
 *   h2 = dflash_conv(h, delta, base_row)      the rounded bf16 convolution output
 *   x  = x + h2                               the residual add, in place
 *
 * This is `dflash_conv -> add` and NOTHING ELSE. It exists because the unfused pair materialises
 * an [T, n_embd] bf16 plane that the add reads back exactly once, and because the block issues
 * that pair twice a layer, five layers deep (arch/common/rad_block_dflash2.h:1124-1126,
 * :1151-1153).
 *
 * ============================== WHAT IT IS NOT ==============================
 *
 * NOT A TOLERANCE. The unfused pair rounds the convolution's f32 accumulator to bf16 and THEN
 * adds, and that rounding is part of the function the model was trained with -- it is not
 * intermediate precision a fusion may keep in f32 to be "more accurate". The kernel stores
 * `to_bf16(to_f32(residual) + to_f32(to_bf16(acc)))`: two bf16 roundings, in the unfused order.
 *
 * NOT AN ATTENTION FUSION. The all-reduce that precedes it in the graph stays where it is: this op
 * is issued only after the TP reduction, and it neither reads nor writes any collective.
 *
 * NOT A SECOND CONVOLUTION. taps 2 and group_size 16 are the DFlash2 geometry and the only one
 * compiled; `rad_dflash_conv_add_args` refuses every other, and the row's constraints say so
 * before selection rather than after it. There is no non-vector tail: hidden_size is a whole
 * number of 8-channel lanes and a multiple of 64, which is the unfused row's own band.
 *
 * ============================== THE LAYOUT ==============================
 *
 * The ARITHMETIC is `libr4d/r4d_dflash_conv_body.h`'s, expression for expression: one lane owns
 * eight contiguous channels, the group index -- and both delta scalars -- are uniform over that
 * span because group_size 16 divides it, the position inside the draft block is `t % block_size`
 * (wave-uniform, from grid.y), and tap `k` contributes only where that position is at least `k`.
 * The loop STRUCTURE differs in one respect, recorded below: the shifted addresses are formed
 * inside the tap predicate. That is an addressing difference, not an arithmetic one -- no operand
 * value, operation or rounding changes -- and it has not been compiled, so it is not a claim about
 * the resulting instruction stream.
 *
 * THE ROWS ARE PITCHED, AND SO IS THE OUTPUT. `x`, `residual` and `out` each carry their own
 * row pitch, read off the operand: the add's call site hands the residual stream itself, and the
 * convolution's hands a row range of the activation buffer. The delta operand is the SLICE of the
 * [T, 2, taps, NG] kernel projection as issued -- its row pitch is 2*taps*NG, not taps*NG, which
 * is a difference no shape check would catch and which silently reads the other half's
 * coefficients.
 *
 * ============================== ALIASING AND `restrict` ==============================
 *
 * `out` MAY BE `residual`, as the SAME view (same base, same pitch): that is the in-place call
 * arch/common/rad_block_dflash2.h:1126 issues. A partial overlap is refused, because a lane's
 * store landing on another lane's unread residual is not ordered by anything.
 *
 * `out` MUST NOT overlap `x`, `delta` or `base`. That is the only property the kernel relies on:
 * every read of those three happens before the store.
 *
 * THE DEVICE ARGUMENT STRUCT CARRIES NO `restrict` AT ALL. `restrict` promises pairwise
 * disjointness among the qualified pointers, and the validation above proves only output-vs-input
 * disjointness while explicitly allowing the in-place form -- so qualifying anything would be a
 * promise the kernel breaks on its most common call. The alternative, proving all pairs disjoint,
 * would forbid the in-place call the graph actually makes.
 *
 * The second tap's pointers (`xrow - tap*pitch`, and the base/delta rows) are formed ONLY on the
 * rows where the tap fires, so row 0 never forms an address before its object.
 *
 * ============================== OWNERSHIP ==============================
 *
 * `base` is a WEIGHT operand, exactly as it is on `dflash_conv`: base_kernel is a checkpoint
 * parameter and the role is what gives it a residency class. The two halves of the [2, taps,
 * n_embd] tensor are two issues of the same op against `wrow_slice(...)`, and the op receives one
 * row of that slice, contiguous.
 *
 * Nothing is allocated and nothing synchronises: one launch, one argument struct by value, and
 * the only host work is the validation below, which dereferences no device memory. */
enum {
    RAD_DFLASH_CONV_ADD_X        = 0,
    RAD_DFLASH_CONV_ADD_DELTA    = 1,
    RAD_DFLASH_CONV_ADD_BASE     = 2,
    RAD_DFLASH_CONV_ADD_RESIDUAL = 3,
    RAD_DFLASH_CONV_ADD_OUT      = 4,
    RAD_DFLASH_CONV_ADD_OPERANDS = 5
};

/* Validated, device-free view of one call. Shape, stride and alias checks only: nothing here
 * dereferences a pointer, because at issue every operand is device memory. */
struct RadDflashConvAddCall {
    const void* x;
    const void* delta;
    const void* base;
    const void* residual;
    void*       out;
    int64_t     rows;        /* the operand's rows, not the T band ceiling */
    int64_t     H;           /* hidden_size */
    int64_t     NG;          /* H / group_size */
    int64_t     dpitch;      /* delta's row pitch: 2*taps*NG */
    int64_t     x_pitch;     /* stride[0] of each activation row view */
    int64_t     r_pitch;
    int64_t     o_pitch;
    int64_t     block_size;  /* the draft block: the position law is t % block_size */
};

int rad_dflash_conv_add_schema_count(void);
const RadOpSchema* rad_dflash_conv_add_schema_at(int);
int rad_dflash_conv_add_shape(const RadParam*, int, int, RadOpdDesc*);
/* RAD_OK, or the reason this call is not the op: RAD_E_UNSUPPORTED outside taps 2 / group 16 /
 * the declared width band, RAD_E_SHAPE for extents or pitches that disagree with the parameters,
 * RAD_E_DTYPE for anything but bf16, RAD_E_ALIGN for a row the 16-byte loads cannot address, and
 * RAD_E_INVAL for a required operand absent or an output aliasing the convolution's reads. */
int rad_dflash_conv_add_args(const RadArgs*, RadDflashConvAddCall*);
/* The declared chain -- `dflash_conv` then `add`. It is a DESCRIPTION of what this kernel
 * computes, and it is not checkable by stock rad-kbench: that tool resolves each step inside the
 * fused row's own plugin, and `mine` publishes no `dflash_conv` and no `add` row. Parity must be
 * established by a comparator that names this row and each unfused reference row explicitly. */
int rad_dflash_conv_add_fuse(const RadParam*, int, uint64_t, int, RadFuseStep*);
RadKernelInfo rad_dflash_conv_add_row(RadLaunchFn);
int rad_mine_dflash_conv_add_count(void);
const RadKernelInfo* rad_mine_dflash_conv_add_at(int);

#endif /* RAD_MINE_DFLASH_CONV_ADD_H */
