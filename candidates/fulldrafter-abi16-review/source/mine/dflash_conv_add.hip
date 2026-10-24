/* dflash_conv_add.hip -- DFlash2's closing grouped convolution and its residual add, one launch.
 * The definition of the fusion is mine/dflash_conv_add.h; this file is the device form.
 *
 * ============================== THE ARITHMETIC IS libr4d's ==============================
 *
 * The convolution below is `libr4d/r4d_dflash_conv_body.h`'s loop, and its primitives --
 * `R4dV8`, `r4d_bf2f`, `r4d_f2bf` -- are that header's own, included rather than re-spelled. What
 * is repeated is the loop structure, because the body stores its accumulator straight to `out` and
 * this op has to keep the eight f32 values long enough to round them, read the residual, and round
 * again. There is no way to redirect the body's store into a register block without pointing `out`
 * at a per-lane buffer it computes the address of itself, and a wild pointer is not a saving.
 *
 * THE DUPLICATION IS NOT SELF-VERIFYING. The row declares `replaces` = dflash_conv -> add, and
 * that declaration is not executable by stock rad-kbench -- the tool resolves every chain step
 * inside the fused row's own plugin, and this plugin publishes neither of those rows. So nothing
 * in this file's structure proves the loops agree: parity has to come from a comparator that
 * selects this row by name and each unfused reference row (libr4d's `dflash_conv_t2_g16_bf16` and
 * `add_bf16`) explicitly, on the same bytes, with zero tolerance. Until that runs, no parity claim
 * rests on this file.
 *
 * ============================== TWO ROUNDINGS, IN THE UNFUSED ORDER ==============================
 *
 *     h2  = to_bf16(acc)                       <- the bf16 plane the pair materialises
 *     out = to_bf16(to_f32(residual) + to_f32(h2))
 *
 * The second line is `add_bf16` (`binary_kernel` in libr4d/r4d_model_bf16.hip: read both in f32,
 * add, round once) with `h2` already rounded. Carrying `acc` in f32 into the add instead would be
 * a different function -- more accurate, and not the one the model was trained with.
 *
 * ============================== THE LAYOUT ==============================
 *
 * A lane owns eight contiguous channels: one dwordx4 load of `x`, of `base`, of the residual and
 * one dwordx4 store. group_size 16 divides that span, so the group index and both delta scalars
 * are uniform over the lane and are loaded once per row rather than per channel. The position
 * inside the draft block is `t % block_size`, one scalar remainder per workgroup, and tap `k`
 * contributes only where that position is at least `k` -- which is what keeps one sequence's
 * block from convolving with the previous sequence's rows.
 *
 * EVERY ROW VIEW IS PITCHED: `x`, the residual and `out` arrive with their own row pitch, and the
 * delta with the projection's (`2*taps*NG`, not the slice's `taps*NG`).
 *
 * NOTHING IS ALLOCATED AND NOTHING SYNCHRONISES: one launch, one argument struct by value.
 */
#include <hip/hip_runtime.h>

#include "dflash_conv_add.h"
#include "../libr4d/r4d_dflash_conv_body.h"

namespace {

constexpr int kThreads = 128;
constexpr int kVec     = 8;

/* The specialisation the row declares, as compile-time constants: `rad_dflash_conv_add_args` has
 * already refused every other geometry, so the loop bounds are literal and the group index is a
 * shift. */
constexpr int kTaps  = 2;
constexpr int kGroup = 16;

struct Args {
    /* NO `restrict` ON ANY OF THESE, AND THAT IS THE CONTRACT RATHER THAN AN OVERSIGHT.
     *
     * `restrict` promises no overlap with ANY other restrict-qualified pointer in the scope. The
     * host validation proves only what this kernel actually needs: that the output is disjoint
     * from `x`, `delta` and `base`, which is what makes "every read happens before the store"
     * sufficient. It proves nothing about the reads aliasing EACH OTHER, and it deliberately
     * permits `out == residual`, which is the in-place form rad_block_dflash2.h:1126 issues.
     * Qualifying any member here would be a promise the kernel breaks on its most common call.
     *
     * The cost is a pointer kept live instead of a base+offset reconstruction, and it is not
     * measured: no build was run. If a later code object shows it matters, the honest fix is to
     * strengthen rad_dflash_conv_add_args to prove pairwise disjointness -- not to write
     * `restrict` and hope. */
    const r4d_u16* x;
    const r4d_u16* delta;
    const r4d_u16* base;
    const r4d_u16* residual;
    r4d_u16*       out;
    int H, dpitch, NG, vpr, block_size;
    int x_pitch, r_pitch, o_pitch;
};

__global__ __launch_bounds__(kThreads) void conv_add_kernel(Args a) {
    const int hv = blockIdx.x * kThreads + threadIdx.x;
    if (hv >= a.vpr) return;
    const int t = blockIdx.y;
    const int bpos = t % a.block_size;

    /* Split every address into a wave-uniform pointer and a lane offset the backend can keep in a
     * single 32-bit VGPR -- gfx12's saddr form. The mask is load-bearing: the backend only folds a
     * lane offset into saddr when it can prove the byte-scaled offset stays inside 32 bits, and a
     * lane index built from workgroup_id has no known bound. The same two rules the shared body
     * states: pointer + offset, never a vector-typed array indexed by lane. */
    const unsigned off = ((unsigned)hv & 0xFFFFFu) * (unsigned)kVec;
    const unsigned g   = off / (unsigned)kGroup;

    const r4d_u16* xrow = a.x + (long)t * a.x_pitch;
    const r4d_u16* rrow = a.residual + (long)t * a.r_pitch;
    r4d_u16*       orow = a.out + (long)t * a.o_pitch;
    const r4d_u16* drow = a.delta + (long)t * a.dpitch;

    R4dV8 x0; x0.v = *(const r4d_u32x4*)(xrow + off);
    R4dV8 b0; b0.v = *(const r4d_u32x4*)(a.base + off);
    const float d0 = r4d_bf2f(drow[g]);

    float acc[kVec];
#pragma unroll
    for (int j = 0; j < kVec; ++j)
        acc[j] = (r4d_bf2f(b0.h[j]) + d0) * r4d_bf2f(x0.h[j]);

    /* Tap `k` contributes only where the position within the block is at least `k`. `bpos` and `t`
     * are wave-uniform, so the predicate is a scalar branch and the shifted load is not issued on
     * the rows it skips.
     *
     * THE SHIFTED POINTER IS FORMED INSIDE THE PREDICATE, unlike the shared body, which hoists it
     * for saddr reasons. Hoisting computes `x - k*pitch` on row 0 of every sequence, i.e. an
     * address before the object -- harmless in practice on this target, UB in the abstract, and
     * refused here rather than rationalised. The same deferral applies to `btap`/`dtap`: they are
     * in bounds (`base` is `taps*H` long, the delta row `taps*NG` wide) but forming them only on
     * the taken path keeps one rule for all three. Whether the deferral costs the saddr fold is a
     * question for a code object, and none was built. */
#pragma unroll
    for (int tap = 1; tap < kTaps; ++tap) {
        if (bpos >= tap) {
            const r4d_u16* xprev = xrow - (long)tap * a.x_pitch;
            const r4d_u16* btap  = a.base + (long)tap * a.H;
            const r4d_u16* dtap  = drow + (long)tap * a.NG;
            R4dV8 xs; xs.v = *(const r4d_u32x4*)(xprev + off);
            R4dV8 bt; bt.v = *(const r4d_u32x4*)(btap + off);
            const float dt = r4d_bf2f(dtap[g]);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                acc[j] += (r4d_bf2f(bt.h[j]) + dt) * r4d_bf2f(xs.h[j]);
        }
    }

    /* THE TWO ROUNDINGS. `h2` is the value the unfused pair writes to memory; the add then reads
     * it back as a bf16 and rounds the sum. The residual is loaded and consumed BEFORE the store,
     * which is what makes `out == residual` an in-place update rather than a read of its own
     * output -- and the reason neither pointer carries `restrict`. */
    R4dV8 res; res.v = *(const r4d_u32x4*)(rrow + off);
    R4dV8 o;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
        const r4d_u16 h2 = r4d_f2bf(acc[j]);
        o.h[j] = r4d_f2bf(r4d_bf2f(res.h[j]) + r4d_bf2f(h2));
    }
    *(r4d_u32x4*)(orow + off) = o.v;
}

}  /* namespace */

/* The launch. `rad_dflash_conv_add_args` has already refused every geometry and every view this
 * kernel was not given, so there is no second validation here and no path that computes
 * something the oracle would not: either the call is the op or it returned a reason. */
int rad_mine_dflash_conv_add_launch(const RadArgs* a, RadStream stream) {
    RadDflashConvAddCall call{};
    const int rc = rad_dflash_conv_add_args(a, &call);
    if (rc != RAD_OK) return rc;

    Args args{};
    args.x        = static_cast<const r4d_u16*>(call.x);
    args.delta    = static_cast<const r4d_u16*>(call.delta);
    args.base     = static_cast<const r4d_u16*>(call.base);
    args.residual = static_cast<const r4d_u16*>(call.residual);
    args.out      = static_cast<r4d_u16*>(call.out);
    args.H        = (int)call.H;
    args.dpitch   = (int)call.dpitch;
    args.NG       = (int)call.NG;
    args.vpr      = (int)(call.H / kVec);
    args.block_size = (int)call.block_size;
    args.x_pitch  = (int)call.x_pitch;
    args.r_pitch  = (int)call.r_pitch;
    args.o_pitch  = (int)call.o_pitch;

    const unsigned gx = (unsigned)((args.vpr + kThreads - 1) / kThreads);
    const unsigned gy = (unsigned)call.rows;
    void* kargs[] = { &args };
    const hipError_t e = hipLaunchKernel((const void*)conv_add_kernel, dim3(gx, gy),
                                         dim3(kThreads), kargs, 0, (hipStream_t)stream);
    return e == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_dflash_conv_add_count(void) { return 1; }

const RadKernelInfo* rad_mine_dflash_conv_add_at(int i) {
    static const RadKernelInfo row = rad_dflash_conv_add_row(rad_mine_dflash_conv_add_launch);
    return i == 0 ? &row : nullptr;
}
