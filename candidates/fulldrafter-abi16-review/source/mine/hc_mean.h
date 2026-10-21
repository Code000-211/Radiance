#ifndef RAD_MINE_HC_MEAN_H
#define RAD_MINE_HC_MEAN_H

#include "rad_plugin.h"
#include <stdint.h>

/* hc_mean -- the DFlash2 tap: x[m,i] = mean over the hc streams of h[m, c*N+i], summed in f32 and
 * stored bf16 once. The host reference and the definition of the op are libref's
 * (libref/ref_hc.cpp); this is mine's device specialisation of the same schema.
 *
 * NOT THE GATED RESIDUAL'S READ. No RMS, no gate, no weight, no read of a block input: a caller
 * that substituted hc_read would normalise and gate the streams and divide by a different number,
 * and every tap would differ.
 *
 * THE OUTPUT IS A COLUMN VIEW. The caller passes `x` as width N at column t*N of a [M, n_taps*N]
 * tap plane, which reaches the kernel as a rank-2 tensor whose ROW pitch is stride[0] -- the
 * PLANE's width -- while stride[1] is the column step, 1. The store goes through the row pitch, so
 * the other taps are never touched.
 *
 * The row is specialised at hc 4 and N a multiple of 8 -- the DFlash2 geometry -- and the launch
 * refuses anything it was not given: bf16 only, rank 2 only, declared widths only, unit column
 * stride, and a row pitch that cannot make rows overlap. No allocation and no synchronisation on
 * the launch path. */
enum { RAD_HC_MEAN_IN = 0, RAD_HC_MEAN_OUT = 1, RAD_HC_MEAN_OPERANDS = 2 };

/* Validated, device-free view of one call. Shape, stride and alias checks only: nothing here
 * dereferences a pointer, because at issue every operand is device memory. */
struct RadHcMeanCall {
    const void* h;
    void*       x;
    int64_t     rows, N;
    int64_t     h_pitch, x_pitch;   /* stride[0]: elements between consecutive rows */
};

int rad_hc_mean_schema_count(void);
const RadOpSchema* rad_hc_mean_schema_at(int);
int rad_hc_mean_shape(const RadParam*, int, int, RadOpdDesc*);
/* RAD_OK, or the reason this call is not the op: RAD_E_UNSUPPORTED outside the declared geometry,
 * RAD_E_SHAPE for a width that disagrees with the parameters, RAD_E_DTYPE for anything but bf16,
 * RAD_E_INVAL for a required operand absent or an output aliasing the input. */
int rad_hc_mean_args(const RadArgs*, RadHcMeanCall*);
RadKernelInfo rad_hc_mean_row(RadLaunchFn);
int rad_mine_hc_mean_count(void);
const RadKernelInfo* rad_mine_hc_mean_at(int);

#endif /* RAD_MINE_HC_MEAN_H */
