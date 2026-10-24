// Baseline side of the byte-exactness comparison: this unit inlines the header as it stands at
// the PR's base commit, renamed so it can share a binary with the ported one below.
//
// The include resolves through -I, so the build script points this unit at a checkout of the base
// header and the candidate unit at this tree's libr4d.
#define r4d_ar_hc_write_word hc_baseline_word
#include "r4d_ar_hc.h"
#undef r4d_ar_hc_write_word

// Same signature, same launch shape, same arguments as hc_candidate: the only difference between
// the two kernels is the body of r4d_ar_hc_write_word.
extern "C" __global__ void hc_baseline(const unsigned short* inj, unsigned short* h,
    const unsigned short* y, int n, int hc, long hrow, long irow, long yrow) {
    const int i0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i0 >= n) return;
    const int row = (int)blockIdx.y;
    hc_baseline_word(inj, h, row, i0, n, hc, irow, hrow, y + row * yrow + i0);
}
