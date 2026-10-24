// Candidate side of the byte-exactness comparison: this unit inlines this tree's header, i.e. the
// ported r4d_ar_hc_write_word. It is the same kernel as hc_baseline apart from that body.
#define r4d_ar_hc_write_word hc_candidate_word
#include "r4d_ar_hc.h"
#undef r4d_ar_hc_write_word

extern "C" __global__ void hc_candidate(const unsigned short* inj, unsigned short* h,
    const unsigned short* y, int n, int hc, long hrow, long irow, long yrow) {
    const int i0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i0 >= n) return;
    const int row = (int)blockIdx.y;
    hc_candidate_word(inj, h, row, i0, n, hc, irow, hrow, y + row * yrow + i0);
}
