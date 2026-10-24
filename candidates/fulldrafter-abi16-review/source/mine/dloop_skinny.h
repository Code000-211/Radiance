#ifndef RAD_MINE_DLOOP_SKINNY_H
#define RAD_MINE_DLOOP_SKINNY_H
/* Skinny GEMM of the full-context decode shape: at most 64 output rows, one lane per row, one wave
 * per column group, weights read wave-uniformly out of global memory.
 *
 * WHY THIS EXISTS. The two register-tiled kernels this file complements stage both operands through
 * LDS, which is what a 64-row x 64-column tile wants and what an 18-row slab wastes: every lane then
 * walks the SAME weight address, so the load broadcasts and no staging, barrier or reduction is
 * needed. The row threshold is measured, not assumed -- docs/dloop_speed/W1_skinny_gemm.md carries the
 * crossover and the per-shape numbers.
 *
 *   lane = row (row = blockIdx.y*32 + lane), wave = TN output columns, 4 waves per 128-thread CTA.
 *   No LDS, no barrier, no reduction: the K loop of a wave is one serial FP32 chain per output
 *   column, ascending in k, one multiply and one add per product. The plugin builds with
 *   -ffp-contract=off, so `a*b+c` keeps its two roundings and the chain is element for element the
 *   host reference's `z+=x[k]*bf(w[k])`.
 *
 * The kernel takes the full path's GemmReq directly: `kind` picks the weight mapping and `source` the
 * activation source, both exactly as dloop_full_device.h defines them. A request outside the shape
 * class, or one whose layout cannot satisfy the vector loads, stays with the calling file's own
 * register-tiled kernel. */
#include <cstdint>
#include <hip/hip_runtime.h>
#include "dloop_full_device.h"

namespace rad_dloop {
/* Rows above this keep the register-tiled kernel: it stages the activation once for a 64-row tile and
 * shares it across the CTA, which is what a second row tile starts to need. */
constexpr int skinny_rows_max=64;

/* Elements per convolution group in the fused producer: sixteen, the width convolve_column's vector
 * body is written for and the divisor that makes `hidden/16` the dynamic projection's group count. */
constexpr int skinny_dyn_group=16;

/* The skinny kernel reads its operands with vector loads, so the layout has to satisfy them. The
 * rejected cases are the ring gather (source 1), the append's dead-row exit (`live`), rows past the
 * threshold, and every misalignment -- all of which the tiled kernel serves without a vector load. */
inline bool skinny_fits(const GemmReq& r) {
    if(r.rows>skinny_rows_max||r.live)return false;
    if(r.source!=0&&r.source!=2)return false;
    if((r.in&7)!=0)return false;
    if((reinterpret_cast<uintptr_t>(r.w0)|reinterpret_cast<uintptr_t>(r.w1)|
        reinterpret_cast<uintptr_t>(r.w2))&15)return false;
    if(r.source==2)return !((reinterpret_cast<uintptr_t>(r.anchor)|
        reinterpret_cast<uintptr_t>(r.embed))&7);
    return !(r.xstride&3)&&!(reinterpret_cast<uintptr_t>(r.x)&15);
}
/* `fuse` reproduces the accumulation of the kernel this one replaces: false for the tree policy (its
 * register-tiled kernel accumulates with `a*b+c`, two roundings), true for the base policy (its tiled
 * kernel uses an explicit fmaf, one rounding). See skinny_k8 in dloop_skinny.hip. */
int skinny_gemm_launch(const GemmReq& r,bool fuse,hipStream_t stream);

/* -------------------------------------------------- dyn (kernel) GEMM + group convolution
 *
 * conv_pre's TWO kernels in ONE launch:
 *      delta[row][j] = sum_k norm[row][k]*KERNEL[j][k]        j < 4*(hidden/16)
 *      pre[row][f]   = sum_taps base[tap*h+f]*norm[row-tap][f]
 *                             + delta[row][(2*tap+branch)*ng+f/16]*norm[row-tap][f]
 * A group's delta columns are {g, ng+g, 2ng+g, 3ng+g} with ng = hidden/16, which is why
 * `out = 4*(in/16)` is the only shape this producer serves; phase 0 (pre) uses the first two, the
 * finish convolution the last two.
 *
 * ARITHMETIC. delta reuses skinny_k8's own chain (ascending k, FUSE = the replaced tiled kernel's
 * rounding); pre calls convolve_column's own conv_one, and `two` is its `taps>1`, so a job's row 0
 * skips the second tap pair instead of adding a zero term.
 *
 * VALIDITY. delta is stored for every covered row -- an invalid sequence's row keeps the zeros its
 * zeroed accumulator gives today -- while w.pre is written only for a valid sequence's row, because
 * full_conv_pre returns before its first store. Rows >= rows are touched by neither.
 *
 * false / RAD_E_UNSUPPORTED means "run the original gemm + conv kernel", never a partial run. */
inline bool skinny_dyn_fits(const GemmReq& r,const RadDLoopConfig& c) {
    if(!r.x||!r.w0||!r.y||!r.valid||!r.rows_per_seq||r.rows_per_seq<0)return false;
    if(r.kind!=0||r.source!=0||r.live)return false;      /* one weight matrix, dense F32 rows, no dead-row exit */
    if(c.context_window!=0)return false;                 /* the full-context arena is the only one this reads */
    if(r.rows<=0||r.rows>skinny_rows_max)return false;   /* the decode shape, as skinny_rows_max defines it */
    if(r.xstride!=r.in||(r.in&7)!=0)return false;        /* vector loads need the dense, 8-multiple row */
    if(r.in!=c.hidden_size)return false;                 /* base rows are h wide and the stride above is in */
    if(c.conv_group_size!=16||r.in<16||(r.in&15)!=0)return false;
    if(r.out!=4*(r.in/16))return false;                  /* the four columns per group ARE the shape */
    return !((reinterpret_cast<uintptr_t>(r.x)|reinterpret_cast<uintptr_t>(r.w0))&15);
}
/* `fuse` is skinny_k8's switch and applies to the delta GEMM only (the base file's tiled kernel
 * accumulates with fmaf, the tree file's with mul+add); the convolution is mul+add either way. */
int skinny_dyn_launch(const GemmReq& r,const DeviceArgs& a,int layer,int mlp,bool fuse,hipStream_t stream);
} /* namespace rad_dloop */
#endif
