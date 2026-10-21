#pragma once
// r4d_ar_hc.h: per-word arithmetic of the all-reduces fused with the gated residual write: the MoE
// gather before the message and the hc write after the reduction. Shared by the exact kernels
// (r4d_ar_oneshot_2rank_exact.hip) and the rotated 6-bit ones (r4d_ar_oneshot_2rank_wht6.hip), so
// both fused forms produce the same bytes as the unfused chain moe_gather, all-reduce, hc_write.
//
// A word is eight consecutive bf16 elements of one row. A row is `n` elements and n is a multiple
// of eight, so a word never straddles a row.
#include <hip/hip_runtime.h>
#include "r4d_gdn_wmma.h"   // f2bf / bf2f: the gated residual's own bf16 arithmetic

// LDS slot table bounds: rows per block, and slots per row.
#define R4D_ARG_ROWS  8
#define R4D_ARG_SLOTS 64

__device__ __forceinline__ unsigned short arg_bf16_rne(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct R4dArGather {
  const unsigned short* ye;   // [T, ye_ld] the routed experts' outputs, sorted-row order
  const unsigned short* ew;   // [M, ew_ld] each token's slot weights
  const int* sorted;          // [T] the (token * top_k + slot) each sorted row holds
  const unsigned short* sh;   // [M, sh_ld] the shared arm, or null
  const unsigned short* sg;   // [M, sg_ld] its gate, or null
  long ye_ld, ew_ld, sh_ld, sg_ld;
  int top_k, T, sig;
};

// One word of the gather: y[row][i0 .. i0+8) = sum over the row's slots, ascending, of
// w[row][j] * y_expert[slot][i0 ..], rounded to bf16, with the shared arm folded in as moe_gather
// does it. Its narrows are round-to-nearest-even, unlike hc_write's.
//
// With INV, slots come from the inverse table inv[token * top_k + slot] (the sorted row holding
// it, or -1) and weights and gate from global memory. Without it they come from the block's LDS
// table, where `r` is the row's index.
template <bool INV>
__device__ __forceinline__ int4 r4d_ar_gather_word(const R4dArGather& G,
                                                   const int* __restrict__ inv, const int* s_i,
                                                   const float* s_w, const float* s_g, int row,
                                                   int r, int i0) {
  const int K = G.top_k;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
  /* Four slot rows in flight at once: all four loads issue before any add, so they do not
   * serialise behind the empty-slot test. An empty slot reads row 0, which always exists, and is
   * skipped in the ascending add. */
  constexpr int kU = 4;
  for (int j0 = 0; j0 < K; j0 += kU) {
    int4 v[kU];
    int ii[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      ii[u] = j0 + u < K ? (INV ? inv[(long)row * K + j0 + u] : s_i[r * K + j0 + u]) : -1;
      v[u] = *reinterpret_cast<const int4*>(G.ye + (long)(ii[u] < 0 ? 0 : ii[u]) * G.ye_ld + i0);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (ii[u] < 0) continue;
      const float wj = INV ? bf2f(G.ew[(long)row * G.ew_ld + j0 + u]) : s_w[r * K + j0 + u];
      const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v[u]);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] += wj * bf2f(ve[e]);
    }
  }
  int4 out;
  unsigned short* oe = reinterpret_cast<unsigned short*>(&out);
  if (!G.sh) {
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = arg_bf16_rne(acc[e]);
  } else {
    float gate;
    if (INV) {
      const float v0 = bf2f(G.sg[(long)row * G.sg_ld]);
      gate = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    } else {
      gate = s_g[r];
    }
    const unsigned short* shp = G.sh + (long)row * G.sh_ld + i0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned short g = arg_bf16_rne(acc[e]);
      const unsigned short pr = arg_bf16_rne(bf2f(shp[e]) * gate);
      oe[e] = arg_bf16_rne(bf2f(pr) + bf2f(g));
    }
  }
  return out;
}

// One word of the write: h[row][c*n + i0 ..] += inj[row][c] * y[row][i0 ..] for every stream c, in
// hc_write's arithmetic (r4d_hc_bf16.hip). `yv` is the reduced word as stored bf16, which is what
// hc_write reads back from memory.
__device__ __forceinline__ void r4d_ar_hc_write_word(const unsigned short* __restrict__ inj,
                                                     unsigned short* __restrict__ h, int row,
                                                     int i0, int n, int hc, long irow, long hrow,
                                                     const unsigned short* yv) {
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = bf2f(yv[j]);
  if (hc <= 0) return;

  float g_curr = bf2f(inj[(long)row * irow]);
  int4 hv_curr = *reinterpret_cast<const int4*>(h + (long)row * hrow + i0);

  for (int c = 0; c < hc; ++c) {
    const bool more = c + 1 < hc;
    float g_next;
    int4 hv_next;
    if (more) {
      g_next = bf2f(inj[(long)row * irow + c + 1]);
      hv_next = *reinterpret_cast<const int4*>(h + (long)row * hrow + (long)(c + 1) * n + i0);
    }
    unsigned short* he = reinterpret_cast<unsigned short*>(&hv_curr);
#pragma unroll
    for (int j = 0; j < 8; ++j) he[j] = f2bf(bf2f(he[j]) + g_curr * v[j]);
    *reinterpret_cast<int4*>(h + (long)row * hrow + (long)c * n + i0) = hv_curr;
    if (more) {   /* nothing to carry past the last stream */
      g_curr = g_next;
      hv_curr = hv_next;
    }
  }
}
