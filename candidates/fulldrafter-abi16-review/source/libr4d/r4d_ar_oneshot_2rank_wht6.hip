// r4d_ar_oneshot_2rank_wht6.hip: translation unit for r4d_ar_oneshot_2rank_wht6, the all-reduce whose wire
// payload is Walsh-Hadamard rotated and quantised to 6 bits. The device code and the launcher live
// in r4d_ar_oneshot_2rank_wht6.h so the same source also compiles standalone (no Python, no r4d); this file gives
// them the library's C ABI and reports the packing constants the caller needs to size its scratch.
// It also holds the rotated wire's fused forms of ar_hc_write and ar_gather_hc_write, which need
// the library's gather and hc-write arithmetic (r4d_ar_hc.h) and so live here, not in the header.
#include "r4d.h"
#include "r4d_ar_hc.h"   // the gather word and the hc write word, shared with the exact wire
#include "r4d_ar_oneshot_2rank_wht6.h"

void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq) {
  r4d_ar_wht6_launch(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs, loc_pack,
                     slot_stride_bytes, scale_off_bytes, inp, out, n_elem, dtype, stream, nblocks,
                     nthreads, drain, acq);
}

// Block cap of both all-reduce kernels (the flag array is sized from it).
int r4d_ar_max_blocks(void) { return R4D_AR_MAX_BLOCKS; }

// The rotated-6-bit wire format, which the caller needs to size its packed scratch.
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems) {
  if (group)       *group       = GROUP;
  if (bits)        *bits        = BITS;
  if (chunk_elems) *chunk_elems = CHUNK_ELEMS;
}

// ---------------------------------------------------------------------------------------------
// THE ROTATED WIRE WITH THE GATED RESIDUAL'S WRITE AS ITS EPILOGUE, and with GATHER the MoE gather
// as its prologue: the 6-bit twins of r4d_ar_hc_write_kernel and r4d_ar_gather_hc_write_kernel
// (r4d_ar_oneshot_2rank_exact.hip). Each reproduces the unfused chain it replaces to the byte --
// moe_gather, r4d_ar_oneshot_2rank_wht6, hc_write -- because every stage is the same code: the
// gather word and the write word are r4d_ar_hc.h's, and the wire is wht6_push_chunk and
// wht6_reduce_chunk (r4d_ar_oneshot_2rank_wht6.h) at a smaller chunk, which moves bytes between
// owners and never changes them. The message's words are the chain's words: the gather's are rounded to
// bf16 as moe_gather stores them, and the reduced sum is rounded to bf16 as the standalone
// all-reduce stores it, stored into `y`, and read back as that bf16 by the write.
//
// A lane's EPL elements are one word, so a word -- and the write behind it -- never straddles a row
// even where a group or a chunk does: the rotation closes over the flattened message, the write
// over the row the word sits in.
//
// THE GATHER READS ITS SLOTS FROM A TABLE IN LDS, built by the block from the sorted list: the
// rows a block owns (whole wave-chunks, so tens of rows at a prefill chunk) and every slot of them,
// found by one pass over the list -- the exact form's decode-sized table, sized here by the launch
// rather than fixed at R4D_ARG_ROWS. A word's reads of the routed rows then wait on nothing but
// their own addresses, where the inverse table puts a dependent load of the slot in front of each.
// The launcher widens the grid until a block's table fits.
static_assert(EPL == 8, "a lane's share of a group is one 16B word of the message");

// The LDS a block's slot table may take; the handshake's two words sit beside it.
static constexpr size_t kWht6TableBytes = 60 * 1024;

// THE FUSED FORMS' CHUNK IS 8 GROUPS -- 512 elements, two group-iterations a wave -- where the
// standalone all-reduce's is 32. A wave walks its chunk's groups one after another and each one
// waits on its own reads -- the routed rows of the gather, the streams of the write -- so a
// quarter of the chunk is four times the waves at work on the same message: at a 2048-row chunk
// the gather form takes 403 us a call, against 500 at the standalone chunk and 416 at 16 groups.
// The two ranks run the same kernel, so the chunking both push and reduce under is the same.
static constexpr int kWht6FusedCG = 8;

// The rows block b's chunks of `ce` elements touch: [row_lo, row_lo + nrow).
__device__ __forceinline__ void wht6_block_rows(int c0, int c1, int ce, int n_groups, int n,
                                                int& row_lo, int& nrow) {
  const long e0 = (long)c0 * ce;
  long e1 = (long)c1 * ce;
  if (e1 > (long)n_groups * GROUP) e1 = (long)n_groups * GROUP;
  row_lo = (int)(e0 / n);
  nrow = e1 > e0 ? (int)((e1 - 1) / n) - row_lo + 1 : 0;
}

template <bool GATHER, int CG>
__device__ __forceinline__ void r4d_ar_hc_write_wht6_impl(
    unsigned short* __restrict__ y, char* peer_scratch, const char* my_scratch, char* loc_pack,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n, int hc,
    long irow, long hrow, int n_groups, int n_chunks, long slot_stride_bytes,
    long scale_off_bytes, int drain, int acq, unsigned int nbw, R4dArGather G, int rows_max) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid & (WAVE - 1), wid = tid / WAVE, nwarp = nt / WAVE;
  const int lig = lane & (LPG - 1), gw = lane / LPG;
  __shared__ unsigned int s_seq, s_slot;
  extern __shared__ int s_tab[];   // slots, weights: [rows_max * top_k]; gates: [rows_max]
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe (cudagraph)
  const unsigned int s = s_seq;
  const Wht6Slot w = wht6_slot(peer_scratch, my_scratch, loc_pack, (int)s_slot,
                               slot_stride_bytes, scale_off_bytes);

  const int cpb = (n_chunks + nb - 1) / nb;                 // block b owns chunks [c0,c1)
  const int c0 = b * cpb; int c1 = c0 + cpb; if (c1 > n_chunks) c1 = n_chunks;

  /* The slot table, as the exact form's: a slot no sorted row claims reads -1, so the clear and
   * the scan are two phases with a barrier between them. */
  int* s_i = s_tab;
  float* s_w = reinterpret_cast<float*>(s_tab + rows_max * G.top_k);
  float* s_g = s_w + rows_max * G.top_k;
  int row_lo = 0, nrow = 0;
  if (GATHER) {
    wht6_block_rows(c0, c1, Wht6Chunk<CG>::ELEMS, n_groups, n, row_lo, nrow);
    const int K = G.top_k;
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }

  const auto ld = [&](long off, float* v) {
    int4 q;
    if (GATHER) {
      const int row = (int)(off / n), i0 = (int)(off - (long)row * n);
      q = r4d_ar_gather_word<false>(G, nullptr, s_i, s_w, s_g, row, row - row_lo, i0);
    } else {
      q = *reinterpret_cast<const int4*>(y + off);
    }
    const unsigned short* e = reinterpret_cast<const unsigned short*>(&q);
#pragma unroll
    for (int i = 0; i < EPL; ++i) v[i] = bf2f(e[i]);
  };
  const auto st = [&](long off, const float* v) {
    int4 q;
    __hip_bfloat16* e = reinterpret_cast<__hip_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < EPL; ++i) rd_from_f(v[i], e[i]);
    *reinterpret_cast<int4*>(y + off) = q;
    const int row = (int)(off / n), i0 = (int)(off - (long)row * n);
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&q));
  };
  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_push_chunk<CG>(ld, c, n_groups, lane, lig, gw, w.p_pay, w.p_sc, w.l_pay, w.l_sc);

  wht6_handshake(peer_flags, my_flags, b, s, tid, drain, acq);

  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_reduce_chunk<CG>(st, c, n_groups, lane, lig, gw, w.m_pay, w.m_sc, w.l_pay, w.l_sc);
}

template <int CG>
__global__ __attribute__((amdgpu_num_vgpr(64))) void r4d_ar_hc_write_wht6_kernel(
    unsigned short* __restrict__ y, char* peer_scratch, const char* my_scratch, char* loc_pack,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n, int hc,
    long irow, long hrow, int n_groups, int n_chunks, long slot_stride_bytes,
    long scale_off_bytes, int drain, int acq, unsigned int nbw, R4dArGather G, int rows_max) {
  r4d_ar_hc_write_wht6_impl<false, CG>(y, peer_scratch, my_scratch, loc_pack, peer_flags, my_flags,
                                       seq_ctrs, inj, h, n, hc, irow, hrow, n_groups, n_chunks,
                                       slot_stride_bytes, scale_off_bytes, drain, acq, nbw, G, rows_max);
}

template <int CG>
__global__ void r4d_ar_gather_hc_write_wht6_kernel(
    unsigned short* __restrict__ y, char* peer_scratch, const char* my_scratch, char* loc_pack,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n, int hc,
    long irow, long hrow, int n_groups, int n_chunks, long slot_stride_bytes,
    long scale_off_bytes, int drain, int acq, unsigned int nbw, R4dArGather G, int rows_max) {
  r4d_ar_hc_write_wht6_impl<true, CG>(y, peer_scratch, my_scratch, loc_pack, peer_flags, my_flags,
                                      seq_ctrs, inj, h, n, hc, irow, hrow, n_groups, n_chunks,
                                      slot_stride_bytes, scale_off_bytes, drain, acq, nbw, G, rows_max);
}

int r4d_ar_hc_write_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long loc_pack, long slot_stride_bytes,
                         long scale_off_bytes, long y, long inj, long h, long M, long n, long hc,
                         long irow, long hrow, long ye, long ew, long sorted, long sh, long sg,
                         long ye_ld, long ew_ld, long sh_ld, long sg_ld, long top_k, long sig,
                         long stream_i, long nblocks, long nthreads, long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8 || hc < 1 || hrow % 8) return -1;
  if ((M * n) % GROUP) return -1;
  const bool gather = ye != 0;
  if (gather && (!ew || !sorted || top_k < 1 || top_k > R4D_ARG_SLOTS || ye_ld % 8 ||
                 (sh && sh_ld < n)))
    return -2;
  const int n_groups = (int)(M * n / GROUP);
  const int ce = Wht6Chunk<kWht6FusedCG>::ELEMS;
  const int n_chunks = (n_groups + kWht6FusedCG - 1) / kWht6FusedCG;
  int nb = clamp_blocks((int)nblocks, n_chunks);
  const int nt = clamp_threads((int)nthreads);
  /* THE TABLE HAS TO FIT, so the grid widens until it does: a block of cpb chunks touches at most
   * (cpb * ce - 1) / n + 2 rows. Both ranks compute the same grid from the same M and n, which the
   * per-block handshake requires. */
  int rows_max = 0;
  size_t lds = 0;
  if (gather) {
    for (;;) {
      const long cpb = (n_chunks + nb - 1) / nb;
      rows_max = (int)((cpb * ce - 1) / n + 2);
      lds = (size_t)rows_max * (size_t)top_k * 8 + (size_t)rows_max * 4;
      if (lds <= kWht6TableBytes || nb >= n_chunks) break;
      ++nb;
    }
    if (lds > kWht6TableBytes || nb > R4D_AR_MAX_BLOCKS) return -3;
  }
  R4dArGather G{};
  if (gather) {
    G.ye = (const unsigned short*)ye;
    G.ew = (const unsigned short*)ew;
    G.sorted = (const int*)sorted;
    G.sh = (const unsigned short*)sh;
    G.sg = (const unsigned short*)sg;
    G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
    G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  }
  if (gather) {
    r4d_ar_gather_hc_write_wht6_kernel<kWht6FusedCG><<<nb, nt, lds, st>>>(
        (unsigned short*)y, (char*)peer_scratch, (const char*)my_scratch, (char*)loc_pack,
        (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj,
        (unsigned short*)h, (int)n, (int)hc, irow, hrow, n_groups, n_chunks, slot_stride_bytes,
        scale_off_bytes, (int)drain, (int)acq, (unsigned)R4D_AR_MAX_BLOCKS, G, rows_max);
  } else {
    r4d_ar_hc_write_wht6_kernel<kWht6FusedCG><<<nb, nt, lds, st>>>(
        (unsigned short*)y, (char*)peer_scratch, (const char*)my_scratch, (char*)loc_pack,
        (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj,
        (unsigned short*)h, (int)n, (int)hc, irow, hrow, n_groups, n_chunks, slot_stride_bytes,
        scale_off_bytes, (int)drain, (int)acq, (unsigned)R4D_AR_MAX_BLOCKS, G, rows_max);
  }
  return (int)hipGetLastError();
}
