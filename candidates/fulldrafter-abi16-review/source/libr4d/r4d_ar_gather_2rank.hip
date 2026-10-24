// r4d_ar_gather_2rank.hip: r4d_ar_gather_2rank -- multi-block one-shot P2P all-GATHER, exactly two
// ranks, for dual R9700 (gfx1201/RDNA4, PCIe).
//
// WHY IT EXISTS AS A KERNEL AND NOT AS A COPY. r4d_ar_oneshot_2rank_exact.hip says of
// r4d_p2p_copy2d that a strided device-to-device copy "is what an ALL-GATHER between two ranks is
// once the peer's buffer is IPC-mapped", and that is true of the DATA MOVEMENT and false of the
// whole operation: a copy carries no handshake, so the reader has no way to know the writer's
// bytes have landed. Ordering that with a separate fence plus a barrier collective is three
// launches and a second rendezvous to keep aligned with this one. This is the same kernel as the
// one-shot all-reduce with the reduction removed -- same push, same per-block flag protocol, same
// double-buffered slot -- and it is a separate file rather than a template variant because the
// all-reduce's shape (out is numel) and this one's (out is world_size * numel) differ.
//
// WHAT IT IS FOR. The vocab-parallel sampler. lm_head is row-sharded, so each rank reduces its own
// slice of the vocabulary to k candidates and only the CANDIDATE SETS are combined: at 4 rows and
// 1024 candidates the message is 32 KiB, against the ~1 GiB a full-logits gather would move. The
// payload is (u32 global token id, f32 logit) pairs, which is why this MOVES bytes and does not
// sum them -- adding two f32 lanes would work on the logit and destroy the id, whose bit pattern
// at any real vocabulary size is an f32 denormal.
//
// The element type is therefore irrelevant beyond its size: `dtype` is taken for symmetry with the
// rest of the family and used only to turn n_elem into bytes.
#include "r4d.h"
#include "r4d_ar_seq.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#define R4D_AR_GATHER_SPIN_MAX 4000000000ULL
#define R4D_AR_GATHER_MAX_BLOCKS 512

__device__ __forceinline__ void r4d_g_store_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int r4d_g_load_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// WHERE A RANK'S WORDS LAND. `row16` is one ROW of this rank's contribution, in 16-byte words.
//
//   row16 <= 0 or row16 == n16  -- one row: rank r's whole slice at r * n16, the plain
//                                 concatenation an all-gather is usually taken to mean.
//   otherwise                   -- ROW-INTERLEAVED: the input is [rows][row16] and the output is
//                                 [rows][world][row16], so row m carries every rank's words for
//                                 that row side by side.
//
// The second form is the one the vocab-parallel sampler needs and it is not a convenience. The
// merge that consumes this reads ONE ROW of gathered candidates per sampled position, at a single
// row stride; the concatenated form puts rank 1's row m a whole message away from rank 0's, which
// no row stride can express. Interleaving here costs one division per word in a kernel that moves
// 32 KiB and saves a transpose launch on the step path.
__device__ __forceinline__ int r4d_g_dst(int k, int row16, int n16, int who) {
  if (row16 <= 0 || row16 >= n16) return who * n16 + k;
  const int m = k / row16, c = k - m * row16;
  return m * (2 * row16) + who * row16 + c;
}

// A tail makes rank 1's output start only 4-byte aligned. Keep vector loads and
// scratch stores, but use scalar output stores rather than misaligned int4 stores.
template<bool HasTail>
__device__ __forceinline__ void r4d_g_put(int4* out4, int k, int row16, int n16,
                                         int tail32, int who, int4 v) {
  if (HasTail) {
    uint32_t* dst = (uint32_t*)out4 + (size_t)who * (n16 * 4 + tail32) + k * 4;
    dst[0] = (uint32_t)v.x; dst[1] = (uint32_t)v.y;
    dst[2] = (uint32_t)v.z; dst[3] = (uint32_t)v.w;
  } else {
    out4[r4d_g_dst(k, row16, n16, who)] = v;
  }
}

// Block b owns words [start,end) of the 16B-word space and drives its own flag handshake; no
// cross-block sync, exactly as in the all-reduce.
template<bool HasTail>
__global__ void r4d_ar_gather_2rank_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int tail32, int slot_stride16, int rank, int row16, int drain, int acq, unsigned int nbw) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // push chunk -> peer slot, and place this rank's own slice while the push is in flight: it is a
  // local store that needs no handshake, so it costs nothing to hide here.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      r4d_g_put<HasTail>(out4, k,          row16, n16, tail32, rank, v0);
      r4d_g_put<HasTail>(out4, k + nt,     row16, n16, tail32, rank, v1);
      r4d_g_put<HasTail>(out4, k + 2 * nt, row16, n16, tail32, rank, v2);
      r4d_g_put<HasTail>(out4, k + 3 * nt, row16, n16, tail32, rank, v3);
    }
    for (; k < end; k += nt) {
      int4 v = in4[k]; ps[k] = v;
      r4d_g_put<HasTail>(out4, k, row16, n16, tail32, rank, v);
    }
  }
  // Block zero owns the 0..3 words after the last vector and its handshake.
  if (HasTail && b == 0 && tid < tail32) {
    const int k = n16 * 4 + tid;
    const uint32_t v = ((const uint32_t*)in4)[k];
    ((uint32_t*)ps)[k] = v;
    ((uint32_t*)out4)[(size_t)rank * (n16 * 4 + tail32) + k] = v;
  }
  // Drain modes, identical to the all-reduce's: they order the pushes against the release flag so
  // the peer cannot see the flag before the data.
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    r4d_g_store_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation, and placing a slot the peer never
    // filled would pass stale bytes on as the peer's, so it traps instead.
    while (r4d_ar_seq_behind(r4d_g_load_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_GATHER_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();

  // The peer's slice, now staged in this rank's scratch, into the places that name it.
  {
    const int who = 1 - rank;
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      r4d_g_put<HasTail>(out4, k,          row16, n16, tail32, who, ms[k]);
      r4d_g_put<HasTail>(out4, k + nt,     row16, n16, tail32, who, ms[k + nt]);
      r4d_g_put<HasTail>(out4, k + 2 * nt, row16, n16, tail32, who, ms[k + 2 * nt]);
      r4d_g_put<HasTail>(out4, k + 3 * nt, row16, n16, tail32, who, ms[k + 3 * nt]);
    }
    for (; k < end; k += nt)
      r4d_g_put<HasTail>(out4, k, row16, n16, tail32, who, ms[k]);
  }
  if (HasTail && b == 0 && tid < tail32) {
    const int k = n16 * 4 + tid;
    ((uint32_t*)out4)[(size_t)(1 - rank) * (n16 * 4 + tail32) + k] =
        ((const uint32_t*)ms)[k];
  }
}

void r4d_ar_gather_2rank(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                         long dtype, long rank, long row_elems, long stream_i, long nblocks,
                         long nthreads, long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 4 != 0) throw std::runtime_error("all_gather: n_elem*esize not 4B-aligned");
  if (rank != 0 && rank != 1) throw std::runtime_error("all_gather: rank must be 0 or 1");
  const int n16 = (int)(nbytes / 16);
  const int tail32 = (int)((nbytes % 16) / 4);
  // One row of this rank's contribution, in 16-byte words. Zero means "one row", which is the
  // plain concatenation; anything else has to divide evenly or a row would straddle a word.
  long rowb = (long)row_elems * esize;
  if (row_elems > 0 && (rowb % 16 != 0 || nbytes % rowb != 0))
    throw std::runtime_error("all_gather: row is not a whole number of 16B words");
  const int row16 = row_elems > 0 ? (int)(rowb / 16) : 0;
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16 > 0 ? n16 : 1;
  if (nb > R4D_AR_GATHER_MAX_BLOCKS) nb = R4D_AR_GATHER_MAX_BLOCKS;
  // The counter allocation's block capacity, shared with the all-reduce this rides beside
  // (r4d_ar_seq.h).
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  if (tail32) {
    r4d_ar_gather_2rank_kernel<true><<<nb, nt, 0, st>>>(
        (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,
        (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16, tail32,
        (int)slot_stride16, (int)rank, row16, (int)drain, (int)acq, nbw);
  } else {
    r4d_ar_gather_2rank_kernel<false><<<nb, nt, 0, st>>>(
        (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,
        (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16, 0,
        (int)slot_stride16, (int)rank, row16, (int)drain, (int)acq, nbw);
  }
}
