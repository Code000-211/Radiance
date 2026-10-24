#pragma once
// r4d_ar_wide.h: the wide (ws>2) all-reduce family's shared transport. Every wide kernel --
// one-shot exact, two-shot exact, two-shot ti8 -- publishes in the 2*seq flag space (the
// two-shots add the 2s-1 half-step), advances seq_ctrs[b] once per AR, and lays scratch out
// as 2*ws slots of one stride (parity x source rank). The parity is the launch's, from the
// counters r4d_ar_seq.h keeps beside the handshake ones, so seq_ctrs is 2 * R4D_AR_WIDE_MAX_BLOCKS
// words. The constants that size that shared flag/seq/scratch set live here exactly once, a
// safety against drift over time across them.
//
// The 2-rank pair is deliberately NOT here: it has its own flag layout (one word per block,
// no ws stride), an acquire-on-load spin, and its own block ceiling.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "r4d_ar_seq.h"

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_WIDE_MAX_BLOCKS 64
#define R4D_AR_NB_DESIGN 16
#define R4D_AR_WIDE_MAX_PEERS 7

// Ascending peer rank throughout; rank[] maps a spin lane to its global source rank. The
// scratch word type is the one per-family difference (int4 exact wire, char ti8 wire).
template <typename S, int NPEERS>
struct R4DArWidePeers {
  S* scratch[NPEERS];
  unsigned int* flags[NPEERS];
  int rank[NPEERS];
};

__device__ __forceinline__ void r4d_ar_store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void r4d_ar_store_sys_rlx(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int r4d_ar_load_sys_rlx(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float r4d_ar_to_f(const float& x) { return x; }
__device__ __forceinline__ float r4d_ar_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float r4d_ar_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void r4d_ar_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void r4d_ar_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void r4d_ar_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// drain: 0 none, 1 all-thread threadfence_system, 2 single-thread (applied by the publisher
// inside the handshake), 3 s_wait_storecnt 0.
__device__ __forceinline__ void r4d_ar_drain(int drain) {
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
}
// acq: 0 acquire-only fence (global_inv, no writeback half), 1 full threadfence_system,
// 2 none -- fine-grained scratch only.
__device__ __forceinline__ void r4d_ar_acquire(int acq) {
  if (acq == 1) __threadfence_system();
  else if (acq != 2) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// publish `val` to every peer's flag word [b][rank], then wait until every one of this block's
// own words has reached or passed it (peers may already be a step ahead; the counter only
// grows). One spin lane per source, wrap-safe monotonic compare, never equality. Every rank
// publishes before it waits, so there is no cycle; a handshake that cannot complete traps.
template <int WS, typename S, int NPEERS>
__device__ __forceinline__ void r4d_ar_handshake(const R4DArWidePeers<S, NPEERS>& peers,
                                                 unsigned int* __restrict__ my_flags, int b,
                                                 int rank, unsigned int val, int drain, int acq,
                                                 int pub) {
  r4d_ar_drain(drain);
  __syncthreads();
  const int tid = threadIdx.x;
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (pub) {
      // One system writeback for the whole publish, then relaxed stores: the extra writebacks
      // a per-peer release store inserts add no ordering, since the flags go to different peers.
      __threadfence_system();
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rlx(&peers.flags[p][(size_t)b * WS + rank], val);
    } else {
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rel(&peers.flags[p][(size_t)b * WS + rank], val);
    }
  }
  if (tid < NPEERS) {
    const unsigned int* f = &my_flags[(size_t)b * WS + peers.rank[tid]];
    unsigned long long z = 0;
    while ((int)(r4d_ar_load_sys_rlx(f) - val) < 0) {
      __builtin_amdgcn_s_sleep(2);
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();  // lockstep violation
    }
  }
  __syncthreads();
  r4d_ar_acquire(acq);
}

// Host side. The stride and grid are enforced against the real allocations, not trusted:
// every scratch, flag, and seq pointer came from hipMalloc / hipIpcOpenMemHandle, so the
// runtime can report the extent behind it. A pointer the runtime cannot size is refused
// rather than assumed.
inline void r4d_ar_check_extent(long p, long need, const char* prefix, const char* what) {
  hipDeviceptr_t base = nullptr;
  size_t sz = 0;
  hipError_t e = hipMemGetAddressRange(&base, &sz, (hipDeviceptr_t)(uintptr_t)p);
  if (e != hipSuccess)
    throw std::runtime_error(std::string(prefix) + ": " + what +
                             " is not an allocation the runtime can size (" +
                             hipGetErrorString(e) + ")");
  const long avail = (long)(((char*)base + sz) - (char*)(uintptr_t)p);
  if (avail < need)
    throw std::runtime_error(std::string(prefix) + ": " + what + " backs " +
                             std::to_string(avail) + " bytes from the given pointer, needs " +
                             std::to_string(need));
}
