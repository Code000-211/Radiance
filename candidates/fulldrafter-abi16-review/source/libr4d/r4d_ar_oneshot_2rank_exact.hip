// r4d_ar_oneshot_2rank_exact.hip: r4d_ar_oneshot_2rank_exact -- multi-block one-shot P2P-BAR
// all-reduce, exactly two ranks, for dual R9700 (gfx1201/RDNA4, PCIe). "One-shot" is the
// algorithm class: each rank pushes its whole input, so the scratch is sized by the full message
// and there is no ring or reduce-scatter stage. "2rank" is a hard constraint, not a tuning
// choice -- the handshake is one peer flag per block, not a tree. Pure HIP behind the C ABI in
// r4d.h (wheel-ROCm ships no rocThrust, which torch/extension.h needs); the caller passes raw
// data_ptr / numel / dtype / stream.
//
// Design:
//  * PUSH model: each rank writes ITS input into the PEER's IPC scratch, then reduces
//    (local input + peer's data now in MY scratch) -> out. Only scratch+flags are
//    IPC-shared; input/output stay local -> no cudagraph buffer registration needed.
//  * MULTI-BLOCK, per-chunk independent flags (NO grid barrier): block b owns a
//    contiguous chunk of 16B words -> pushes it to peer, sets peer_flags[b], spins on
//    my_flags[b], reduces its chunk. Spreads the push across CUs (PCIe saturation) and
//    the reduce across CUs (HBM saturation): the single-block bandwidth fix.
//  * cudagraph-safe: seq is a DEVICE-resident PER-BLOCK counter the kernel increments.
//    Both ranks execute the identical graph sequence (SPMD) so seq_ctr[b] stays in
//    lockstep across ranks regardless of which batch-size graphs ran.
//  * race-free reuse: scratch DOUBLE-BUFFERED by a per-LAUNCH parity (r4d_ar_seq.h says why it
//    is not seq&1); same-stream serialization + the per-block flag handshake guarantee a slot
//    is drained before reuse.
//  * FINE-GRAINED memory (hipDeviceMallocFinegrained) => uncached => posted straight to
//    the fabric => SCOPE_SYS release/acquire skip the L2 writeback. drain/acq explicit
//    fences are tunable (all-thread system fences; default drain=1/acq=0 on fine-grained).
#include "r4d.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_hc.h"      // the gather word and the hc write word, shared with the rotated wire
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void r4d_ar_acquire(int acq) {
  if (acq) __threadfence_system();
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// vectorized 128-bit add: an int4 packs LANES elements of T; fp32-accumulate reduce.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word
// space and drives its own flag handshake; fully independent, no cross-block sync.
template <typename T, int LANES>
__global__ void r4d_ar_oneshot_2rank_exact_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq, unsigned int nbw) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // push chunk -> peer slot, 4x-unrolled for memory-level parallelism (PCIe saturation)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = in4[k];
  }
  // drain modes (ordering of the pushes vs the release flag, so the peer sees data first):
  //  0 = none (relies purely on PCIe posted-write ordering + the release flag)
  //  1 = all-thread __threadfence_system  (safe but emits an L2 wb-invalidate per
  //      wavefront (WASTED on fine-grained/uncached memory; floods at high nt*nb)
  //  2 = single-thread __threadfence_system (cheap but only drains wavefront 0's stores)
  //  3 = all-thread store-completion wait (s_wait_storecnt 0): every wavefront drains
  //      its posted writes into the fabric with NO L2 wb-invalidate; PCIe keeps posted
  //      writes to the same peer ordered, so the later release flag lands after the data.
  //      Correct for FINE-GRAINED memory (uncached; coarse would leave data in L2).
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation: reducing a slot the peer never
    // filled would hand stale bytes on as a sum, so it traps instead.
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  r4d_ar_acquire(acq);
  // reduce chunk (vectorized, 4x-unrolled): out = local input + peer input (staged in local scratch)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
}

// THE SAME ALL-REDUCE WITH THE GATED RESIDUAL'S WRITE AS ITS EPILOGUE. `y` is the message: every
// word is reduced exactly as the kernel above reduces it and stored back, and then the hc streams
// behind it take the write, h[r][c*n + i] += inj[r][c] * y[r][i], in hc_write's arithmetic
// (r4d_hc_bf16.hip) -- its bf16 helpers, its expression, and the reduced value read as the bf16
// the reduction stored, which is what hc_write reads back from memory. So the pair's output is
// reproduced to the byte with no second launch: on this part a dependent launch is a gap of about
// three microseconds, around a write that costs one or two.
//
// A word is eight elements and a row is `n` of them with n a multiple of eight, so a word never
// straddles a row and the block that reduces it owns every stream element behind it.
__global__ void r4d_ar_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = y4[k], v1 = y4[k + nt], v2 = y4[k + 2 * nt], v3 = y4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = y4[k];
  }
  // The launch's drain, applied exactly as the one-shot above applies it. `drain 3` orders the
  // pushes against the release flag by waiting on the local store counter, which is a drain of
  // posted writes and nothing more: sound on FINE-GRAINED scratch, where the peer reads the bytes
  // uncached, and not on a coarse fallback, whose bytes would still sit in an L2 the peer never
  // invalidates. The entry point therefore passes it only for a fine-grained instance.
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  r4d_ar_acquire(acq);

  const int wpr = n >> 3;                 // 16B words a row
  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
}

// THE SAME WRITE WITH THE MoE GATHER AS ITS PROLOGUE. The routed experts' rows go back to token
// order -- y[t][i] = sum over t's slots, ascending, of w[t][j] * y_expert[slot][i], rounded to
// bf16 -- with the shared arm folded as moe_gather folds it, and the word is pushed to the peer
// and kept as this rank's message exactly as if moe_gather had written it and this kernel had read
// it back. The arithmetic is moe_gather's, rounding helpers included: its narrows are
// round-to-nearest-even (r4d_common.h), where the hc write below keeps hc_write's own. So the
// three kernels' bytes are reproduced with two fewer dependent launches.
//
// A block's words sit in at most R4D_ARG_ROWS rows (the launcher sizes the grid so), and the
// slots of those rows are found once, in LDS, by one pass over the sorted list -- the scan
// moe_gather makes a token at a time.
//
// AT A PREFILL CHUNK THAT TABLE IS THE WRONG SHAPE. Eight rows a block is what makes it fit in
// LDS, and at 2048 rows it takes hundreds of blocks -- each scanning the whole sorted list, and
// each paying its own flag round trip over a fabric the all-reduce saturates with a few dozen. So
// with INV the slots come from an inverse table instead -- inv[token * top_k + slot] = the sorted
// row holding it, or -1 -- built by one launch in front (r4d_ar_gather_inv_kernel), and a block
// owns as many rows as the all-reduce's own block count gives it. The gather's reads of the
// routed rows then run beside the push of the words already summed, where the pair of kernels runs
// them one after the other. The arithmetic is the table form's, value for value.
/* inv[f] = i for every sorted row i holding slot f, and -1 for a slot no row holds: one block, so
 * the clear and the scatter are two phases with a barrier between them. */
__global__ void r4d_ar_gather_inv_kernel(int* __restrict__ inv, const int* __restrict__ sorted,
                                         int T) {
  for (int x = threadIdx.x; x < T; x += blockDim.x) inv[x] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < T; i += blockDim.x) {
    const int f = sorted[i];
    if (f >= 0 && f < T) inv[f] = i;
  }
}

template <bool INV>
__global__ void r4d_ar_gather_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, R4dArGather G,
    const int* __restrict__ inv) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
  const int K = G.top_k;

  /* The slot table. A slot no sorted row claims reads -1, so the clear and the scan are two
   * phases with a barrier between them, as in moe_gather. */
  if (!INV) {
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

  /* The gather, one word a thread a pass, pushed and kept. */
  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    const int4 out = r4d_ar_gather_word<INV>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
    y4[k] = out;
    ps[k] = out;
  }
  // The same drain as the write above, for the same reason: `drain 3` is a posted-write drain and
  // needs the peer's scratch to be uncached, which the entry point guarantees before passing it.
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  r4d_ar_acquire(acq);

  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
}

// The IPC handle is returned through a caller-owned buffer rather than a std::string so this
// stays a C ABI; R4D_AR_HANDLE_BYTES is the fixed size the header promises.
static_assert(sizeof(hipIpcMemHandle_t) <= R4D_AR_HANDLE_BYTES, "ipc handle larger than the ABI buffer");

long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  // The zeroes must be in memory before the pointer is handed to a peer, whose first push or flag
  // store would otherwise race a memset still queued behind it.
  HIPOK(hipStreamSynchronize(nullptr));
  hipIpcMemHandle_t h;
  HIPOK(hipIpcGetMemHandle(&h, p));
  std::memcpy(out_handle, &h, sizeof(h));
  *out_len = (int)sizeof(h);
  return (long)p;
}
long r4d_ar_ipc_open(const char* handle, int len) {
  hipIpcMemHandle_t h;
  if (len != (int)sizeof(h)) throw std::runtime_error("bad ipc handle size");
  std::memcpy(&h, handle, sizeof(h));
  void* p = nullptr;
  HIPOK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return (long)p;
}
// A strided device-to-device copy, which is what an ALL-GATHER between two ranks is once the
// peer's buffer is IPC-mapped: no reduction, no codec, nothing to schedule. The runtime's copy
// path is the fast one for peer traffic on this part -- 13.52 GB/s per direction with both
// directions running, against 9.30 for a kernel doing b128 peer loads and 10.1 for RCCL.
// `dpitch`, `spitch` and `width` are BYTES; `height` is rows.
// THE SAME COPY AS A KERNEL, WITH A BOUNDED WORKGROUP COUNT.
//
// `hipMemcpy2DAsync` is not a DMA engine on this part -- it runs as
// `__amd_rocclr_copyBufferRectAligned` -- and it launches ONE 16-thread workgroup per 16 bytes of
// a row: a 64 x 248 KB rectangle is 62080 workgroups.  Alone that is fine and reaches the link
// rate.  Sharing the part with a GEMM it is meant to hide under, it fills every scheduling slot,
// costs that GEMM around 60% of its runtime again, and gains almost nothing itself.
//
// A peer copy needs a very small grid: across access width x workgroup size x workgroup count x
// loads-in-flight (swept by `dev/prrun.py`), 32 workgroups of 64 threads is within 0.1% of the
// best point at every size.  So this takes the grid as an argument and defaults to 64 workgroups,
// which is a 0.1% footprint against the blit's 62080.
//
// 16-byte accesses, so every pitch, the width and both base addresses must be 16-byte aligned; the
// caller checks and falls back to `r4d_p2p_copy2d` when they are not.
typedef unsigned r4d_u4v __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(256) void r4d_p2p_copy2d_k(const r4d_u4v* __restrict__ src,
                                                        r4d_u4v* __restrict__ dst, long spitch16,
                                                        long dpitch16, long wpr, int height) {
  const long cstride = (long)gridDim.x * blockDim.x;
  for (int r = blockIdx.y; r < height; r += gridDim.y) {
    const r4d_u4v* s = src + (long)r * spitch16;
    r4d_u4v* d = dst + (long)r * dpitch16;
    for (long c = (long)blockIdx.x * blockDim.x + threadIdx.x; c < wpr; c += cstride) d[c] = s[c];
  }
}

int r4d_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream) {
  if ((dst | src | dpitch | spitch | width) & 15L) return -1;
  if (height <= 0 || width <= 0) return 0;
  if (threads <= 0 || threads > 256 || (threads & 31)) return -2;
  const long wpr = width / 16;
  if (bx <= 0) bx = (int)((wpr + threads - 1) / threads);
  if (bx > 64) bx = 64;
  if (by <= 0) by = (int)height;
  if (by > (int)height) by = (int)height;
  hipLaunchKernelGGL(r4d_p2p_copy2d_k, dim3(bx, by), dim3(threads), 0, (hipStream_t)stream,
                     (const r4d_u4v*)src, (r4d_u4v*)dst, spitch / 16, dpitch / 16, wpr,
                     (int)height);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A SYSTEM-SCOPE RELEASE on `stream`, for the case where this device has written into the PEER's
// memory and something other than this device is about to read it.  A peer write issued by a copy
// or a kernel is a posted write: it retires from the wave long before it lands, and the only thing
// that orders it against a later handshake on a different path is a system-scope release.  One
// thread is enough -- the fence is a memory-system operation, not a per-lane one -- and it costs a
// dispatch.
__global__ void r4d_p2p_fence_kernel() { __threadfence_system(); }

int r4d_p2p_fence(long stream) {
  hipLaunchKernelGGL(r4d_p2p_fence_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream) {
  hipError_t e = hipMemcpy2DAsync((void*)dst, (size_t)dpitch, (const void*)src, (size_t)spitch,
                                  (size_t)width, (size_t)height, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
  return e == hipSuccess ? 0 : -100 - (int)e;
}

void r4d_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream_i, long nblocks, long nthreads,
                     long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8) throw std::runtime_error("ar_hc_write: n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8) throw std::runtime_error("ar_hc_write: the streams must be whole 16B words");
  const int n16 = (int)(M * n * 2 / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  r4d_ar_hc_write_kernel<<<nb, nt, 0, st>>>(
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h,
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw);
}

int r4d_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                           long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                           long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                           long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                           long top_k, long sig, long stream_i, long nblocks, long nthreads,
                           long drain, long acq, long inv) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8 || hc < 1 || hrow % 8) return -1;
  if (top_k < 1 || top_k > R4D_ARG_SLOTS) return -2;
  if (ye_ld % 8 || (sh && sh_ld < n)) return -3;
  const int n16 = (int)(M * n * 2 / 16);
  const int wpr = (int)(n / 8);
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  /* THE GRID IS AT LEAST WIDE ENOUGH THAT NO BLOCK SPANS MORE THAN R4D_ARG_ROWS ROWS: a block of
   * w words starting anywhere touches at most (w - 1) / wpr + 2 rows. Both ranks compute the same
   * count from the same M and n, which the per-block handshake requires. */
  const int wmax = (R4D_ARG_ROWS - 2) * wpr + 1;
  const int nb_min = (n16 + wmax - 1) / wmax;
  /* THE INVERSE TABLE WHEN THE LDS ONE WOULD NEED MORE BLOCKS THAN THE REDUCTION WANTS, which is
   * a prefill chunk and never a decode step. Both ranks decide it from the same M and n. The
   * block count stays the reduction's: at a 2048-row chunk 48 and 96 blocks measure within half
   * a percent of its 24, and 192 no better. */
  const bool by_inv = inv != 0 && nb_min > nb;
  if (!by_inv && nb < nb_min) nb = nb_min;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS || (unsigned)nb > nbw) return -4;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  /* Zeroed whole, padding included: the struct is passed by value, so its tail padding is
   * kernel-argument bytes too, and a recorded pass's audit compares those bytes. */
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  if (by_inv) {
    r4d_ar_gather_inv_kernel<<<1, 1024, 0, st>>>((int*)inv, (const int*)sorted, G.T);
    r4d_ar_gather_hc_write_kernel<true><<<nb, nt, 0, st>>>(
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h,
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,
        (const int*)inv);
  } else {
    r4d_ar_gather_hc_write_kernel<false><<<nb, nt, 0, st>>>(
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h,
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,
        nullptr);
  }
  return (int)hipGetLastError();
}

/* The inverse slot table alone, on `stream`, for the wide fused gather (r4d_ar_oneshot_Nrank_exact.hip):
 * the same kernel and the same launch as above, so both widths read one table. */
int r4d_ar_gather_inv(long inv, long sorted, long T, long stream_i) {
  if (!inv || !sorted || T < 1) return -1;
  r4d_ar_gather_inv_kernel<<<1, 1024, 0, (hipStream_t)stream_i>>>((int*)inv, (const int*)sorted,
                                                                  (int)T);
  return (int)hipGetLastError();
}

void r4d_ar_ipc_free(long p) { if (p) (void)hipFree((void*)p); }
void r4d_ar_ipc_memzero(long p, long size) { if (p) HIPOK(hipMemset((void*)p, 0, size)); }

// Enable direct peer access. radiance runs one process with a thread per rank sharing an address
// space, so a peer pointer is valid here once this has run and no IPC handle exchange is needed;
// see the header of r4d_ar_entry.hip.
void r4d_ar_ipc_enable_peer(long peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags,
                            long my_flags, long seq_ctrs, long slot_stride16,
                            long inp, long out, long n_elem, long dtype,
                            long stream_i, long nblocks, long nthreads,
                            long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  // The counter allocation's block capacity, which every 2-rank kernel on an instance must agree
  // on because the parity counters sit that many words in (r4d_ar_seq.h).
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
#define L(T, LN) r4d_ar_oneshot_2rank_exact_kernel<T, LN><<<nb, nt, 0, st>>>(              \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,        \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,              \
    (int)slot_stride16, (int)drain, (int)acq, nbw)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
}

