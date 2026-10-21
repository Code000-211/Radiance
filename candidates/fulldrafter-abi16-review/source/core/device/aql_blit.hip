/* aql_blit.hip -- the copy and fill kernels the AQL backend dispatches for rad_memcpy_async,
 * rad_memcpy_2d_async and rad_memset_async.
 *
 * A copy on this backend is a kernel on the stream's own queue, the same as HIP's blit path: the
 * alternative is the DMA engine, which is a second queue to order against with signals and costs
 * more in latency than a small copy takes to run. Every address these kernels see is one the GPU
 * can reach directly -- device memory, a peer's, or pinned host memory; pageable host memory is
 * staged by the caller.
 *
 * The launchers go through hipLaunchKernel like any plugin's, and the backend intercepts that, so
 * these kernels are dispatched exactly as a kernel library's are.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "aql.h"
#include "device.h"
#include "../../arch/common/rad_kv_compact.h"

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaxBlocks = 2048;

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

/* Four loads in flight a thread before the first store: a copy that reads pinned host memory is
 * link-latency bound, and one outstanding load a thread leaves most of the link idle. */
__global__ __launch_bounds__(kThreads) void aql_copy16(const u32x4* __restrict__ src,
                                                       u32x4* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const u32x4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = src[i];
}

/* A COPY THAT READS OR WRITES ACROSS THE LINK IS ONE WORKGROUP OF FOUR WAVES.
 *
 * The link delivers ~29 GB/s whatever the grid, so the only question is how few waves reach that,
 * because every wave a copy holds is one the compute queue's kernels do not get while the copy
 * crawls along at link speed. A wide link copy -- 16 workgroups of 256 threads, 128 waves -- costs
 * a prefill whose stager keeps a copy running beside every layer ~15%: hc_read at a 2048-row chunk
 * takes 1154 us a call with such copies running and 609 without, because its down product is
 * exactly one wave of workgroups on the part and any slot a copy holds pushes some of it into a
 * second. Link bandwidth by grid, one card, 89 MB from pinned host memory:
 *
 *     16 x 256 threads, 4 loads in flight    28.9 GB/s
 *      1 x 256 threads, 16 in flight         28.9
 *      1 x 128 threads, 16 in flight         28.9           <- this
 *      1 x  64 threads, 16 in flight         16.8
 *      1 x  64 threads,  8 in flight         10.4
 *
 * 32 KiB in flight is several times the link's bandwidth-delay product; fewer threads leave the
 * link idle, and 32 loads a thread spill. Non-temporal loads and stores change nothing: it is the
 * slots, not the caches.
 *
 * The last round loads clamped rather than guarded, so its loads are all in flight at once: a
 * guarded load waits for the one before it. */
constexpr int kLinkThreads = 128, kLinkDepth = 16;
constexpr uint64_t kLinkByteBlocks = 16;

__global__ __launch_bounds__(kLinkThreads) void aql_copy_link(const u32x4* __restrict__ src,
                                                              u32x4* __restrict__ dst, uint64_t n) {
    uint64_t i = threadIdx.x;
    for (; i + (uint64_t)(kLinkDepth - 1) * kLinkThreads < n; i += (uint64_t)kLinkDepth * kLinkThreads) {
        u32x4 v[kLinkDepth];
#pragma unroll
        for (int k = 0; k < kLinkDepth; ++k) v[k] = src[i + (uint64_t)k * kLinkThreads];
#pragma unroll
        for (int k = 0; k < kLinkDepth; ++k) dst[i + (uint64_t)k * kLinkThreads] = v[k];
    }
    if (i >= n) return;
    u32x4 v[kLinkDepth];
#pragma unroll
    for (int k = 0; k < kLinkDepth; ++k) {
        const uint64_t j = i + (uint64_t)k * kLinkThreads;
        v[k] = src[j < n ? j : n - 1];
    }
#pragma unroll
    for (int k = 0; k < kLinkDepth; ++k) {
        const uint64_t j = i + (uint64_t)k * kLinkThreads;
        if (j < n) dst[j] = v[k];
    }
}

__global__ __launch_bounds__(kThreads) void aql_copy1(const unsigned char* __restrict__ src,
                                                      unsigned char* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

__global__ __launch_bounds__(kThreads) void aql_copy2d16(const u32x4* __restrict__ src,
                                                         u32x4* __restrict__ dst, uint64_t spitch16,
                                                         uint64_t dpitch16, uint64_t w16,
                                                         uint64_t height) {
    const uint64_t cstride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t r = blockIdx.y; r < height; r += gridDim.y) {
        const u32x4* s = src + r * spitch16;
        u32x4* d = dst + r * dpitch16;
        for (uint64_t c = (uint64_t)blockIdx.x * kThreads + threadIdx.x; c < w16; c += cstride)
            d[c] = s[c];
    }
}

__global__ __launch_bounds__(kThreads) void aql_copy2d1(const unsigned char* __restrict__ src,
                                                        unsigned char* __restrict__ dst,
                                                        uint64_t spitch, uint64_t dpitch,
                                                        uint64_t width, uint64_t height) {
    const uint64_t cstride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t r = blockIdx.y; r < height; r += gridDim.y) {
        const unsigned char* s = src + r * spitch;
        unsigned char* d = dst + r * dpitch;
        for (uint64_t c = (uint64_t)blockIdx.x * kThreads + threadIdx.x; c < width; c += cstride)
            d[c] = s[c];
    }
}

/* A RANGED COPY: the first `prefix_blocks` workgroups move the prefix as aql_copy16 does, and each
 * workgroup after them one range. A range is four-byte aligned only, so it moves in words, sixteen
 * a thread a round: every load of a round is issued before its first store, since each is a read
 * across the link, and the loads past the range's end read its last word rather than branch
 * around the load -- a guarded load waits for the one before it. */
constexpr int kRangeWords = 16;

__global__ __launch_bounds__(kThreads) void aql_copy_ranges(const u32x4* __restrict__ src16,
                                                            u32x4* __restrict__ dst16, uint64_t n16,
                                                            unsigned prefix_blocks,
                                                            const RadCopyRange* __restrict__ r) {
    if (blockIdx.x < prefix_blocks) {
        const uint64_t stride = (uint64_t)prefix_blocks * kThreads;
        uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
        for (; i + 3 * stride < n16; i += 4 * stride) {
            const u32x4 a = src16[i], b = src16[i + stride], c = src16[i + 2 * stride],
                        d = src16[i + 3 * stride];
            dst16[i] = a; dst16[i + stride] = b; dst16[i + 2 * stride] = c; dst16[i + 3 * stride] = d;
        }
        for (; i < n16; i += stride) dst16[i] = src16[i];
        return;
    }
    const RadCopyRange g = r[blockIdx.x - prefix_blocks];
    const unsigned* s4 = (const unsigned*)((const char*)src16 + g.src);
    unsigned* d4 = (unsigned*)((char*)dst16 + g.dst);
    const int64_t n = g.bytes / 4;
    for (int64_t base = 0; base < n; base += (int64_t)kThreads * kRangeWords) {
        unsigned v[kRangeWords];
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            v[k] = s4[w < n ? w : n - 1];
        }
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            if (w < n) d4[w] = v[k];
        }
    }
}

__global__ __launch_bounds__(kThreads) void aql_fill16(u32x4* __restrict__ dst, unsigned v,
                                                       uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    const u32x4 w = { v, v, v, v };
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = w;
}

__global__ __launch_bounds__(kThreads) void aql_fill1(unsigned char* __restrict__ dst,
                                                      unsigned char v, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = v;
}

inline unsigned blocks_for(uint64_t n, uint64_t per_block, uint64_t cap = kMaxBlocks) {
    uint64_t b = (n + per_block - 1) / per_block;
    if (b < 1) b = 1;
    if (b > cap) b = cap;
    return (unsigned)b;
}

inline bool aligned16(uint64_t v) { return (v & 15u) == 0; }

inline int launched() { return hipGetLastError() == hipSuccess ? 0 : -1; }

}  // namespace

extern "C" int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream,
                                 bool link) {
    if (bytes == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    if (aligned16((uint64_t)(uintptr_t)dst | (uint64_t)(uintptr_t)src | bytes)) {
        const uint64_t n = bytes / 16;
        if (link)
            hipLaunchKernelGGL(aql_copy_link, dim3(1), dim3(kLinkThreads), 0, s,
                               (const u32x4*)src, (u32x4*)dst, n);
        else
            hipLaunchKernelGGL(aql_copy16, dim3(blocks_for(n, 4 * kThreads)), dim3(kThreads), 0,
                               s, (const u32x4*)src, (u32x4*)dst, n);
    } else {
        /* Unaligned: byte copies, one byte a thread in flight, so across the link they take
         * sixteen workgroups rather than the one narrow workgroup of the aligned path. */
        const uint64_t cap = link ? kLinkByteBlocks : kMaxBlocks;
        hipLaunchKernelGGL(aql_copy1, dim3(blocks_for(bytes, 4 * kThreads, cap)), dim3(kThreads),
                           0, s, (const unsigned char*)src, (unsigned char*)dst, bytes);
    }
    return launched();
}

extern "C" int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                                   uint64_t width, uint64_t height, void* stream) {
    if (width == 0 || height == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    const unsigned by = (unsigned)(height < 65535 ? height : 65535);
    if (aligned16((uint64_t)(uintptr_t)dst | (uint64_t)(uintptr_t)src | dpitch | spitch | width)) {
        const uint64_t w16 = width / 16;
        hipLaunchKernelGGL(aql_copy2d16, dim3(blocks_for(w16, kThreads), by), dim3(kThreads), 0, s,
                           (const u32x4*)src, (u32x4*)dst, spitch / 16, dpitch / 16, w16, height);
    } else {
        hipLaunchKernelGGL(aql_copy2d1, dim3(blocks_for(width, kThreads), by), dim3(kThreads), 0, s,
                           (const unsigned char*)src, (unsigned char*)dst, spitch, dpitch, width,
                           height);
    }
    return launched();
}

extern "C" int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix,
                                   const RadCopyRange* ranges, int n, void* stream) {
    const uint64_t n16 = prefix / 16;
    const unsigned pb = n16 ? blocks_for(n16, 4 * kThreads) : 0;
    if (pb == 0 && n == 0) return 0;
    hipLaunchKernelGGL(aql_copy_ranges, dim3(pb + (unsigned)n), dim3(kThreads), 0,
                       (hipStream_t)stream, (const u32x4*)src, (u32x4*)dst, n16, pb, ranges);
    return launched();
}

extern "C" int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream) {
    if (bytes == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    const unsigned char b = (unsigned char)value;
    if (aligned16((uint64_t)(uintptr_t)dst | bytes)) {
        const unsigned v = 0x01010101u * b;
        const uint64_t n = bytes / 16;
        hipLaunchKernelGGL(aql_fill16, dim3(blocks_for(n, 4 * kThreads)), dim3(kThreads), 0, s,
                           (u32x4*)dst, v, n);
    } else {
        hipLaunchKernelGGL(aql_fill1, dim3(blocks_for(bytes, 4 * kThreads)), dim3(kThreads), 0, s,
                           (unsigned char*)dst, b, bytes);
    }
    return launched();
}

extern "C" int rad_aql_kv_compact(const radiance::RadKVCompactParams& p, RadStream stream) {
    // The AQL queue copies these arguments before returning; no staged move buffer is needed.
    static_assert(sizeof(radiance::RadKVCompactParams) + 256 <= 768,
                  "KV compaction plus AMDGPU implicit arguments must fit an AQL slot");
    void* args[] = {const_cast<radiance::RadKVCompactParams*>(&p)};
    const hipError_t status = hipLaunchKernel(
        reinterpret_cast<const void*>(radiance::rad_kv_compact_winning_branch_kernel),
        dim3(p.n_heads, p.n_layers), dim3(32), args, 0, (hipStream_t)stream);
    return status == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
