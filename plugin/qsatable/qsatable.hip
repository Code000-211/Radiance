/* plugin/qsatable/qsatable.hip -- the device launch for `qsa_table`.
 *
 * ONE WORKGROUP A TILE, 256 threads. A tile is at most 32 query rows, so the membership of a block
 * fits a 32-bit word: bit r says row r of the tile holds the block. The union is then the blocks
 * whose word is non-zero, in ascending block index -- block order -- and the mask is that bit.
 *
 * TWO PASSES over the block range, each carrying a running output counter across the strided
 * chunks, so the table fills in ascending order without a global scan: the first counts the union's
 * length, the second writes it. Membership is a binary search in the row's `selb` run, which is
 * ascending by contract; the row's complete-block count `(pos + 1) / ratio` bounds it, and that bound
 * is also what keeps the partial page out -- the partial page's block index IS that count, so it can
 * never be below it.
 *
 * PROTOTYPE. `blocks * tile * log(topk)` per tile is fine at emulator and bench sizes and is not what
 * a serving kernel would do; what this row is for is a test article with a host oracle beside it.
 */
#include <hip/hip_runtime.h>

#include "qsatable.h"
#include "../../libr4d/r4d_emit.h"

namespace {

constexpr int kThreads = 256;

/* Does row `r` hold block `b`? `own` is the row's complete-block count: entries at or above it are
 * the partial page, which no table carries. */
__device__ bool holds(const int32_t* row, int64_t n, int64_t own, int64_t b) {
    if (b >= own || n < 1) return false;
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        const int64_t v = row[mid];
        if (v < b) lo = mid + 1; else hi = mid;
    }
    return lo < n && row[lo] == b;
}

__device__ unsigned word_at(const RadQsaTableCall& c, const int64_t* own, int64_t r0, int64_t r1,
                            int64_t b) {
    unsigned word = 0;
    for (int64_t r = r0; r < r1; ++r)
        if (holds(c.selb + r * c.ld_selb, c.nsel[r], own[r - r0], b)) word |= 1u << (r - r0);
    return word;
}

__global__ void qsa_table_kernel(RadQsaTableCall c) {
    const int64_t tile = blockIdx.x;
    if (tile >= c.tiles) return;
    const int tid = threadIdx.x;
    __shared__ int64_t own[32];   /* each row's complete-block count, tile-local */
    __shared__ int64_t scan[kThreads];
    __shared__ int64_t base;      /* the running output counter before this chunk */
    __shared__ int64_t tail;      /* the union's length, once the counting pass is done */

    if (tid == 0) {
        for (int64_t i = 0; i < c.tile; ++i) {
            const int64_t r = tile * c.tile + i;
            own[i] = r < c.M ? (c.pos[r] + 1) / c.ratio : 0;
        }
        base = 0;
    }
    __syncthreads();
    const int64_t r0 = tile * c.tile;
    const int64_t r1 = r0 + c.tile < c.M ? r0 + c.tile : c.M;

    /* PASS 1: the union's length. */
    for (int64_t b0 = 0; b0 < c.blocks; b0 += kThreads) {
        const int64_t b = b0 + tid;
        scan[tid] = b < c.blocks && word_at(c, own, r0, r1, b) ? 1 : 0;
        __syncthreads();
        for (int stride = 1; stride < kThreads; stride <<= 1) {
            const int64_t add = tid >= stride ? scan[tid - stride] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int64_t chunk = scan[kThreads - 1];
        __syncthreads();
        if (tid == 0) base += chunk;
        __syncthreads();
    }
    if (tid == 0) tail = base;
    if (tid == 0) base = 0;
    __syncthreads();

    /* PASS 2: write the union in ascending block order and the mask beside it. */
    int32_t* tab = c.tab + tile * c.u_max;
    for (int64_t b0 = 0; b0 < c.blocks; b0 += kThreads) {
        const int64_t b = b0 + tid;
        const unsigned word = b < c.blocks ? word_at(c, own, r0, r1, b) : 0u;
        scan[tid] = word ? 1 : 0;
        __syncthreads();
        for (int stride = 1; stride < kThreads; stride <<= 1) {
            const int64_t add = tid >= stride ? scan[tid - stride] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int64_t chunk = scan[kThreads - 1];
        const int64_t slot = base + scan[tid] - 1;
        if (word && slot < c.u_max) {
            tab[slot] = (int32_t)b;
            for (int64_t r = r0; r < r1; ++r)
                c.mask[r * c.u_max + slot] = (int32_t)((word >> (r - r0)) & 1u);
        }
        __syncthreads();
        if (tid == 0) base += chunk;
        __syncthreads();
    }

    /* The tail past `tlen`, the mask's entries past it and the rows past the tile: written, not left
     * to whatever the arena held. */
    for (int64_t j = tail + tid; j < c.u_max; j += kThreads) tab[j] = 0;
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t j = tail + tid; j < c.u_max; j += kThreads) c.mask[r * c.u_max + j] = 0;
    for (int64_t r = r0 + tid; r < r1; r += kThreads) c.row_off[r] = (int32_t)(tile * c.u_max);
    if (tid == 0) c.tlen[tile] = (int32_t)tail;
}

}  /* namespace */

int rad_qsa_table_launch(const RadArgs* a, RadStream stream) {
    RadQsaTableCall c{};
    const int rc = rad_qsa_table_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, qsa_table_kernel, dim3((unsigned)c.tiles), dim3(kThreads), 0, c);
}
