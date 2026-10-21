// The ten small bf16 ops the engine issues between the GEMMs, and the reason they are here at all.
//
// The rest of this library is kernels worth writing: a WMMA GEMM, a paged attention, a chunked
// scan.  These are not -- they are an embedding gather, a norm, an add, a sigmoid.  They
// are here because a step is not a set of interesting kernels, it is a CHAIN, and a chain with one
// host link does not run: the engine's buffers live in VRAM, libref is domain HOST, and
// `rad_issue` refuses an op with no launch for the domain its operands are on.  Twenty-one ops
// short of a device implementation is not "slower", it is a server that answers the first request
// with `no device kernel for this band` and exits.  So: one kernel library, and this is where a
// kernel with no other home goes -- including the ops nobody would choose to write.
//
// THE ARITHMETIC IS libref's, DELIBERATELY.  Ref is the oracle (spec §17) and these are the
// ops whose disagreements would be invisible -- an RMS norm that accumulates in bf16, a gather that
// reads a negative id as unsigned, a rotary that pairs the wrong halves.  Each kernel below states
// which ref function it must equal.  Where ref rounds, this rounds, in the same place:
//
//   * every reduction accumulates in f32 and only the store narrows to bf16;
//   * `f32_to_bf16` is round-to-nearest-even (r4d_common.h), which is ref's rounder;
//   * a NEGATIVE index -- token id, gather row, KV slot -- is the PADDING convention and writes a
//     zero row or skips the store.  It is not an error and it is not clamped to zero, which would
//     silently duplicate token 0 across every padding position of a ragged batch.
//
// ONE DELIBERATE DIVERGENCE FROM REF, and it is about what a kernel can do rather than about the
// arithmetic.  Ref reads its index arrays on the host and REFUSES the launch when one is out of
// range -- an id at or above n_vocab, a gather row past the source, a KV slot past the last block.
// A device kernel cannot: the indices are in VRAM, checking them means a copy and a synchronisation
// on the step path, and a per-row `return` cannot become a status.  So the bound is an ARGUMENT and
// an out-of-range index takes the same path as a negative one -- a zero row, or no store.  That is
// a defined answer to a caller bug; the alternative is a page fault that takes the server down.
//
// AND ONE ON PRECISION: `expf`, `powf` and `sincosf` here are the ACCURATE forms, not the `__`
// hardware approximations.  A rope angle is position * inv_freq and position reaches 262144, where
// __sincosf's argument reduction has nothing left; and every one of these ops is an oracle
// comparison away from libref, which uses the libm forms.  None of them is on a hot path
// that would notice.
//
// WHAT IS NOT HERE.  No fusion.  `rmsnorm` and `add` next to each other are the unfused sequence
// the architecture plugin asked for, and fusing them is a SELECTION the plugin makes by declaring
// `rmsnorm_add` (spec §2.3), not something a kernel library does behind its back.  When the fused
// forms are wanted they are new entry points and new rows, not a rewrite of these.
//
// Every entry point returns a negative code rather than throwing, matching r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"

#include <cstdlib>
#include <mutex>

namespace {

struct PleVramCache {
    int32_t* tags     = nullptr;
    uint8_t* heat     = nullptr;
    uint8_t* data     = nullptr;
    uint32_t num_sets = 0;
};

static std::mutex g_ple_mu;
static PleVramCache g_ple_caches[8];

static PleVramCache get_ple_vram_cache(int dev_id, int n_embd) {
    if (dev_id < 0 || dev_id >= 8 || n_embd != 160) {
        return PleVramCache{};
    }
    PleVramCache& c = g_ple_caches[dev_id];
    if (c.tags) {
        return c; // Fast lock-free path
    }
    std::lock_guard<std::mutex> lk(g_ple_mu);
    if (c.tags) return c;

    const char* sets_env = std::getenv("RADIANCE_PLE_VRAM_CACHE_SETS");
    uint32_t num_sets = sets_env ? (uint32_t)std::strtoul(sets_env, nullptr, 0) : 65536u;
    if (num_sets == 0) return PleVramCache{};
    if ((num_sets & (num_sets - 1)) != 0) {
        num_sets = 1u << (31 - __builtin_clz(num_sets));
    }

    const uint32_t total_entries = num_sets * 4u;
    const size_t tag_bytes  = (size_t)total_entries * sizeof(int32_t);
    const size_t heat_bytes = (size_t)total_entries * sizeof(uint8_t);
    const size_t data_bytes = (size_t)total_entries * 320u; // 160 elements x 2 bytes (BF16)

    (void)hipSetDevice(dev_id);
    if (hipMalloc((void**)&c.tags, tag_bytes) != hipSuccess) {
        c.tags = nullptr;
        return PleVramCache{};
    }
    if (hipMalloc((void**)&c.heat, heat_bytes) != hipSuccess) {
        (void)hipFree(c.tags);
        c.tags = nullptr;
        return PleVramCache{};
    }
    if (hipMalloc((void**)&c.data, data_bytes) != hipSuccess) {
        (void)hipFree(c.tags);
        (void)hipFree(c.heat);
        c.tags = nullptr;
        c.heat = nullptr;
        return PleVramCache{};
    }
    (void)hipMemset(c.tags, 0xFF, tag_bytes);
    (void)hipMemset(c.heat, 0, heat_bytes);
    c.num_sets = num_sets;
    fprintf(stderr, "I[ple] device %d: allocated 4-Way Set-Associative RLFU L1 VRAM N-Gram Cache: "
                    "%u sets x 4 ways = %u entries (%.2f MiB VRAM: %zu KiB tags, %zu KiB heat, %.1f MiB data [dequantized BF16])\n",
            dev_id, num_sets, total_entries, (tag_bytes + heat_bytes + data_bytes) / (1024.0 * 1024.0),
            tag_bytes / 1024, heat_bytes / 1024, data_bytes / (1024.0 * 1024.0));
    return c;
}

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

__device__ __forceinline__ float bf2f(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

/* An orthonormal Walsh-Hadamard over N values a THREAD owns, in place and its own inverse. Only
 * `kv_store_kernel`'s quality probe uses it, and only there does a sequential butterfly make sense:
 * r4d_fwht.h's is the one to reuse anywhere speed matters, and it wants a WAVE per N-wide block
 * with N/32 values a lane. N must be a power of two, which every scale group the probe accepts is. */
template <uint32_t N>
__device__ __forceinline__ void hadamard(float* a) {
    static_assert(N && (N & (N - 1u)) == 0u, "the Walsh-Hadamard width must be a power of two");
    for (uint32_t len = 1; len < N; len <<= 1)
        for (uint32_t i = 0; i < N; i += len << 1)
            for (uint32_t j = 0; j < len; ++j) {
                const float u = a[i + j], w = a[i + j + len];
                a[i + j] = u + w;
                a[i + j + len] = u - w;
            }
    const float s = rsqrtf((float)N);
    for (uint32_t e = 0; e < N; ++e) a[e] *= s;
}

// Sum across the block, broadcast to every thread. One LDS round trip after the wave reduction,
// which is what the norms need and nothing else here does.
template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// ---------------------------------------------------------------------------- embed_lookup
// ref_embed_lookup.  One block a token; a negative id -- or one at or above n_vocab, see the
// divergence note at the top -- writes a zero row.
//
// `vocab_off` is the first GLOBAL row this rank's table stands for, which makes the embedding
// vocab-parallel: ids stay global, a rank subtracts the offset, and a token belonging to another
// rank takes the same zero-row path the padding id takes.  Summing the ranks then reconstructs x.
// At vocab_off = 0 this is the plain gather, to the bit.
__global__ __launch_bounds__(kThreads) void embed_lookup_kernel(
    const int32_t* __restrict__ tok, const uint16_t* __restrict__ wte, uint16_t* __restrict__ x,
    uint32_t n_embd, int32_t n_vocab, int32_t vocab_off, uint32_t x_ld) {
    const uint32_t m = blockIdx.x;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld;
    if (id < 0 || id >= n_vocab) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = 0;
        return;
    }
    const uint16_t* __restrict__ src = wte + (size_t)id * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = src[i];
}

// ---------------------------------------------------------------------------- embed_lookup_q
// ref_embed_lookup_q.  x[m] = bf16(wte[id] * scale), from a table stored as E4M3 codes with ONE
// scale (Q) or as plain bf16, where an absent scale is one.  This is the n-gram table's gather
// (arch/common/rad_block_ple.h) wherever --ngram-placement lets a card read the table: in VRAM, or
// in pinned host memory the card reads in place over the link.
//
// A THREAD A QUAD OF ENTRIES, NOT A BLOCK A ROW.  A row of the n-gram table is 160 entries, so a
// block a row leaves most of its threads idle and the grid is as deep as the chunk's row count.
// Flattened, consecutive threads take consecutive quads, and a row's 40 threads read its 160
// contiguous bytes.  V = 1 is the same walk an entry at a time, for a width or a pitch the quad's
// 4- and 8-byte accesses would misalign.
//
// THE VALUE libavx's LOOKUP TABLE HOLDS, to the bit: a code decodes to f32 exactly
// (v_cvt_pk_f32_fp8, or the software form beside it in r4d_fp8_wmma.h), the product with the
// scale rounds once in f32, and f32_to_bf16 rounds to nearest even.  An id outside [0, n_vocab)
// after the offset writes a zero row (the divergence note at the top of this file).
// ---------------------------------------------------------------------------- ple_embed_lookup_q_kernel
// Specialized 1-wave-per-row kernel for PLE N-Gram table (n_embd = 160, FP8 E4M3).
// Exactly 32 threads (1 wavefront) per row: threads 0..19 process 8 elements each (uint2 FP8 -> uint4 BF16).
// Lane 0 evaluates the 4-way L1 cache tags once and broadcasts the result via __shfl,
// eliminating 19 redundant memory loads of tags and branches!
// All 20 threads reside in wave 0, guaranteeing zero cross-wave races and zero straddling.
__global__ __launch_bounds__(32) void ple_embed_lookup_q_kernel(
    const int32_t* __restrict__ tok, const void* __restrict__ wte, const void* __restrict__ sc,
    uint32_t sc_f32, uint16_t* __restrict__ x, uint32_t M, int32_t n_vocab,
    int32_t vocab_off, uint32_t x_ld,
    int32_t* __restrict__ cache_tags, uint8_t* __restrict__ cache_heat,
    uint8_t* __restrict__ cache_data, uint32_t num_sets) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint32_t lane = threadIdx.x;

    const int32_t id = tok[m] - vocab_off;
    if (id < 0 || id >= n_vocab) {
        if (lane < 20u) {
            uint16_t* __restrict__ dst = x + (size_t)m * x_ld + lane * 8u;
            *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }

    uint32_t slot = 0;
    uint32_t is_hit = 0;

    if (cache_tags && cache_heat && cache_data && num_sets > 0) {
        if (lane == 0) {
            const uint32_t set = ((uint32_t)id * 2654435761u) & (num_sets - 1);
            const uint32_t base = set * 4u;
            const int4 tags = *(const int4*)(cache_tags + base);

            int way = -1;
            if      (tags.x == id) way = 0;
            else if (tags.y == id) way = 1;
            else if (tags.z == id) way = 2;
            else if (tags.w == id) way = 3;

            if (way >= 0) {
                slot = base + (uint32_t)way;
                is_hit = 1;
            } else {
                int victim_way = 0;
                if      (tags.x < 0) { victim_way = 0; }
                else if (tags.y < 0) { victim_way = 1; }
                else if (tags.z < 0) { victim_way = 2; }
                else if (tags.w < 0) { victim_way = 3; }
                else {
                    const uint32_t h4 = *(const uint32_t*)(cache_heat + base);
                    const uint32_t h0 = h4 & 0xFFu;
                    const uint32_t h1 = (h4 >> 8) & 0xFFu;
                    const uint32_t h2 = (h4 >> 16) & 0xFFu;
                    const uint32_t h3 = (h4 >> 24) & 0xFFu;
                    uint32_t min_h = h0;
                    if (h1 < min_h) { min_h = h1; victim_way = 1; }
                    if (h2 < min_h) { min_h = h2; victim_way = 2; }
                    if (h3 < min_h) { min_h = h3; victim_way = 3; }
                }
                slot = base + (uint32_t)victim_way;
                is_hit = 0;
            }
        }
        slot = __shfl(slot, 0, 32);
        is_hit = __shfl(is_hit, 0, 32);
    }

    if (lane < 20u) {
        const uint32_t j = lane * 8u;
        uint16_t* __restrict__ dst = x + (size_t)m * x_ld + j;

        if (cache_tags && cache_heat && cache_data && num_sets > 0) {
            const size_t c_at = (size_t)slot * 320u + (size_t)lane * 16u;
            if (is_hit) {
                // 4-WAY L1 VRAM HIT: Direct 128-bit vector load! Zero dequant arithmetic!
                *(uint4*)dst = *(const uint4*)(cache_data + c_at);
            } else {
                // 4-WAY L1 VRAM MISS: Load FP8 from L2 Host RAM, dequantize, store to dst and L1 cache
                const size_t at = (size_t)id * 160u + j;
                const uint2 q2 = *(const uint2*)((const uint8_t*)wte + at);
                const float s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : bf16_to_f32(*(const uint16_t*)sc);
                const uint32_t q0 = q2.x, q1 = q2.y;
                const v2f lo0 = r4d_fp8::e4m3_pk<false>(q0), hi0 = r4d_fp8::e4m3_pk<true>(q0);
                const v2f lo1 = r4d_fp8::e4m3_pk<false>(q1), hi1 = r4d_fp8::e4m3_pk<true>(q1);

                const uint4 bf16_quad = make_uint4(
                    (uint32_t)f32_to_bf16(lo0.x * s) | ((uint32_t)f32_to_bf16(lo0.y * s) << 16),
                    (uint32_t)f32_to_bf16(hi0.x * s) | ((uint32_t)f32_to_bf16(hi0.y * s) << 16),
                    (uint32_t)f32_to_bf16(lo1.x * s) | ((uint32_t)f32_to_bf16(lo1.y * s) << 16),
                    (uint32_t)f32_to_bf16(hi1.x * s) | ((uint32_t)f32_to_bf16(hi1.y * s) << 16));
                *(uint4*)dst = bf16_quad;
                *(uint4*)(cache_data + c_at) = bf16_quad;
            }
        } else {
            const size_t at = (size_t)id * 160u + j;
            const uint2 q2 = *(const uint2*)((const uint8_t*)wte + at);
            const float s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : bf16_to_f32(*(const uint16_t*)sc);
            const uint32_t q0 = q2.x, q1 = q2.y;
            const v2f lo0 = r4d_fp8::e4m3_pk<false>(q0), hi0 = r4d_fp8::e4m3_pk<true>(q0);
            const v2f lo1 = r4d_fp8::e4m3_pk<false>(q1), hi1 = r4d_fp8::e4m3_pk<true>(q1);

            *(uint4*)dst = make_uint4(
                (uint32_t)f32_to_bf16(lo0.x * s) | ((uint32_t)f32_to_bf16(lo0.y * s) << 16),
                (uint32_t)f32_to_bf16(hi0.x * s) | ((uint32_t)f32_to_bf16(hi0.y * s) << 16),
                (uint32_t)f32_to_bf16(lo1.x * s) | ((uint32_t)f32_to_bf16(lo1.y * s) << 16),
                (uint32_t)f32_to_bf16(hi1.x * s) | ((uint32_t)f32_to_bf16(hi1.y * s) << 16));
        }
    }

    if (cache_tags && cache_heat && num_sets > 0) {
        if (!is_hit) {
            __threadfence();
        }
        if (lane == 0) {
            if (is_hit) {
                const uint8_t h = cache_heat[slot];
                if (h < 255) cache_heat[slot] = h + 1;
            } else {
                cache_tags[slot] = id;
                cache_heat[slot] = 4;
            }
        }
    }
}

template <uint32_t V, bool Q>
__global__ __launch_bounds__(kThreads) void embed_lookup_q_kernel(
    const int32_t* __restrict__ tok, const void* __restrict__ wte, const void* __restrict__ sc,
    uint32_t sc_f32, uint16_t* __restrict__ x, uint32_t M, uint32_t n_embd, int32_t n_vocab,
    int32_t vocab_off, uint32_t x_ld) {
    const uint32_t per_row = n_embd / V;
    const uint64_t g = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    if (g >= (uint64_t)M * per_row) return;
    const uint32_t m = (uint32_t)(g / per_row), j = (uint32_t)(g % per_row) * V;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld + j;
    if (id < 0 || id >= n_vocab) {
        if constexpr (V == 4) *(uint2*)dst = make_uint2(0u, 0u);
        else                  dst[0] = 0;
        return;
    }
    const float s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : bf16_to_f32(*(const uint16_t*)sc);
    const size_t at = (size_t)id * n_embd + j;
    float v[V];
    if constexpr (Q && V == 4) {
        const uint32_t q = *(const uint32_t*)((const uint8_t*)wte + at);
        const v2f lo = r4d_fp8::e4m3_pk<false>(q), hi = r4d_fp8::e4m3_pk<true>(q);
        v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
    } else if constexpr (Q) {
        v[0] = r4d_fp8::e4m3(((const uint8_t*)wte)[at]);
    } else if constexpr (V == 4) {
        const uint2 b = *(const uint2*)((const uint16_t*)wte + at);
        v[0] = bf16_to_f32((uint16_t)b.x); v[1] = bf16_to_f32((uint16_t)(b.x >> 16));
        v[2] = bf16_to_f32((uint16_t)b.y); v[3] = bf16_to_f32((uint16_t)(b.y >> 16));
    } else {
        v[0] = bf16_to_f32(((const uint16_t*)wte)[at]);
    }
    if constexpr (V == 4) {
        *(uint2*)dst = make_uint2(
            (uint32_t)f32_to_bf16(v[0] * s) | ((uint32_t)f32_to_bf16(v[1] * s) << 16),
            (uint32_t)f32_to_bf16(v[2] * s) | ((uint32_t)f32_to_bf16(v[3] * s) << 16));
    } else {
        dst[0] = f32_to_bf16(v[0] * s);
    }
}

// ---------------------------------------------------------------------------- gather_rows
// ref_gather_rows.  Same shape as the embedding gather and the same negative-index rule, but the
// source row pitch is the caller's rather than n: a gather off a padded activation buffer is what
// the op exists for (one logits row per sequence out of a whole prefill chunk).
__global__ __launch_bounds__(kThreads) void gather_rows_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ idx, uint16_t* __restrict__ y,
    uint32_t n, int32_t src_rows, uint32_t x_ld, uint32_t y_ld) {
    if (!x || !idx || !y) return;
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    uint16_t* __restrict__ dst = y + (size_t)m * y_ld;
    if (r < 0 || r >= src_rows) {
        for (uint32_t i = threadIdx.x; i < n; i += kThreads) dst[i] = 0;
        return;
    }
    const uint16_t* __restrict__ src = x + (size_t)r * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) dst[i] = src[i];
}


// ---------------------------------------------------------------- byte row gather / scatter
//
// ref_gather_rows and ref_scatter_rows at dtype u8.  A SEPARATE PAIR FROM THE bf16 ONES, and not
// a widened version of them, because these move OPAQUE BYTES: the KV pool's tier traffic is the
// caller, a row is one layer's fragment of one block, and what is in it is whatever the attention
// kernel wrote.  Reading that as bf16 would be a lie with teeth -- libref is the oracle for every
// op here, its bf16 path loads through float, and a bf16 NaN payload is not guaranteed to survive
// that round trip.  At u8 every value is 0..255 and float carries it exactly, so oracle and kernel
// agree bit for bit on any payload at all.
//
// 16 BYTES A THREAD WHERE THE ROW ALLOWS IT.  A KV fragment is a tensor extent and is aligned; the
// registry constrains `n` to a multiple of 16 so this is the only path, and a row that did not
// satisfy it would not have resolved here.
__global__ __launch_bounds__(kThreads) void gather_rows_u8_kernel(
    const uchar4* __restrict__ x, const int32_t* __restrict__ idx, uchar4* __restrict__ y,
    uint32_t n4, int32_t src_rows, uint32_t x_ld4, uint32_t y_ld4) {
    if (!x || !idx || !y) return;
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    uchar4* __restrict__ dst = y + (size_t)m * y_ld4;
    if (r < 0 || r >= src_rows) {
        const uchar4 z = make_uchar4(0, 0, 0, 0);
        for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = z;
        return;
    }
    const uchar4* __restrict__ src = x + (size_t)r * x_ld4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = src[i];
}

// The mirror.  A negative index writes NOTHING -- the destination is a live pool and the rows this
// call was not asked about belong to somebody else, so the zero-fill the gather does for an absent
// row would be data loss here rather than padding.
__global__ __launch_bounds__(kThreads) void scatter_rows_u8_kernel(
    const uchar4* __restrict__ v, const int32_t* __restrict__ idx, uchar4* __restrict__ x,
    uint32_t n4, int32_t dst_rows, uint32_t v_ld4, uint32_t x_ld4) {
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    if (r < 0 || r >= dst_rows) return;
    const uchar4* __restrict__ src = v + (size_t)m * v_ld4;
    uchar4* __restrict__ dst = x + (size_t)r * x_ld4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = src[i];
}

// ---------------------------------------------------------------------------- rmsnorm
// ref_rmsnorm: y = x * rsqrt(mean(x^2) + eps) * (w + wadd).
//
// THE GAIN IS ADDED IN f32 AFTER THE bf16 LOAD.  `wadd` is GemmaRMSNorm's `1 +` (Qwen3.5,
// Qwen3-Next, Gemma store a zero-centred gamma); forming `1 + w` in bf16 first loses 0.3% per
// channel, which is why r4d_rmsnorm_had_quant_i8 takes the same argument and why it is not the
// caller's job to bake it into the weight.
//
// THE GAIN IS f32 OR bf16 AND THE CALLER SAYS WHICH.  Every norm gain in this project's
// architecture plugins is declared RAD_F32 -- they are a few thousand elements against gigabytes of
// weights, and the norm is where a rounded gain shows -- while the activation either side is bf16.
// A kernel that assumed one width would read the other as two halves of the wrong number, so the
// width is a template parameter and the shim reads it off the operand.
//
// Two passes over the row.  n is 256 (a QK-norm head) or 5120 (the hidden state), so the row is in
// L1 for the second pass and a one-pass Welford would buy nothing but a different rounding.
template <bool WF32, uint32_t THR>
__global__ __launch_bounds__(THR) void rmsnorm_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = bf2f(xr[i]); ss += v * v; }
    ss = block_sum<THR / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float g = WF32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
        yr[i] = f32_to_bf16(bf2f(xr[i]) * s * (g + wadd));
    }
}

// ---------------------------------------------------------------------------- add / mul
// ref_add / ref_mul, including THE ONE BROADCAST RULE: an operand carrying exactly n elements
// against an output of rows*n broadcasts along rows, and nothing else broadcasts at all.  That is
// what makes a bias add this op instead of a second one, and the narrowness is the point -- silent
// broadcasting of an arbitrary shape is how a shape bug becomes a plausible number.
__global__ __launch_bounds__(kThreads) void binary_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, int mul) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ ar = a + (size_t)m * a_ld;
    const uint16_t* __restrict__ br = b + (size_t)m * b_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float u = bf2f(ar[i]), v = bf2f(br[i]);
        yr[i] = f32_to_bf16(mul ? u * v : u + v);
    }
}

// ---------------------------------------------------------------------------- sigmoid
// ref_sigmoid.  Its own op and not a mode on a unary kernel for the same reason it is its own row:
// a gated attention output multiplies by sigmoid(gate) and the selector has to be able to name that.
//
// ROW-WISE WITH A PITCH, not flat over the extent, because the operand it is issued against is a
// COLUMN SLICE: the gate is the second half of each head of an interleaved [tokens, heads,
// 2*head_dim] projection, so its rows are `n` wide inside a stride of 2n and there is no contiguous
// extent to walk.  A flat kernel would either refuse it or read the query.
__global__ __launch_bounds__(kThreads) void sigmoid_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(1.0f / (1.0f + expf(-bf2f(xr[i]))));
}

// ---------------------------------------------------------------------------- silu_mul
// ref_silu_mul: y[.., i] = silu(gu[.., i]) * gu[.., n + i].  THE GATE IS THE FIRST HALF, which is
// the order rad-convert fuses gate_proj and up_proj into one weight in; the two conventions cannot
// drift because both are stated and neither is inferred.
__global__ __launch_bounds__(kThreads) void silu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = bf2f(gr[i]);
        yr[i] = f32_to_bf16((g / (1.0f + expf(-g))) * bf2f(gr[n + i]));
    }
}

// ---------------------------------------------------------------------------- rope
// ref_rope, in place over a fused [n_head + 2*n_head_kv] head row -- the V heads at the end are
// left alone, which is the whole reason the op takes the fused projection rather than two tensors.
//
// NeoX pairs element i with i + rot/2; GPT-J pairs 2i with 2i+1.  Which one a checkpoint wants is a
// property of the checkpoint, and getting it wrong produces fluent text with a broken sense of
// position -- the hardest kind of wrong to notice.  `mrope` is NOT a third case: RadBatch carries
// one position component per token, and mrope over equal components is bit-identical to neox
// (rad_block_attn_gated.h says so and says what would have to change first), so the shim maps it.
//
// PARTIAL ROTARY.  `rot` may be less than head_dim -- 64 of 256 on this target -- and the tail of
// the head is untouched.  The frequency exponent divides by `rot`, not by head_dim: the rotated
// prefix is a complete rotary of its own width.
//
// One thread a PAIR.  The pair's two loads and two stores are the whole body, so there is no
// reduction and no LDS; the grid is (M, heads).
//
// THE MULTI-COMPONENT MODES (ref_rope's rope_freqs): `mode` 2 mrope, 3 imrope, 4 axial, with up to
// three sections, read a position per component out of `plane`-strided rows -- component c of
// token m at pos[c * plane + m]. `plane` 0 is the one-component form, where every mode is neox.
__device__ __forceinline__ int rope_comp(int mode, uint32_t i, const int* sec, int n_sec,
                                         uint32_t* j) {
    *j = i;
    if (mode == 3) {
        if (i % 3 == 1 && (int)i < 3 * sec[1]) return 1;
        if (i % 3 == 2 && (int)i < 3 * sec[2]) return 2;
        return 0;
    }
    int c = 0, start = 0;
    while (c + 1 < n_sec && (int)i >= start + sec[c]) { start += sec[c]; ++c; }
    *j = i - (uint32_t)start;
    return c;
}

__global__ __launch_bounds__(kThreads) void rope_kernel(
    uint16_t* __restrict__ qkv, const int32_t* __restrict__ pos,
    uint32_t head_dim, uint32_t n_rope_heads, uint32_t rot, int gptj,
    float theta, float rscale, uint32_t row_pitch,
    int mode, int s0, int s1, int s2, int n_sec, uint32_t plane) {
    const uint32_t m = blockIdx.x;
    const uint32_t half = rot / 2;
    const float div = rscale != 0.0f ? rscale : 1.0f;
    const float p0 = (float)pos[m] / div;
    const int sec[3] = { s0, s1, s2 };

    for (uint32_t t = threadIdx.x + blockIdx.y * kThreads; t < n_rope_heads * half;
         t += kThreads * gridDim.y) {
        const uint32_t h = t / half, i = t % half;
        uint16_t* __restrict__ hd = qkv + (size_t)m * row_pitch + (size_t)h * head_dim;
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rot);
        float p = p0;
        if (plane) {
            uint32_t j = i;
            const int c = rope_comp(mode, i, sec, n_sec, &j);
            p = (float)pos[(size_t)c * plane + m] / div;
            if (mode == 4)
                inv = 1.0f / powf(theta, (float)(2 * j) / (float)(2 * sec[c]));
        }
        const float ang = p * inv;
        float c, s;
        sincosf(ang, &s, &c);
        const uint32_t i0 = gptj ? 2 * i : i;
        const uint32_t i1 = gptj ? 2 * i + 1 : i + half;
        const float x0 = bf2f(hd[i0]), x1 = bf2f(hd[i1]);
        hd[i0] = f32_to_bf16(x0 * c - x1 * s);
        hd[i1] = f32_to_bf16(x0 * s + x1 * c);
    }
}

// ---------------------------------------------------------------------------- kv_store
// ref_kv_store, and THE PAGED LAYOUT IS THE ONE EVERY KERNEL IN THIS PROJECT SHARES:
//
//     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]
//
// K first and V second inside a slot: element d of the key at [..., d] and of the value at
// [..., head_dim + d].  A flat slot index decomposes as block = slot / block_size and
// offset = slot % block_size, which is what `slot_mapping` carries.  A NEGATIVE slot is a padding
// position and stores nothing.
template <uint32_t QG, uint32_t QF, uint32_t QW, uint32_t QB>
__global__ __launch_bounds__(kThreads) void kv_store_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint16_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * 2 * head_dim;
    const size_t head_step = (size_t)block_size * 2 * head_dim;

    if constexpr (QG) {
        /* ---- A QUALITY PROBE, NOT A FORMAT. `R4D_KV_QSIM=<g>` round-trips every `g`-element
         * group of k and of v through a quantiser with its own scale and stores the result as
         * ordinary bf16. Nothing about the cache's layout, size or bandwidth changes -- the point
         * is to answer "would this format still write correct code" BEFORE any of that is built,
         * rather than building a real cache to find out.
         *
         * THREE KNOBS, and each one is a question:
         *   R4D_KV_QSIM=<g>   the scale group. int8 spends its 8 bits on the range the group
         *                     occupies, so the components that carry the dot product land near
         *                     bf16 and only the small ones are coarse.
         *   R4D_KV_QSIMF=1    snap to E4M3 instead of int8. E4M3 keeps 3 mantissa bits at EVERY
         *                     magnitude -- ~6% relative everywhere -- so a scale cannot change its
         *                     step, only keep a small head out of the subnormals below 2^-6.
         *   R4D_KV_QSIMW=1    ROTATE the group by an orthonormal Walsh-Hadamard first. A scale is
         *                     set by the group's largest element, so one outlier channel costs
         *                     every other element its resolution; the rotation spreads that
         *                     outlier over all QG channels. It is why this engine's own WIRE codec
         *                     is `wht6` and not `u6`.
         *
         * THE ROTATION IS FREE TO DEPLOY ON BOTH SIDES, which is what makes it worth measuring.
         * The transform is orthonormal and its own inverse, so
         *
         *     score[t] = q . k[t]      = (R q) . (R k[t])
         *     out      = sum p[t] v[t] = R^T ( sum p[t] (R v[t]) )
         *
         * -- a cache holding R k and R v needs the QUERY rotated once per (token, head) and the
         * OUTPUT rotated back once per (token, head). Neither is per CACHED token, so the cost
         * does not grow with context, which is the whole reason a KV cache is expensive. This
         * probe rotates back itself instead, because it stores bf16: same values, one extra round.
         *
         * WHAT THE PROBE SAYS ABOUT EIGHT BITS. Top-1 agreement against a bf16 control is a
         * Bernoulli trial a prompt, so at p ~ 0.93 its standard error is a couple of points at any
         * prompt count worth running -- and every 8-bit arm sits inside that band at ~93-94%
         * agreement. The mean absolute delta, which has thousands of samples rather than hundreds,
         * separates them by only ~5% end to end: int8 with a per-head scale and a rotated K at
         * 0.196, int8 with a scale per 32 at 0.199, unscaled fp8 e4m3 at 0.205.
         *
         * THE CASE FOR AN INTEGER KV CACHE IS THEREFORE WEAK: fp8 is already built, is DENSER
         * (50.0% of bf16 against 50.8% for scaled int8), and measures the same. The rotation does
         * not change that. The rotation MECHANISM is real all the same -- at int4 it is worth
         * +5.0 points of agreement and takes |delta| from 0.308 to 0.255 -- there is simply little
         * left for it to fix at eight bits.
         *
         * ONE THREAD OWNS A GROUP, and k and v are done in SEPARATE PASSES so the thread holds QG
         * floats and not 2*QG -- at QG = 256 that is the difference between 1 KiB of scratch a
         * lane and 2. This runs once per stored token and its speed is irrelevant; a plain
         * sequential butterfly is used rather than r4d_fwht.h's, which wants a WAVE per block. */
        const uint32_t ng = kv_heads * head_dim / QG;
        for (uint32_t g = threadIdx.x; g < ng; g += kThreads) {
            const uint32_t j0 = g * QG, h = j0 / head_dim, d0 = j0 % head_dim;
            uint16_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
            const uint16_t* const src[2] = { k + (size_t)t * k_ld + j0,
                                             v + (size_t)t * v_ld + j0 };
            uint16_t* const dst[2] = { c + d0, c + head_dim + d0 };
#pragma unroll 1
            for (uint32_t which = 0; which < 2u; ++which) {
                /* QW: 0 none, 1 both, 2 K only, 3 V only. The asymmetry is the point -- the
                 * outlier structure a rotation exists to break up is a KEY phenomenon (channels
                 * that are consistently large after RoPE), and a value row has no such structure,
                 * so rotating V may only be mixing its own quantisation noise across channels. */
                constexpr bool rot_k = (QW == 1u || QW == 2u);
                constexpr bool rot_v = (QW == 1u || QW == 3u);
                const bool rot = which == 0u ? rot_k : rot_v;
                float b[QG];
                for (uint32_t e = 0; e < QG; ++e) b[e] = bf2f(src[which][e]);
                if constexpr (rot_k || rot_v) { if (rot) hadamard<QG>(b); }
                float amax = 0.0f;
                for (uint32_t e = 0; e < QG; ++e) amax = fmaxf(amax, fabsf(b[e]));
                /* QB is the integer width: 8 bits is +-127 and 4 is +-7. Four is where a rotation
                 * has the most to gain, because that is where one outlier really does cost every
                 * other element in the group its resolution. E4M3 ignores it. */
                const float lvl = (float)((1u << (QB - 1u)) - 1u);
                const float sc =
                    amax > 0.0f ? amax * (QF ? (1.0f / 448.0f) : (1.0f / lvl)) : 1.0f;
                for (uint32_t e = 0; e < QG; ++e) {
                    const float q = b[e] / sc;
                    if constexpr (QF) {
                        /* The clamp is the fp8 quantiser's own: the builtin answers an
                         * out-of-range input with a NaN rather than saturating, and amax/scale is
                         * 448 in exact arithmetic but can land an ulp above it. */
                        int pk = 0;
                        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(q, -448.0f), 448.0f),
                                                             0.0f, pk, false);
                        b[e] = r4d_fp8::e4m3((uint32_t)pk & 0xFFu) * sc;
                    } else {
                        b[e] = fminf(fmaxf(rintf(q), -lvl), lvl) * sc;
                    }
                }
                if constexpr (rot_k || rot_v) { if (rot) hadamard<QG>(b); }
                for (uint32_t e = 0; e < QG; ++e) dst[which][e] = f32_to_bf16(b[e]);
            }
        }
        return;
    }
    for (uint32_t j = threadIdx.x; j < kv_heads * head_dim; j += kThreads) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint16_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        c[d]            = k[(size_t)t * k_ld + j];
        c[head_dim + d] = v[(size_t)t * v_ld + j];
    }
}

// The same scatter into an E4M3 cache.  It is the write side of the fp8kv attention family and its
// `k_descale` / `v_descale` operands: without it a cache declared fp8 has no way to be filled.
//
// THE SCALE IS A CALLER CONSTANT AND IT HAS TO BE.  The reader's descale is indexed
// `[seq * kv_heads + kv_head]` (r4d.h), which is ONE number for a whole sequence -- so it cannot
// depend on the token being written, because tokens written at different steps share it.  vLLM has
// the same contract and defaults both scales to 1.0.  This takes them as arguments so a calibrated
// pair costs nothing later; at 1.0 the store is a plain E4M3 round of the bf16 the attention would
// otherwise have cached.
//
// FOUR COLUMNS A THREAD, for the reason quant_act_fp8 takes four: it is exactly the two
// v_cvt_pk_fp8_f32 that fill one dword, and the store is then a single dword rather than four
// byte writes.  head_dim is a multiple of 4 on every shape this serves and the entry point refuses
// anything else, so a head's 4-column group never straddles two heads and there is no ragged tail.
__global__ __launch_bounds__(kThreads) void kv_store_fp8_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld, float k_inv, float v_inv) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * 2 * head_dim;
    const size_t head_step = (size_t)block_size * 2 * head_dim;

    for (uint32_t j = threadIdx.x * 4u; j < kv_heads * head_dim; j += kThreads * 4u) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint8_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        float kf[4], vf[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kf[e] = bf2f(k[(size_t)t * k_ld + j + (uint32_t)e]) * k_inv;
            vf[e] = bf2f(v[(size_t)t * v_ld + j + (uint32_t)e]) * v_inv;
        }
        // Clamped because v_cvt_pk_fp8_f32 answers an out-of-range input with NaN, not a
        // saturate -- the same reason quant_act_fp8 clamps.
        int pk = 0, pv = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[0], -448.0f), 448.0f),
                                             fminf(fmaxf(kf[1], -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[2], -448.0f), 448.0f),
                                             fminf(fmaxf(kf[3], -448.0f), 448.0f), pk, true);
        pv = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vf[0], -448.0f), 448.0f),
                                             fminf(fmaxf(vf[1], -448.0f), 448.0f), pv, false);
        pv = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vf[2], -448.0f), 448.0f),
                                             fminf(fmaxf(vf[3], -448.0f), 448.0f), pv, true);
        *(uint32_t*)(c + d)            = (uint32_t)pk;
        *(uint32_t*)(c + head_dim + d) = (uint32_t)pv;
    }
}

// In-register wave-local FWHT-256 K4V4 store with INT4 packing
__global__ __launch_bounds__(kThreads) void kv_store_k4v4_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;

    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t n_waves = kThreads / kWave;

    for (uint32_t h = wave; h < kv_heads; h += n_waves) {
        const size_t byte_offset = (((size_t)blk * kv_heads + (size_t)h) * block_size + (size_t)off) * head_dim;
        uint8_t* __restrict__ c = kv + byte_offset;

        float kf[8];
        const uint16_t* k_src = k + (size_t)t * k_ld + (size_t)h * head_dim + (size_t)lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            kf[j] = bf2f(k_src[j]);
        }
        r4d_fwht_wave<256>(kf, (int)lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            kf[j] *= (1.0f / 16.0f);
        }

        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(kf[j]));
        amax = r4d_wave_max(amax);
        const float sc = amax > 0.0f ? amax * (1.0f / 7.0f) : 1.0f;
        const float inv_sc = 1.0f / sc;
        uint8_t qk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int q = (int)fminf(fmaxf(rintf(kf[j] * inv_sc), -7.0f), 7.0f);
            qk[j] = (uint8_t)(q & 0x0F);
        }
        uint8_t pk[4];
        pk[0] = (uint8_t)(qk[0] | (qk[1] << 4));
        pk[1] = (uint8_t)(qk[2] | (qk[3] << 4));
        pk[2] = (uint8_t)(qk[4] | (qk[5] << 4));
        pk[3] = (uint8_t)(qk[6] | (qk[7] << 4));

        float vf[8];
        const uint16_t* v_src = v + (size_t)t * v_ld + (size_t)h * head_dim + (size_t)lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            vf[j] = bf2f(v_src[j]);
        }
        float vamax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) vamax = fmaxf(vamax, fabsf(vf[j]));
        vamax = r4d_wave_max(vamax);
        const float vsc = vamax > 0.0f ? vamax * (1.0f / 7.0f) : 1.0f;
        const float inv_vsc = 1.0f / vsc;
        uint8_t qv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int q = (int)fminf(fmaxf(rintf(vf[j] * inv_vsc), -7.0f), 7.0f);
            qv[j] = (uint8_t)(q & 0x0F);
        }
        uint8_t pv[4];
        pv[0] = (uint8_t)(qv[0] | (qv[1] << 4));
        pv[1] = (uint8_t)(qv[2] | (qv[3] << 4));
        pv[2] = (uint8_t)(qv[4] | (qv[5] << 4));
        pv[3] = (uint8_t)(qv[6] | (qv[7] << 4));

        *(uint32_t*)(c + lane * 4) = *(const uint32_t*)pk;
        *(uint32_t*)(c + head_dim / 2 + lane * 4) = *(const uint32_t*)pv;
    }
}

// In-place wave-local FWHT-256 for queries with 1/16 scaling
__global__ __launch_bounds__(kThreads) void query_fwht256_kernel(
    uint16_t* __restrict__ q, int64_t total_heads) {
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t n_waves = kThreads / kWave;
    const int64_t head_idx = (int64_t)blockIdx.x * n_waves + wave;
    if (head_idx >= total_heads) return;

    uint16_t* __restrict__ q_head = q + head_idx * 256 + (int64_t)lane * 8;
    float qf[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        qf[j] = bf2f(q_head[j]);
    }
    r4d_fwht_wave<256>(qf, (int)lane);
    constexpr float inv_16 = 1.0f / 16.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        q_head[j] = f32_to_bf16(qf[j] * inv_16);
    }
}

}  // namespace

// ================================================================================ entry points
//
// Pitches are in ELEMENTS and 0 means "tight".  Every pointer is bf16 except the index arrays,
// which are i32 -- the core's derived buffers are all i32 (core/sched/batch.cpp refuses anything
// else), so there is no 64-bit index form to serve.

/* THE BODY TAKES A SINK; THE PUBLIC ENTRY IS A WRAPPER. `r4d.h`'s long-stream signature is
 * libr4d's driver API and callers outside this tree use it, so it stays exactly as it was -- but
 * the launch has to be describable, and writing the geometry twice is the one thing ABI v10's
 * contract says not to do. One body, two ways in. */
static int embed_lookup_bf16_emit(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                  int vocab_off, int x_ld, const R4dSink& sk) {
    if (!tok || !wte || !x) return -1;
    if (M <= 0 || n_embd <= 0 || n_vocab <= 0 || vocab_off < 0) return -2;
    if (!x_ld) x_ld = n_embd;
    return R4D_EMIT(sk, embed_lookup_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const int32_t*)tok, (const uint16_t*)wte,
                    (uint16_t*)x, (uint32_t)n_embd, (int32_t)n_vocab, (int32_t)vocab_off,
                    (uint32_t)x_ld) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_embed_lookup_bf16(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                     int vocab_off, int x_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return embed_lookup_bf16_emit(tok, wte, x, M, n_embd, n_vocab, vocab_off, x_ld, sk);
}

/* The quad walk when the width, the pitch and both bases allow its 4- and 8-byte accesses (a bf16
 * table's quad is 8 bytes), an entry a thread otherwise. */
static int embed_lookup_q_emit(const int32_t* tok, const void* wte, bool q, const void* sc,
                               bool sc_f32, uint16_t* x, long long M, int n_embd, int n_vocab,
                               int vocab_off, long long x_ld, const R4dSink& sk) {
    const bool quad = n_embd % 4 == 0 && x_ld % 4 == 0 && r4d_aligned(x, 8) &&
                      r4d_aligned(wte, q ? 4 : 8);

    if (q && quad && n_embd == 160 && M <= 64) {
        int dev = 0;
        hipPointerAttribute_t attr;
        if (hipPointerGetAttributes(&attr, (const void*)x) == hipSuccess) {
            dev = attr.device;
        } else {
            (void)hipGetDevice(&dev);
        }
        const PleVramCache cache = get_ple_vram_cache(dev, n_embd);
        const int rc = R4D_EMIT(sk, ple_embed_lookup_q_kernel, dim3((uint32_t)M), dim3(32), 0,
                                tok, wte, sc, (uint32_t)(sc_f32 ? 1u : 0u), x,
                                (uint32_t)M, (int32_t)n_vocab, (int32_t)vocab_off,
                                (uint32_t)x_ld, cache.tags, cache.heat, cache.data, cache.num_sets);
        return rc == RAD_OK ? 0 : -4;
    }

    const uint64_t threads = (uint64_t)M * (uint64_t)(quad ? n_embd / 4 : n_embd);
    const dim3 grid((uint32_t)((threads + kThreads - 1) / kThreads));

#define R4D_ELQ(V, Q)                                                                             \
    R4D_EMIT(sk, (embed_lookup_q_kernel<V, Q>), grid, dim3(kThreads), 0, tok, wte, sc,          \
             (uint32_t)(sc_f32 ? 1u : 0u), x, (uint32_t)M, (uint32_t)n_embd, (int32_t)n_vocab,   \
             (int32_t)vocab_off, (uint32_t)x_ld)
    const int rc = quad ? (q ? R4D_ELQ(4, true) : R4D_ELQ(4, false))
                        : (q ? R4D_ELQ(1, true) : R4D_ELQ(1, false));
#undef R4D_ELQ
    return rc == RAD_OK ? 0 : -4;
}

static int gather_rows_bf16_emit(long x, long idx, long y, int M, int n, int src_rows,
                                 int x_ld, int y_ld, const R4dSink& sk) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    return R4D_EMIT(sk, gather_rows_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)x, (const int32_t*)idx,
                    (uint16_t*)y, (uint32_t)n, (int32_t)src_rows,
                    (uint32_t)x_ld, (uint32_t)y_ld) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_bf16(long x, long idx, long y, int M, int n, int src_rows,
                                    int x_ld, int y_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gather_rows_bf16_emit(x, idx, y, M, n, src_rows, x_ld, y_ld, sk);
}


/* Byte rows. `n`, `x_ld` and `y_ld` are in BYTES at this boundary and are divided down to uchar4
 * inside, so a caller that thinks in a pool's byte strides never has to convert -- and the one
 * place the /4 happens is the one place it can be wrong. */
static int gather_rows_u8_emit(long x, long idx, long y, int M, int n, int src_rows,
                               int x_ld, int y_ld, const R4dSink& sk) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if ((n | x_ld | y_ld) & 3) return -2;        /* the registry constrains n; pitches too */
    return R4D_EMIT(sk, gather_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)x, (const int32_t*)idx, (uchar4*)y,
                    (uint32_t)(n / 4), (int32_t)src_rows,
                    (uint32_t)(x_ld / 4), (uint32_t)(y_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_u8(long x, long idx, long y, int M, int n, int src_rows,
                                  int x_ld, int y_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gather_rows_u8_emit(x, idx, y, M, n, src_rows, x_ld, y_ld, sk);
}

static int scatter_rows_u8_emit(long v, long idx, long x, int M, int n, int dst_rows,
                                int v_ld, int x_ld, const R4dSink& sk) {
    if (!v || !idx || !x) return -1;
    if (M <= 0 || n <= 0 || dst_rows <= 0) return -2;
    if (!v_ld) v_ld = n;
    if (!x_ld) x_ld = n;
    if ((n | v_ld | x_ld) & 3) return -2;
    return R4D_EMIT(sk, scatter_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)v, (const int32_t*)idx, (uchar4*)x,
                    (uint32_t)(n / 4), (int32_t)dst_rows,
                    (uint32_t)(v_ld / 4), (uint32_t)(x_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_scatter_rows_u8(long v, long idx, long x, int M, int n, int dst_rows,
                                   int v_ld, int x_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return scatter_rows_u8_emit(v, idx, x, M, n, dst_rows, v_ld, x_ld, sk);
}

// `w_f32` says whether the gain is f32 (every norm gain the architecture plugins declare) or bf16.
extern "C" int r4d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                int w_f32, int x_ld, int y_ld, long stream) {
    if (!x || !w || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    // ONE WORKGROUP A ROW, so the grid at a decode batch is a handful of workgroups on a 64-CU
    // part and the row is latency bound. r4d_act_threads is the measured crossover table the int8
    // activation kernels already use; r4d_rmsnorm_quant_fp8 calls the SAME function, and it has to,
    // because the width decides which columns a thread folds into the sum of squares. Two kernels
    // that disagree about it disagree in the last bits of the norm, and the fused one is required
    // to be byte-identical to this kernel followed by the quantiser.
    const int thr = r4d_act_threads(M, n);
#define R4D_RN_T(wf, T)                                                                         \
    hipLaunchKernelGGL((rmsnorm_kernel<wf, T>), dim3((uint32_t)M), dim3(T), 0,                  \
                       (hipStream_t)stream, (const uint16_t*)x, (const void*)w, (uint16_t*)y,   \
                       (uint32_t)n, eps, wadd, (uint32_t)x_ld, (uint32_t)y_ld)
#define R4D_RN(wf)                                                                              \
    do { if (thr == 1024)     R4D_RN_T(wf, 1024);                                               \
         else if (thr == 512) R4D_RN_T(wf, 512);                                                \
         else                 R4D_RN_T(wf, 256); } while (0)
    if (w_f32) R4D_RN(true);
    else       R4D_RN(false);
#undef R4D_RN
#undef R4D_RN_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `op` is 0 for a sum and 1 for a product.  `b_ld` of 0 with the same n is the broadcast form: the
// caller passes a b_ld of zero and every row reads the same n elements.
extern "C" int r4d_binary_bf16(long a, long b, long y, int M, int n, int op,
                               int a_ld, int b_ld, int y_ld, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!a_ld) a_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(binary_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, op);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(sigmoid_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_silu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream) {
    if (!gu || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(silu_mul_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)gu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `n_rope_heads` is n_head + n_head_kv: the Q heads and the K heads of a fused projection, or just
// the one buffer's heads when q and k are separate tensors (which is what the gated block does).
extern "C" int r4d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                             int rot, int gptj, float theta, float rscale, int row_pitch,
                             long stream) {
    return r4d_rope_mc_bf16(qkv, pos, M, head_dim, n_rope_heads, rot, gptj, theta, rscale,
                            row_pitch, 0, nullptr, 0, 0, stream);
}

// The multi-component form: `mode` 2 mrope, 3 imrope, 4 axial, `sec[0..n_sec)` the sections and
// `plane` the component pitch of `pos`; plane 0 is the one-component rotary above.
extern "C" int r4d_rope_mc_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                                int rot, int gptj, float theta, float rscale, int row_pitch,
                                int mode, const int* sec, int n_sec, int plane, long stream) {
    if (!qkv || !pos) return -1;
    if (M <= 0 || head_dim <= 0 || n_rope_heads <= 0) return -2;
    if (rot <= 0 || rot > head_dim || (rot & 1)) return -3;
    if (n_sec < 0 || n_sec > 3 || (n_sec && !sec)) return -3;
    if (!row_pitch) row_pitch = n_rope_heads * head_dim;
    const uint32_t pairs = (uint32_t)n_rope_heads * (uint32_t)(rot / 2);
    const uint32_t gy = (pairs + kThreads - 1) / kThreads;
    const int s0 = n_sec > 0 ? sec[0] : 0, s1 = n_sec > 1 ? sec[1] : 0, s2 = n_sec > 2 ? sec[2] : 0;
    hipLaunchKernelGGL(rope_kernel, dim3((uint32_t)M, gy), dim3(kThreads), 0, (hipStream_t)stream,
                       (uint16_t*)qkv, (const int32_t*)pos, (uint32_t)head_dim,
                       (uint32_t)n_rope_heads, (uint32_t)rot, gptj, theta, rscale,
                       (uint32_t)row_pitch, mode, s0, s1, s2, n_sec, (uint32_t)plane);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_kv_store_bf16(long k, long v, long slot, long kv, int M, int head_dim,
                                 int kv_heads, int block_size, int n_blocks,
                                 int k_ld, int v_ld, long stream) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    /* NO QUANTISATION-SIMULATION BRANCH ON THE SERVING PATH. The probe above is compiled only when
     * its template parameters are set; this launch pins them off. A simulation that stays in the
     * store path is four branches and a macro tree between every token and its cache, for a
     * question that is answered once.
     *
     * The real fp8 cache is a separate entry point (r4d_kv_store_fp8, below) selected by
     * --kv-cache-dtype, not a mode of this one. */
    hipLaunchKernelGGL((kv_store_kernel<0u, 0u, 0u, 8u>), dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                       (uint16_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                       (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// bf16 k and v in, E4M3 cache out.  `k_scale` and `v_scale` are what the READER will multiply back
// by -- they are the descale, so this divides -- and 1.0 is the vLLM default and the value the
// attention kernels assume when their descale pointers are null.
static int kv_store_fp8_emit(long k, long v, long slot, long kv, int M, int head_dim,
                             int kv_heads, int block_size, int n_blocks,
                             int k_ld, int v_ld, float k_scale, float v_scale,
                             const R4dSink& sk) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    // Four columns a thread is the packing, so a head must be a whole number of groups; otherwise
    // one group would straddle two heads and write the next head's first bytes.
    if (head_dim % 4) return -3;
    if (!(k_scale > 0.0f) || !(v_scale > 0.0f)) return -5;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    return R4D_EMIT(sk, kv_store_fp8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                    (uint8_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                    (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld,
                    1.0f / k_scale, 1.0f / v_scale) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_kv_store_fp8(long k, long v, long slot, long kv, int M, int head_dim,
                                int kv_heads, int block_size, int n_blocks,
                                int k_ld, int v_ld, float k_scale, float v_scale, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return kv_store_fp8_emit(k, v, slot, kv, M, head_dim, kv_heads, block_size, n_blocks,
                             k_ld, v_ld, k_scale, v_scale, sk);
}

static int kv_store_k4v4_emit(long k, long v, long slot, long kv, int M, int head_dim,
                              int kv_heads, int block_size, int n_blocks, int k_ld, int v_ld,
                              const R4dSink& sk) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    if (head_dim != 256) return -3;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    return R4D_EMIT(sk, kv_store_k4v4_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                    (uint8_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                    (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_kv_store_k4v4(long k, long v, long slot, long kv, int M, int head_dim,
                                 int kv_heads, int block_size, int n_blocks,
                                 int k_ld, int v_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return kv_store_k4v4_emit(k, v, slot, kv, M, head_dim, kv_heads, block_size, n_blocks,
                              k_ld, v_ld, sk);
}

static int query_fwht256_emit(long q, int64_t total_heads, const R4dSink& sk) {
    if (!q || total_heads <= 0) return -1;
    constexpr uint32_t n_waves = kThreads / kWave;
    const uint32_t grid = (uint32_t)((total_heads + n_waves - 1) / n_waves);
    return R4D_EMIT(sk, query_fwht256_kernel, dim3(grid), dim3(kThreads), 0,
                    (uint16_t*)q, total_heads) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_query_fwht256(long q, int64_t total_heads, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return query_fwht256_emit(q, total_heads, sk);
}

// ================================================================================ logits_gemm
//
// THE OUTPUT IS f32 AND THAT IS THE WHOLE REASON THIS KERNEL EXISTS.  `logits_gemm` is the same
// arithmetic as r4d_gemm_bf16_nt_m16 under another parameter spelling, so dispatching it there
// looks correct -- but that kernel writes `__hip_bfloat16* C`, and the engine declares its logits
// buffer f32, because a bf16 logit has eight bits of mantissa and a nucleus cut is a comparison of
// neighbouring probabilities.  The projection would write bf16 pairs into an f32 plane: every
// logit wrong, the magnitudes roughly right, and fluent nonsense out of the sampler.  rad-kbench
// catches it as rel_l2 1.2 against the oracle, but only where it has a device domain to run in.
//
// One workgroup a vocabulary column, 256 threads striding K.  At n_vocab 248320 and K 5120 that is
// a 2.5 GiB weight read a step and the read is the cost; the arithmetic is a v_dot2 per pair.  It
// is not a tuned kernel and does not pretend to be -- there is one variant, it is correct, and the
// tuning axis is open.
//
// Rows come in chunks of kLgRows so the accumulators stay in registers whatever M is.  M here is
// one row per SAMPLED position -- `gather_rows` has already reduced a prefill chunk to one row a
// sequence -- so a chunk is almost always the whole of it.

namespace {

constexpr uint32_t kLgRows = 8;    // rows held in registers at once
constexpr uint32_t kLgVec  = 8;    // bf16 per 128-bit load

__global__ __launch_bounds__(kThreads) void logits_gemm_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ w, float* __restrict__ c,
    uint32_t M, uint32_t K, uint32_t N, uint32_t a_ld, uint32_t c_ld) {
    __shared__ float lds[kWaves][2][kLgRows];
    const uint32_t j0 = blockIdx.x * 2;
    const uint32_t j1 = j0 + 1;
    if (j0 >= N) return;
    const uint16_t* __restrict__ wj0 = w + (size_t)j0 * K;
    const uint16_t* __restrict__ wj1 = (j1 < N) ? (w + (size_t)j1 * K) : nullptr;

    for (uint32_t r0 = 0; r0 < M; r0 += kLgRows) {
        const uint32_t nr = (M - r0) < kLgRows ? (M - r0) : kLgRows;
        float acc0[kLgRows];
        float acc1[kLgRows];
#pragma unroll
        for (uint32_t i = 0; i < kLgRows; ++i) {
            acc0[i] = 0.0f;
            acc1[i] = 0.0f;
        }

        for (uint32_t k = threadIdx.x * kLgVec; k < K; k += kThreads * kLgVec) {
            uint4 w0_raw = *reinterpret_cast<const uint4*>(wj0 + k);
            const uint16_t* wv0 = reinterpret_cast<const uint16_t*>(&w0_raw);
            uint4 w1_raw = (wj1 != nullptr) ? *reinterpret_cast<const uint4*>(wj1 + k) : uint4{0, 0, 0, 0};
            const uint16_t* wv1 = reinterpret_cast<const uint16_t*>(&w1_raw);

            for (uint32_t i = 0; i < nr; ++i) {
                const uint16_t* __restrict__ ar = a + (size_t)(r0 + i) * a_ld + k;
                uint4 a_raw = *reinterpret_cast<const uint4*>(ar);
                const uint16_t* av = reinterpret_cast<const uint16_t*>(&a_raw);
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (uint32_t p = 0; p < kLgVec; ++p) {
                    const float af = bf2f(av[p]);
                    s0 += af * bf2f(wv0[p]);
                    if (wj1) s1 += af * bf2f(wv1[p]);
                }
                acc0[i] += s0;
                acc1[i] += s1;
            }
        }

        for (uint32_t i = 0; i < nr; ++i) {
            float v0 = r4d_wave_sum(acc0[i]);
            float v1 = r4d_wave_sum(acc1[i]);
            const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
            if (lane == 0) {
                lds[wave][0][i] = v0;
                lds[wave][1][i] = v1;
            }
        }
        __syncthreads();
        if (threadIdx.x < nr) {
            float v0 = 0.0f;
            float v1 = 0.0f;
#pragma unroll
            for (uint32_t q = 0; q < kWaves; ++q) {
                v0 += lds[q][0][threadIdx.x];
                v1 += lds[q][1][threadIdx.x];
            }
            c[(size_t)(r0 + threadIdx.x) * c_ld + j0] = v0;
            if (j1 < N) c[(size_t)(r0 + threadIdx.x) * c_ld + j1] = v1;
        }
        __syncthreads();
    }
}

}  // namespace

//   a  [M][K]       bf16, `a_ld` elements a row (0 = K)
//   w  [N][K]       bf16, tight
//   c  [M][N]       f32,  `c_ld` elements a row (0 = N)
extern "C" int r4d_logits_gemm_bf16(long a, long w, long c, int M, int K, int N,
                                    int a_ld, int c_ld, long stream) {
    if (!a || !w || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % (int)kLgVec) return -3;      // a thread owns eight columns; see the kernel
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    const uint32_t grid_x = ((uint32_t)N + 1) / 2;
    hipLaunchKernelGGL(logits_gemm_kernel, dim3(grid_x), dim3(kThreads), 0,
                       (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)w, (float*)c,
                       (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)c_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// Unpacking RadArgs and re-deriving the row pitches IS this kernel's own business, so it lives
// here rather than in a translation layer an out-of-tree author cannot write.  See
// libr4d/r4d_args.h.
//
// WHAT THESE CHECK, and why here and not deeper in.  The kernels take raw pointers and one
// pitch, so anything they do not pitch has to be tight -- the "necessary condition only"
// boundary of spec 2.1.  Every op below therefore establishes, before the launch:
//
//   * the dtype.  These kernels are bf16 and read through a uint16_t; an f32 operand would be
//     read as two bf16 halves and the output would be plausible garbage.  The rows constrain
//     `dtype`, but a constraint is a NECESSARY condition on the geometry and not a promise
//     about the tensors, so the operand's own dtype is what is tested.
//   * the row pitch.  r4d_rows_pitch() gives it in elements, or -1 when an inner axis is
//     strided, and -1 is RAD_E_STRIDE rather than a silently wrong walk.
//   * the row count.  It comes from the OPERAND and never from the `M` parameter: M is RANGED,
//     so the value in the args is the band's upper bound and a loop bounded by it walks off
//     the arena (libref/ref_ops.h makes the same point at more length).
/* bf16 is the only activation width these kernels read. Stated once. */
static inline int dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

/* Opaque bytes. The tier traffic moves KV fragments whose contents are whatever the attention
 * kernel wrote, so u8 is the dtype that describes them without claiming to interpret them. */
static inline int dt_u8(const RadTensor* t) {
    return (t && (t->dtype == RAD_U8 || t->dtype == RAD_I8)) ? RAD_OK : RAD_E_DTYPE;
}

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

/* ------------------------------------------------------------------ embed_lookup */

enum { EL_TOK = 0, EL_WTE, EL_X };

static int rad_embed_lookup_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* tok = r4d_opt(a, EL_TOK);
    const RadTensor* wte = r4d_opt(a, EL_WTE);
    const RadTensor* x   = r4d_opt(a, EL_X);
    if (!tok || !wte || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    R4D_NEED(dt_bf16(wte));
    R4D_NEED(dt_bf16(x));
    if (!r4d_contig(tok) || !r4d_contig(wte)) return R4D_NO(RAD_E_STRIDE);

    const long long n_embd = r4d_geti_or(a, "n_embd", r4d_dim(wte, (int)wte->rank - 1));
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(wte, (int)wte->rank - 2));
    const long long voff = r4d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wte) < n_vocab * n_embd) return R4D_NO(RAD_E_SHAPE);

    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);

    return embed_lookup_bf16_emit(r4d_addr(tok), r4d_addr(wte), r4d_addr(x),
                                  (int)M, (int)n_embd, (int)n_vocab, (int)voff, (int)x_ld,
                                  sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_embed_lookup, rad_embed_lookup_body)

/* ------------------------------------------------------------------ embed_lookup_q */

/* Operand 4, `ahead`, is not read. It carries the ids of the prompt after this step for a host
 * reader that fetches table rows from a file before they are wanted; a card reads its table where
 * it already is, so there is nothing to fetch. */
enum { ELQ_TOK = 0, ELQ_WTE, ELQ_SCALE, ELQ_X };

static int rad_embed_lookup_q_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* tok = r4d_opt(a, ELQ_TOK);
    const RadTensor* wte = r4d_opt(a, ELQ_WTE);
    const RadTensor* sc  = r4d_opt(a, ELQ_SCALE);
    const RadTensor* x   = r4d_opt(a, ELQ_X);
    if (!tok || !wte || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    R4D_NEED(dt_bf16(x));
    const bool q = wte->dtype == RAD_F8E4M3;
    if (!q && wte->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (sc && sc->dtype != RAD_BF16 && sc->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (sc && r4d_numel(sc) < 1) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(tok) || !r4d_contig(wte)) return R4D_NO(RAD_E_STRIDE);

    const long long n_embd = r4d_geti_or(a, "n_embd", r4d_dim(wte, (int)wte->rank - 1));
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(wte, (int)wte->rank - 2));
    const long long voff = r4d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wte) < n_vocab * n_embd) return R4D_NO(RAD_E_SHAPE);

    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);

    return embed_lookup_q_emit((const int32_t*)r4d_addr(tok), (const void*)r4d_addr(wte), q,
                               sc ? (const void*)r4d_addr(sc) : nullptr,
                               sc && sc->dtype == RAD_F32, (uint16_t*)r4d_addr(x), M,
                               (int)n_embd, (int)n_vocab, (int)voff, x_ld, sk) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_embed_lookup_q, rad_embed_lookup_q_body)

/* ------------------------------------------------------------------ gather_rows */

enum { GR_X = 0, GR_IDX, GR_Y };

static int rad_gather_rows_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x   = r4d_opt(a, GR_X);
    const RadTensor* idx = r4d_opt(a, GR_IDX);
    const RadTensor* y   = r4d_opt(a, GR_Y);
    if (!x || !idx || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_bf16(y));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, src = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &src);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || src <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return gather_rows_bf16_emit(r4d_addr(x), r4d_addr(idx), r4d_addr(y),
                                 (int)M, (int)n, (int)src, (int)x_ld, (int)y_ld,
                                 sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_gather_rows, rad_gather_rows_body)


/* ------------------------------------------------------------------ byte rows */
/* The u8 forms. `n` here is a COUNT OF BYTES, because at dtype u8 an element is a byte -- so the
 * pitches r4d_rows_pitch returns are byte pitches and go straight through. */
static int rad_gather_rows_u8_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x   = r4d_opt(a, GR_X);
    const RadTensor* idx = r4d_opt(a, GR_IDX);
    const RadTensor* y   = r4d_opt(a, GR_Y);
    if (!x || !idx || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_u8(x));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_u8(y));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, src = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &src);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || src <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return gather_rows_u8_emit(r4d_addr(x), r4d_addr(idx), r4d_addr(y),
                               (int)M, (int)n, (int)src, (int)x_ld, (int)y_ld,
                               sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_gather_rows_u8, rad_gather_rows_u8_body)

/* The operand order is (v, idx, x) and x is INOUT: the destination is the pool and the rows this
 * call does not name stay as they were. */
enum { SR_V = 0, SR_IDX, SR_X };

static int rad_scatter_rows_u8_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* v   = r4d_opt(a, SR_V);
    const RadTensor* idx = r4d_opt(a, SR_IDX);
    const RadTensor* x   = r4d_opt(a, SR_X);
    if (!v || !idx || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_u8(v));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_u8(x));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(v, (int)v->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, dst = 0;
    const long long v_ld = r4d_rows_pitch(v, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &dst);
    if (v_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || dst <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return scatter_rows_u8_emit(r4d_addr(v), r4d_addr(idx), r4d_addr(x),
                                (int)M, (int)n, (int)dst, (int)v_ld, (int)x_ld,
                                sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_scatter_rows_u8, rad_scatter_rows_u8_body)

/* THE SAME COPY, for rows a caller declared bf16: the kernel moves bytes, so a bf16 row of n is
 * 2n bytes and nothing else about it changes. An encoder's rows over a language model's
 * embeddings are the case -- both sides are activations, declared bf16. */
static int rad_scatter_rows_bf16_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* v   = r4d_opt(a, SR_V);
    const RadTensor* idx = r4d_opt(a, SR_IDX);
    const RadTensor* x   = r4d_opt(a, SR_X);
    if (!v || !idx || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(v));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_bf16(x));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(v, (int)v->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, dst = 0;
    const long long v_ld = r4d_rows_pitch(v, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &dst);
    if (v_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || dst <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return scatter_rows_u8_emit(r4d_addr(v), r4d_addr(idx), r4d_addr(x),
                                (int)M, (int)(2 * n), (int)dst, (int)(2 * v_ld), (int)(2 * x_ld),
                                sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_scatter_rows_bf16, rad_scatter_rows_bf16_body)

/* ------------------------------------------------------------------ cast (bf16 -> bf16) */
/* A STRIDED ROW COPY, which is what `cast` degenerates to when the two ends are the same dtype,
 * and the reason there is a device row for it at all: an architecture that conditions a drafter on
 * the trunk's INTERMEDIATE hidden states has to move rows out of the residual stream before the
 * next layer overwrites it, and the only other implementation is ref's -- on the host, which the
 * reference gate refuses in a serve.
 *
 * libr4d's 2D copy takes byte pitches, so a column slice of a wider buffer is expressible and no
 * kernel has to learn about the layout: this is exactly how five taps land side by side in one
 * [T, 5 * n_embd] row. */
enum { CS_X = 0, CS_Y };

extern "C" int r4d_rad_cast(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, CS_X);
    const RadTensor* y = r4d_opt(a, CS_Y);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* The ROW COUNT is the destination's. A tap writes a column slice of a buffer sized for the
     * step's ceiling while the source is the live rows, so the two extents differ by design; what
     * must hold is that the source has the rows the destination asks for. */
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_p2p_copy2d(r4d_addr(y), r4d_addr(x), y_ld * 2, x_ld * 2, n * 2, M,
                          (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rmsnorm */

enum { RN_X = 0, RN_W, RN_Y };

extern "C" int r4d_rad_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, RN_X);
    const RadTensor* w = r4d_opt(a, RN_W);
    const RadTensor* y = r4d_opt(a, RN_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    /* THE GAIN IS THE ONE OPERAND THAT IS NOT bf16. Every architecture plugin here declares its
     * norm gains RAD_F32 -- a few thousand elements against gigabytes of weights, and the norm is
     * where a rounded gain shows -- so the width is read off the operand and handed to the kernel
     * rather than assumed. Assuming it would read an f32 gain as two bf16 halves of the wrong
     * number, which is the silent kind of wrong. */
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const int w_f32 = w->dtype == RAD_F32;

    const long long n = r4d_geti_or(a, "n", r4d_dim(w, (int)w->rank - 1));
    if (n <= 0 || r4d_numel(w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    return r4d_rmsnorm_bf16(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n, eps, wadd,
                            w_f32, (int)x_ld, (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                                 : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ add / mul */

enum { BI_A = 0, BI_B, BI_Y };

static int binary(const RadArgs* a, int mul, RadStream s) {
    const RadTensor* A = r4d_opt(a, BI_A);
    const RadTensor* B = r4d_opt(a, BI_B);
    const RadTensor* Y = r4d_opt(a, BI_Y);
    if (!A || !B || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(A));
    R4D_NEED(dt_bf16(B));
    R4D_NEED(dt_bf16(Y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long a_ld = r4d_rows_pitch(A, n, &Ma);
    if (y_ld < 0 || a_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M) return R4D_NO(RAD_E_SHAPE);

    /* THE ONE BROADCAST RULE: an operand carrying exactly n elements against an output of M rows
     * broadcasts along rows, and nothing else broadcasts at all. A pitch of 0 is how the kernel is
     * told. Anything in between -- more than n, fewer than M*n -- is a shape bug and is refused
     * rather than made plausible. */
    long long b_ld = 0;
    if (r4d_numel(B) == n && M > 1) {
        if (!r4d_contig(B)) return R4D_NO(RAD_E_STRIDE);
    } else {
        long long Mb = 0;
        b_ld = r4d_rows_pitch(B, n, &Mb);
        if (b_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (Mb < M) return R4D_NO(RAD_E_SHAPE);
        if (b_ld == 0) return R4D_NO(RAD_E_STRIDE);   /* 0 means broadcast; a real pitch cannot be 0 */
    }

    return r4d_binary_bf16(r4d_addr(A), r4d_addr(B), r4d_addr(Y), (int)M, (int)n, mul,
                           (int)a_ld, (int)b_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_add(const RadArgs* a, RadStream s) { return binary(a, 0, s); }
extern "C" int r4d_rad_mul(const RadArgs* a, RadStream s) { return binary(a, 1, s); }

/* ------------------------------------------------------------------ sigmoid */

extern "C" int r4d_rad_sigmoid(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    /* ROW-WISE, WITH A PITCH, because the input is usually a COLUMN SLICE. A gated attention block
     * issues this against the second half of each head of an interleaved [tokens, heads,
     * 2*head_dim] projection: `n` wide inside a stride of 2n, with no contiguous extent anywhere.
     * Treating it as a flat elementwise op refused it outright. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_sigmoid_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                            (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ silu_mul */

extern "C" int r4d_rad_silu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = r4d_opt(a, 0);
    const RadTensor* y  = r4d_opt(a, 1);
    if (!gu || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(gu));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0;
    const long long y_ld  = r4d_rows_pitch(y, n, &M);
    const long long gu_ld = r4d_rows_pitch(gu, 2 * n, &Mg);
    if (y_ld < 0 || gu_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mg < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_silu_mul_bf16(r4d_addr(gu), r4d_addr(y), (int)M, (int)n, (int)gu_ld, (int)y_ld,
                             (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rope */

enum { RP_QKV = 0, RP_POS };

extern "C" int r4d_rad_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r4d_opt(a, RP_QKV);
    const RadTensor* pos = r4d_opt(a, RP_POS);
    if (!qkv || !pos) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(qkv));
    R4D_NEED(dt_i32(pos));
    if (!r4d_contig(pos)) return R4D_NO(RAD_E_STRIDE);

    const long long head_dim  = r4d_geti_or(a, "head_dim", 0);
    const long long n_head    = r4d_geti_or(a, "n_head", 0);
    const long long n_head_kv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot       = r4d_geti_or(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0) return R4D_NO(RAD_E_SHAPE);
    if (rot <= 0 || rot > head_dim || (rot & 1)) return R4D_NO(RAD_E_SHAPE);

    /* THE MODE, and with it how many components a position has. With [M] positions -- one
     * component a token -- mrope and imrope ARE neox: every frequency reads the same position and
     * the ladder is the same. With [C, M] positions and `sections`, each frequency reads the
     * component its section names (docs/OPS.md, `rope`). */
    const char* mode = r4d_gets(a, "mode");
    int gptj = 0, mc = 0;
    if (mode) {
        if (!std::strcmp(mode, "gptj")) gptj = 1;
        else if (!std::strcmp(mode, "mrope")) mc = 2;
        else if (!std::strcmp(mode, "imrope")) mc = 3;
        else if (!std::strcmp(mode, "axial")) mc = 4;
        else if (std::strcmp(mode, "neox")) return R4D_NO(RAD_E_SCHEMA);
    }
    int sec[3] = { 0, 0, 0 }, n_sec = 0;
    for (const char* v = r4d_gets(a, "sections"); v && *v;) {
        char* e = nullptr;
        const long x = std::strtol(v, &e, 10);
        if (e == v) break;
        if (x <= 0 || n_sec >= 3) return R4D_NO(RAD_E_SHAPE);
        sec[n_sec++] = (int)x;
        v = e;
    }

    const long long row = (n_head + 2 * n_head_kv) * head_dim;
    long long M = 0;
    const long long pitch = r4d_rows_pitch(qkv, row, &M);
    if (pitch < 0) return R4D_NO(RAD_E_STRIDE);
    const bool multi = pos->rank >= 2;
    const long long Mp = multi ? pos->shape[1] : r4d_numel(pos);
    if (Mp <= 0 || M < Mp) return R4D_NO(RAD_E_SHAPE);
    int plane = 0;
    if (multi) {
        if (!mc || n_sec == 0) return R4D_NO(RAD_E_SHAPE);
        if (pos->shape[0] < (mc == 3 ? 3 : n_sec)) return R4D_NO(RAD_E_SHAPE);
        if (mc == 3 && n_sec != 3) return R4D_NO(RAD_E_SHAPE);
        if (mc != 3 && [&] { int t = 0; for (int c = 0; c < n_sec; ++c) t += sec[c];
                             return t != (int)(rot / 2); }())
            return R4D_NO(RAD_E_SHAPE);
        plane = (int)pos->shape[1];
    }

    const float theta  = (float)r4d_getf(a, "theta", 10000.0);
    const float rscale = (float)r4d_getf(a, "scale", 1.0);
    /* The Q heads and the K heads rotate; the V heads that follow them do not, which is the whole
     * reason this op takes the fused projection rather than two tensors. */
    return r4d_rope_mc_bf16(r4d_addr(qkv), r4d_addr(pos), (int)Mp, (int)head_dim,
                            (int)(n_head + n_head_kv), (int)rot, gptj, theta, rscale, (int)pitch,
                            mc, sec, n_sec, plane, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                            : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ kv_store */

enum { KS_K = 0, KS_V, KS_SLOT, KS_CACHE };

/* The geometry both stores share. The ONLY difference between them is the cache's dtype and what
 * the kernel does with a value on the way in, so everything up to the launch is one function --
 * two copies of a paged-layout check is how the two drift into disagreeing about the layout. */
struct KvStoreGeom {
    const RadTensor *k, *v, *sm, *kv;
    long long head_dim, block_size, kv_heads, n_blocks, k_ld, v_ld, T;
};

static int kv_store_geom(const RadArgs* a, KvStoreGeom* g) {
    g->k  = r4d_opt(a, KS_K);
    g->v  = r4d_opt(a, KS_V);
    g->sm = r4d_opt(a, KS_SLOT);
    g->kv = r4d_opt(a, KS_CACHE);
    if (!g->k || !g->v || !g->sm || !g->kv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(g->k));
    R4D_NEED(dt_bf16(g->v));
    R4D_NEED(dt_i32(g->sm));
    if (!r4d_contig(g->sm) || !r4d_contig(g->kv)) return R4D_NO(RAD_E_STRIDE);

    g->head_dim   = r4d_geti_or(a, "head_dim", r4d_dim(g->k, (int)g->k->rank - 1));
    g->block_size = r4d_geti_or(a, "block_size", 0);
    g->kv_heads   = r4d_geti_or(a, "n_head_kv", g->kv->rank == 4 ? g->kv->shape[1] : 0);
    if (g->head_dim <= 0 || g->block_size <= 0 || g->kv_heads <= 0) return R4D_NO(RAD_E_SHAPE);
    /* The paged layout is [n_blocks, kv_heads, block_size, 2*head_dim] and every kernel in this
     * project shares it. A cache that disagrees is refused rather than walked with the parameters'
     * idea of the shape. */
    if (g->kv->rank != 4) return R4D_NO(RAD_E_SHAPE);
    if (g->kv->shape[1] != g->kv_heads || g->kv->shape[2] != g->block_size ||
        g->kv->shape[3] != 2 * g->head_dim) return R4D_NO(RAD_E_SHAPE);
    g->n_blocks = g->kv->shape[0];

    const long long lane = g->kv_heads * g->head_dim;
    long long M = 0, Mv = 0;
    g->k_ld = r4d_rows_pitch(g->k, lane, &M);
    g->v_ld = r4d_rows_pitch(g->v, lane, &Mv);
    if (g->k_ld < 0 || g->v_ld < 0) return R4D_NO(RAD_E_STRIDE);
    g->T = r4d_numel(g->sm);
    if (g->T <= 0 || M < g->T || Mv < g->T) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int r4d_rad_kv_store(const RadArgs* a, RadStream s) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    R4D_NEED(dt_bf16(g.kv));

    return r4d_kv_store_bf16(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                             (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                             (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld,
                             (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* The E4M3 twin. `k_scale` and `v_scale` are the DESCALES the attention side multiplies back by --
 * one number per (sequence, head) there, so a constant here -- and 1.0 is both vLLM's default and
 * what the fp8kv kernels assume when their descale operands are absent. */
static int rad_kv_store_fp8_body(const RadArgs* a, const R4dSink& sk) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    /* Four columns a thread is the packing the kernel uses, so a head must be a whole number of
     * groups; otherwise one group straddles two heads and writes the next head's first bytes. */
    if (g.head_dim % 4) return R4D_NO(RAD_E_SHAPE);

    const double ks = r4d_getf(a, "k_scale", 1.0);
    const double vs = r4d_getf(a, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);

    return kv_store_fp8_emit(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                             (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                             (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld,
                             (float)ks, (float)vs, sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_kv_store_fp8, rad_kv_store_fp8_body)

static int rad_kv_store_k4v4_body(const RadArgs* a, const R4dSink& sk) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_I4) return R4D_NO(RAD_E_DTYPE);
    if (g.head_dim != 256) return R4D_NO(RAD_E_SHAPE);

    return kv_store_k4v4_emit(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                              (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                              (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld, sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_kv_store_k4v4, rad_kv_store_k4v4_body)

static int rad_query_fwht256_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* q = r4d_opt(a, 0);
    if (!q) return R4D_NO(RAD_E_INVAL);
    const int64_t total_elements = r4d_numel(q);
    if (total_elements <= 0 || total_elements % 256 != 0) return R4D_NO(RAD_E_SHAPE);
    const int64_t total_heads = total_elements / 256;
    return query_fwht256_emit(r4d_addr(q), total_heads, sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_query_fwht256, rad_query_fwht256_body)

/* ------------------------------------------------------------------ logits_gemm
 *
 * ITS OWN KERNEL, AND THE OUTPUT DTYPE IS WHY. `logits_gemm` is `gemm_nt`'s arithmetic under the
 * vocabulary edge's parameter spelling -- `n_vocab` and `n_embd` where gemm_nt says N and K -- so
 * the skinny bf16 GEMM looks like the right target for it. That GEMM writes bf16. The engine
 * declares its logits buffer f32, because a bf16 logit has eight bits of mantissa and a nucleus
 * cut is a comparison of neighbouring probabilities, so that dispatch writes bf16 pairs into an
 * f32 plane: every logit wrong, the magnitudes roughly right, fluent nonsense out of the sampler.
 *
 * M is one row per SAMPLED position, not per token: `gather_rows` runs first and reduces a prefill
 * chunk to one row a sequence. At max_tok 8192 against a 248K vocabulary the ungathered logits
 * buffer alone would be 8 GiB, which is what that op is for.
 *
 */

enum { LG_X = 0, LG_W, LG_Y };

extern "C" int r4d_rad_logits_gemm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(w));
    /* f32, and checked rather than assumed: the bf16 GEMM this op resembles writes bf16 into
     * exactly this buffer. See the note above. */
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);

    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", r4d_dim(w, 1));
    if (n_vocab <= 0 || n_embd <= 0) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd)
        return R4D_NO(RAD_E_SHAPE);

    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);

    return r4d_logits_gemm_bf16(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd,
                                (int)n_vocab, (int)x_ld, (int)y_ld,
                                (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}
