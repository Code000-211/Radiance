// The small bf16 ops the engine issues between the GEMMs: embedding gathers, norms, elementwise
// ops, rope, KV stores. Every op in a step needs a device launch, because the engine's buffers
// live in VRAM and `rad_issue` refuses an op with no launch for its operands' domain.
//
// The arithmetic is libref's (the oracle, spec §17); each kernel names the ref function it must
// equal, rounding in the same places:
//
//   * every reduction accumulates in f32 and only the store narrows to bf16;
//   * `f32_to_bf16` is round-to-nearest-even (r4d_common.h), as in ref;
//   * a negative index (token id, gather row, KV slot) is padding and writes a zero row or skips
//     the store. It is not clamped to zero, which would duplicate token 0 across padding.
//
// One divergence from ref: ref checks index arrays on the host and refuses out-of-range values. A
// kernel cannot without a copy and sync on the step path, so the bound is an argument and an
// out-of-range index is treated like a negative one (zero row or no store) instead of faulting.
//
// `expf`, `powf` and `sincosf` are the accurate forms, not the `__` approximations: rope positions
// reach 262144, beyond __sincosf's argument reduction, and libref uses libm. None is on a hot path.
//
// No fusion: fused forms such as `rmsnorm_add` are separate entry points the architecture plugin
// selects (spec §2.3).
//
// Entry points return a negative code rather than throwing, as in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "rad_yarn.h"
#include "r4d_fp8_wmma.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

// Sum across the block, broadcast to every thread: one LDS round trip after the wave reduction.
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
// ref_embed_lookup. One block per token; a negative id, or one at or above n_vocab, writes a zero
// row.
//
// `vocab_off` is the first global row of this rank's table, making the embedding vocab-parallel:
// ids stay global, a rank subtracts the offset, and another rank's token takes the zero-row path.
// Summing the ranks reconstructs x. `mult` scales every entry with one rounding (Gemma's
// sqrt(hidden)); 1 is a plain copy.
__global__ __launch_bounds__(kThreads) void embed_lookup_kernel(
    const int32_t* __restrict__ tok, const uint16_t* __restrict__ wte, uint16_t* __restrict__ x,
    uint32_t n_embd, int32_t n_vocab, int32_t vocab_off, uint32_t x_ld, float mult) {
    const uint32_t m = blockIdx.x;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld;
    if (id < 0 || id >= n_vocab) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = 0;
        return;
    }
    const uint16_t* __restrict__ src = wte + (size_t)id * n_embd;
    if (mult == 1.0f) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = src[i];
        return;
    }
    for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = f32_to_bf16(bf16_to_f32(src[i]) * mult);
}

// ---------------------------------------------------------------------------- embed_lookup_q
// ref_embed_lookup_q. x[m] = bf16(wte[id] * scale), from a table of E4M3 codes with one scale (Q)
// or plain bf16; an absent scale is one. This is the n-gram table's gather
// (arch/common/rad_block_ple.h) when --ngram-placement lets a card read the table, in VRAM or in
// pinned host memory over the link.
//
// One thread per quad of entries: an n-gram row is only 160 entries, so a block per row would
// leave most threads idle. Consecutive threads take consecutive quads. V = 1 walks one entry at a
// time, for widths or pitches the quad's 4- and 8-byte accesses would misalign.
//
// Bit-identical to libavx's lookup table: a code decodes to f32 exactly (v_cvt_pk_f32_fp8 or the
// software form in r4d_fp8_wmma.h), the scale product rounds once in f32, and f32_to_bf16 rounds
// to nearest even. An id outside [0, n_vocab) after the offset writes a zero row.
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

// ------------------------------------------------------------------ the PLE row cache
//
// The n-gram table (arch/common/rad_block_ple.h) is 51.2 GB of E4M3 rows, and with
// --ngram-placement ram it lives in pinned host memory: every PLE gather then crosses the link and
// decodes. Its rows recur -- a prompt's n-gram ids repeat across turns, and within one request
// across the heads whose hashes collide -- so this keeps the decoded rows on the card, four ways a
// set, and a hit neither crosses the link nor dequantises. The host side below says when it runs.
//
// One wave a row: a lane carries eight of the row's 160 entries, so the payload is one 16-byte
// vector a lane and the tag check is one thread's work. The row width is the model's
// (rad_block_ple.h) and the entry refuses other widths.
//
// A way is a seqlock: `gen` is 0 while empty, odd while one writer owns the way, and even,
// advancing modulo 2^32, once the row is published. The generation is what makes a reader's recheck
// sound -- a plain EMPTY/LOADING/READY word cannot tell a row that stayed put from one reclaimed
// and then rewritten for the same id while the reader was inside its window -- between the lookup
// that found the way and the recheck that would confirm it -- and the payload comes out of L1, so
// that stale read would be served again on later launches.
// Two a publication, so a way's generation wraps to 0 after 2^31 writes: the row reads as empty
// and is refilled, and only a reader kept across all 2^31 rewrites of that one way could take a
// stale row for a hit. Sized for the row, not made wraparound-free.
constexpr uint32_t kPleRow = 160, kPleLanes = kPleRow / 8, kPleWays = 4;
constexpr uint32_t kPleCacheThreads = 32;

__device__ __forceinline__ uint32_t ple_set(int32_t id, uint32_t mask) {
    return ((uint32_t)id * 2654435761u) & mask;
}

__device__ __forceinline__ uint32_t ple_pack2(float a, float b) {
    return (uint32_t)f32_to_bf16(a) | ((uint32_t)f32_to_bf16(b) << 16);
}

/* The slot holding `id` and the generation it was found under, or -1. The loads are scoped
 * atomics, so they read L2 and not the CU's own (non-coherent) L1, and each one invalidates that
 * L1 for the payload reads that follow. */
__device__ __forceinline__ int32_t ple_lookup(const int32_t* __restrict__ tags,
                                              uint32_t* __restrict__ gen, int32_t id,
                                              uint32_t base, uint32_t* out_gen) {
    for (uint32_t w = 0; w < kPleWays; ++w) {
        const uint32_t g =
            __hip_atomic_load(gen + base + w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (!g || (g & 1u)) continue;                 /* empty, or a writer owns it */
        if (__hip_atomic_load(tags + base + w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == id) {
            *out_gen = g;
            return (int32_t)(base + w);
        }
    }
    return -1;
}

/* Claims a way for `id`: an empty one, else the coldest of the four, so a row that keeps recurring
 * outlives the rows around it. A way some writer owns (an odd generation) is no candidate, and a
 * lost compare-and-swap means the choice went stale, so the scan repeats. A writer always publishes
 * before its block ends, so an odd generation that never clears means the kernel never finished and
 * the set simply carries one way less from then on. Bumping the generation to
 * odd *is* the claim, so a reader holding the older even generation sees the change on its
 * recheck. Lane 0 only. */
__device__ __forceinline__ int32_t ple_claim(int32_t* __restrict__ tags,
                                             uint32_t* __restrict__ gen,
                                             uint32_t* __restrict__ heat, int32_t id,
                                             uint32_t num_sets) {
    const uint32_t base = ple_set(id, num_sets - 1u) * kPleWays;
    for (uint32_t attempt = 0; attempt < kPleWays; ++attempt) {
        uint32_t pick = kPleWays, coldest = 0xFFFFFFFFu;
        for (uint32_t w = 0; w < kPleWays; ++w) {
            const uint32_t g =
                __hip_atomic_load(gen + base + w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (!g) { pick = w; break; }
            if (g & 1u) continue;
            const uint32_t h =
                __hip_atomic_load(heat + base + w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (pick == kPleWays || h < coldest) { pick = w; coldest = h; }
        }
        if (pick == kPleWays) continue;                 /* every way is being written */
        const uint32_t cur =
            __hip_atomic_load(gen + base + pick, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (cur & 1u) continue;
        if (atomicCAS((int*)(gen + base + pick), (int)cur, (int)(cur + 1u)) != (int)cur)
            continue;                                   /* another writer took it: choose again */
        tags[base + pick] = id;
        atomicExch(heat + base + pick, 4u);              /* a new row starts warm */
        return (int32_t)(base + pick);
    }
    return -1;                                          /* four live writers: skip the insert */
}

/* One row a block. `gen[slot]` is 0 while empty, odd while a writer owns the way, and even once
 * the row is published; `tags[slot]` is the row id, good only under an even generation. A writer
 * claims by compare-and-swap (even, or 0, -> odd), writes the tag and the payload, fences, then
 * publishes the next even generation. A reader loads the payload, fences, and compares the
 * generation it found at lookup with the one there now: a way reclaimed and rewritten for the same
 * id in between -- X, Y, X while the reader was in the middle -- has moved on by at least two, so
 * the reader sees a stale row even though the id matches again, which a bare READY word cannot
 * tell from a row that stayed put. The fence matters too -- an acquire fence alone leaves the
 * compiler free to sink the payload load past the check (checked in the emitted code for
 * gfx12), where a full one keeps it below. Two a publication, so the reader's "moved on" holds
 * for 2^31 publications of that way, after which its generation wraps back onto `g` and the check
 * can no longer tell the two apart (the local form of the wrap above).
 *
 * `probe` is a test's, not the gather's: null on every launch that is not a test, and when it is
 * set the words of r4d_ple_probe (r4d.h) hold a reader still between its lookup and its payload
 * load while another block rewrites the way it holds, so the load reads that rewrite and the
 * recheck has to reject it. Nothing else changes: same branches, same stores, same bytes. */

/* Test-only: reclaim the way the parked reader named, write a payload that is no row, and publish
 * it -- the reclaim-and-rewrite that reader's recheck has to see. One block a run serves: two
 * would publish a second generation on, and the reader's retire of that way would then lose its
 * compare-and-swap and leave the poisoned row published. A way a writer already owns is left to
 * the next block. Returns whether this block rewrote. */
__device__ __forceinline__ bool ple_probe_rewrite(int32_t* __restrict__ tags,
                                                  uint32_t* __restrict__ gen,
                                                  uint8_t* __restrict__ data,
                                                  uint32_t* __restrict__ probe, uint32_t lane) {
    int32_t svc = -1;
    uint32_t svc_odd = 0;
    if (lane == 0 &&
        __hip_atomic_load(probe + r4d_ple_probe_parked, __ATOMIC_ACQUIRE,
                          __HIP_MEMORY_SCOPE_AGENT) &&
        atomicCAS((int*)(probe + r4d_ple_probe_serve), 0, 1) == 0) {
        const uint32_t s =
            __hip_atomic_load(probe + r4d_ple_probe_slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const uint32_t cur = __hip_atomic_load(gen + s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (!(cur & 1u) && atomicCAS((int*)(gen + s), (int)cur, (int)(cur + 1u)) == (int)cur) {
            tags[s] = (int32_t)__hip_atomic_load(probe + r4d_ple_probe_id, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
            svc = (int32_t)s;
            svc_odd = cur + 1u;
        }
    }
    svc = __shfl(svc, 0);
    svc_odd = __shfl(svc_odd, 0);
    if (svc < 0) return false;
    const uint4 bad = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    if (lane < kPleLanes)
        *(uint4*)(data + (size_t)svc * (kPleRow * 2u) + (size_t)lane * 16u) = bad;
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        __hip_atomic_store(gen + svc, svc_odd + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        atomicAdd(probe + r4d_ple_probe_rewrites, 1u);
        __hip_atomic_store(probe + r4d_ple_probe_parked, 0u, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);       /* the reader may recheck now */
    }
    __syncwarp();
    return true;
}

__global__ __launch_bounds__(kPleCacheThreads) void ple_embed_lookup_q_kernel(
    const int32_t* __restrict__ tok, const uint8_t* __restrict__ wte,
    const void* __restrict__ sc, uint32_t sc_f32, uint16_t* __restrict__ x, uint32_t x_ld,
    int32_t n_vocab, int32_t vocab_off, int32_t* __restrict__ tags, uint32_t* __restrict__ gen,
    uint32_t* __restrict__ heat, uint8_t* __restrict__ data, uint32_t num_sets,
    uint32_t* __restrict__ probe) {
    const uint32_t lane = threadIdx.x, m = blockIdx.x;

    if (probe && probe[r4d_ple_probe_mode] && blockIdx.x > 0) {
        /* The test's servant blocks: wait for block 0 to park in its window, before the payload
         * load that follows, rewrite the way it holds, then gather this row as any launch would.
         * Bounded, so a run that never gets a park still ends -- and fails on the conflict
         * counter. */
        for (uint32_t i = 0; i < (1u << 26); ++i)
            if (ple_probe_rewrite(tags, gen, data, probe, lane) ||
                __hip_atomic_load(probe + r4d_ple_probe_rewrites, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT))
                break;
    }

    const int32_t id = tok[m] - vocab_off;
    const bool keep = id >= 0 && id < n_vocab;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld;

    int32_t slot = -1;
    uint32_t g = 0;
    if (keep && lane == 0)
        slot = ple_lookup(tags, gen, id, ple_set(id, num_sets - 1u) * kPleWays, &g);
    slot = __shfl(slot, 0);
    g = __shfl(g, 0);

    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (slot >= 0) {
        /* The test's handshake: park here, between the lookup and the payload load, until another
         * block has rewritten this way -- so the load reads the rewritten payload, which is no
         * row, and only the generation check can still send this reader to the decode path. A
         * reader that took the hit would put those bytes in the output, which is what makes the
         * check load-bearing for the case. Bounded -- a run that never gets its rewrite fails on
         * the conflict counter, not on a hang. */
        if (probe && probe[r4d_ple_probe_mode] && blockIdx.x == 0 && lane == 0 &&
            atomicCAS((int*)(probe + r4d_ple_probe_claim), 0, 1) == 0) {
            __hip_atomic_store(probe + r4d_ple_probe_slot, (uint32_t)slot, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(probe + r4d_ple_probe_gen, g, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(probe + r4d_ple_probe_id, (uint32_t)id, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();
            __hip_atomic_store(probe + r4d_ple_probe_parked, 1u, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
            for (uint64_t i = 0; i < (1ull << 26); ++i)
                if (!__hip_atomic_load(probe + r4d_ple_probe_parked, __ATOMIC_ACQUIRE,
                                       __HIP_MEMORY_SCOPE_AGENT))
                    break;
        }
        __syncwarp();
        if (lane < kPleLanes)
            v = *(const uint4*)(data + (size_t)slot * (kPleRow * 2u) + (size_t)lane * 16u);
        __threadfence();                   /* every lane's payload load is done before the recheck */
        __syncwarp();
        bool hit = false;
        if (lane == 0) {
            const bool same_gen =
                __hip_atomic_load(gen + slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == g;
            const bool same_id =
                __hip_atomic_load(tags + slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == id;
            if (probe && !same_gen && same_id) {
                atomicAdd(probe + r4d_ple_probe_conflicts, 1u);   /* the window, entered */
                /* Retire the way the rewriter published: a row with this id's tag and a payload
                 * that is no row's would be a hit for the next launch. 0 is empty, so it reads as
                 * a miss and is refilled; a lost swap means a writer has it, and it is then its
                 * own to publish. */
                const uint32_t cur =
                    __hip_atomic_load(gen + slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                atomicCAS((int*)(gen + slot), (int)cur, 0);
            }
            hit = same_gen && same_id;
        }
        hit = __shfl((int)hit, 0) != 0;
        if (hit) {
            if (lane < kPleLanes) *(uint4*)(dst + (size_t)lane * 8u) = v;
            if (lane == 0) atomicAdd(heat + slot, 1u);
            return;
        }
    }

    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    if (!keep) {
        if (lane < kPleLanes) *(uint4*)(dst + (size_t)lane * 8u) = zero;
        return;
    }

    /* Miss: decode the eight codes this lane carries, in the generic kernel's order and rounding. */
    const float s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : bf16_to_f32(*(const uint16_t*)sc);
    if (lane < kPleLanes) {
        const uint2 q = *(const uint2*)(wte + (size_t)id * kPleRow + (size_t)lane * 8u);
        const v2f a = r4d_fp8::e4m3_pk<false>(q.x), b = r4d_fp8::e4m3_pk<true>(q.x);
        const v2f c = r4d_fp8::e4m3_pk<false>(q.y), d = r4d_fp8::e4m3_pk<true>(q.y);
        v = make_uint4(ple_pack2(a.x * s, a.y * s), ple_pack2(b.x * s, b.y * s),
                       ple_pack2(c.x * s, c.y * s), ple_pack2(d.x * s, d.y * s));
        *(uint4*)(dst + (size_t)lane * 8u) = v;
    }

    int32_t islot = -1;
    if (lane == 0) islot = ple_claim(tags, gen, heat, id, num_sets);
    islot = __shfl(islot, 0);
    if (islot < 0) return;
    if (lane < kPleLanes)
        *(uint4*)(data + (size_t)islot * (kPleRow * 2u) + (size_t)lane * 16u) = v;
    __threadfence();                       /* every lane's payload reaches L2 ... */
    __syncthreads();                       /* ... before lane 0 publishes */
    if (lane == 0) {
        const uint32_t odd = __hip_atomic_load(gen + islot, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(gen + islot, odd + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// ---------------------------------------------------------------------------- gather_rows
// ref_gather_rows. Like the embedding gather, with the same negative-index rule, but the source
// row pitch is the caller's: it gathers from padded activation buffers (one logits row per
// sequence out of a prefill chunk).
__global__ __launch_bounds__(kThreads) void gather_rows_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ idx, uint16_t* __restrict__ y,
    uint32_t n, int32_t src_rows, uint32_t x_ld, uint32_t y_ld) {
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
// ref_gather_rows and ref_scatter_rows at dtype u8. Separate from the bf16 pair because these move
// opaque bytes: the KV pool's tier traffic, where a row is one layer's fragment of one block.
// libref's bf16 path loads through float, which need not preserve a NaN payload; at u8 every value
// is exact in float, so oracle and kernel agree on any payload.
//
// One uchar4 per thread step. The registry constrains `n` to a multiple of 16, so there is no
// ragged path.
__global__ __launch_bounds__(kThreads) void gather_rows_u8_kernel(
    const uchar4* __restrict__ x, const int32_t* __restrict__ idx, uchar4* __restrict__ y,
    uint32_t n4, int32_t src_rows, uint32_t x_ld4, uint32_t y_ld4) {
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

// The mirror. A negative index writes nothing: the destination is a live pool, so the gather's
// zero-fill would be data loss here.
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
// `wadd` is added to the gain in f32 after the load: it is GemmaRMSNorm's `1 +` (Qwen3.5,
// Qwen3-Next and Gemma store a zero-centred gamma), and forming `1 + w` in bf16 loses 0.3% per
// channel. r4d_rmsnorm_had_quant_i8 takes the same argument.
//
// The gain is f32 or bf16 (WF32), read off the operand by the shim. Architecture plugins declare
// norm gains RAD_F32, since they are small and a rounded gain shows in the norm.
//
// Two passes over the row: n is 256 (a QK-norm head) or 5120 (the hidden state), so the row is in
// L1 for the second pass, and one-pass Welford would only change the rounding.
//
// `TX` is the row's element: bf16, or f32 for a residual stream kept in f32 (a video decoder's),
// read as it is; the output is bf16 either way.
__device__ __forceinline__ float act_f32(uint16_t v) { return bf16_to_f32(v); }
__device__ __forceinline__ float act_f32(float v) { return v; }

template <bool WF32, uint32_t THR, typename TX = uint16_t>
__global__ __launch_bounds__(THR) void rmsnorm_kernel(
    const TX* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, float mult, uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const TX* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = act_f32(xr[i]); ss += v * v; }
    ss = block_sum<THR / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float g = !w ? 1.0f : WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        yr[i] = f32_to_bf16(act_f32(xr[i]) * s * (g + wadd) * mult);
    }
}

// ---------------------------------------------------------------------------- add_rmsnorm
// x = x + a, rounded as `add` rounds it, then y = rmsnorm(x) over the x just written, computed as
// rmsnorm_kernel computes it: the same width (r4d_act_threads), the same columns a thread, the same
// block_sum. A thread re-reads only the x it wrote itself, so no barrier separates the passes.
// Byte-identical to `add` followed by `rmsnorm`, on x and on y.
template <bool WF32, uint32_t THR>
__global__ __launch_bounds__(THR) void add_rmsnorm_kernel(
    uint16_t* __restrict__ x, const uint16_t* __restrict__ a, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint32_t n, float eps, float wadd, float mult, uint32_t x_ld,
    uint32_t a_ld, uint32_t y_ld, uint32_t M) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    const uint16_t* __restrict__ ar = a + (size_t)m * a_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint16_t h = f32_to_bf16(bf16_to_f32(xr[i]) + bf16_to_f32(ar[i]));
        xr[i] = h;
        const float v = bf16_to_f32(h);
        ss += v * v;
    }
    ss = block_sum<THR / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float g = !w ? 1.0f : WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * s * (g + wadd) * mult);
    }
}

// ---------------------------------------------------------------------------- rmsnorm_p
// ref_rmsnorm_p: y = x * rsqrt(mean(x^2) + eps), rmsnorm with no gain. Gemma's attention applies
// it to V (llama.cpp's ggml_rms_norm(V, eps)), which has no checkpoint gain. Accumulates exactly as
// rmsnorm does, so the two agree bit for bit where rmsnorm's gain is one.
__global__ __launch_bounds__(kThreads) void rmsnorm_p_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    __shared__ float lds[kThreads / kWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) { const float v = bf16_to_f32(xr[i]); ss += v * v; }
    ss = block_sum<kThreads / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * s);
}

// ---------------------------------------------------------------------------- add / mul
// ref_add / ref_mul with one broadcast rule: an operand of exactly n elements against an output of
// rows*n broadcasts along rows; nothing else broadcasts, so shape bugs are not hidden.
// `b_step` 0 reads one element per row; with b_ld 0 too, a scalar for the whole op. `div` (add
// only) divides the sum after its rounding and rounds again, the last step of a mean summed
// pairwise; 1 is the plain sum.
__global__ __launch_bounds__(kThreads) void binary_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, int mul, uint32_t b_step,
    float div) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ ar = a + (size_t)m * a_ld;
    const uint16_t* __restrict__ br = b + (size_t)m * b_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float u = bf16_to_f32(ar[i]), v = bf16_to_f32(br[i * b_step]);
        float r = mul ? u * v : u + v;
        if (div != 1.0f) r = bf16_to_f32(f32_to_bf16(r)) / div;
        yr[i] = f32_to_bf16(r);
    }
}

// The f32 form (a video decoder's residual stream, a vocoder's branch mean), with the same
// broadcast rule and `div`. Flat over the M x n extent: a vocoder's last stages are a few channels
// over a million rows, which a workgroup a row would leave mostly idle. `y` may be `a` (an
// in-place residual): each element is read before it is written, by the same thread.
__global__ __launch_bounds__(kThreads) void binary_f32_kernel(
    const float* __restrict__ a, const float* __restrict__ b, float* y, uint32_t n,
    uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, int mul, uint32_t b_step, float div,
    uint64_t total) {
    for (uint64_t t = (uint64_t)blockIdx.x * kThreads + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * kThreads) {
        const uint64_t m = t / n;
        const uint32_t i = (uint32_t)(t - m * n);
        const float u = a[m * a_ld + i], v = b[m * b_ld + (size_t)i * b_step];
        float r = mul ? u * v : u + v;
        if (div != 1.0f) r = r / div;
        y[m * y_ld + i] = r;
    }
}

// ref_mul_weight over an f32 weight (a layer scale kept f32): y = x * w, one row of n elements
// broadcast down the rows, or with `w_step` 0 one element for the whole op.
__global__ __launch_bounds__(kThreads) void mul_wf32_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld, uint32_t w_step) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * w[i * w_step]);
}

// ref_mul_weight into an f32 output (a layer scale on a branch about to join an f32 residual):
// x bf16 or f32, the weight bf16 or f32, the product rounded once to f32. Flat, as the f32 binary.
template <typename TX, typename TW>
__global__ __launch_bounds__(kThreads) void mul_weight_f32_kernel(
    const TX* __restrict__ x, const TW* __restrict__ w, float* __restrict__ y, uint32_t n,
    uint32_t x_ld, uint32_t y_ld, uint32_t w_step, uint64_t total) {
    for (uint64_t t = (uint64_t)blockIdx.x * kThreads + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * kThreads) {
        const uint64_t m = t / n;
        const uint32_t i = (uint32_t)(t - m * n);
        y[m * y_ld + i] = act_f32(x[m * x_ld + i]) * act_f32(w[(size_t)i * w_step]);
    }
}

// ---------------------------------------------------------------------------- sigmoid
// ref_sigmoid, its own op so the selector can name a gated attention's sigmoid(gate).
//
// Row-wise with a pitch because the operand is a column slice: the gate is the second half of each
// head of an interleaved [tokens, heads, 2*head_dim] projection, `n` wide inside a stride of 2n.
__global__ __launch_bounds__(kThreads) void sigmoid_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(1.0f / (1.0f + expf(-bf16_to_f32(xr[i]))));
}

// ---------------------------------------------------------------------------- logit_softcap
// ref_logit_softcap: y = cap * tanh(x / cap), the final-logit softcap. Row-wise with a pitch like
// sigmoid; on a contiguous buffer x_ld == n.
//
// A row is spread over gridDim.y blocks: it is a vocabulary shard (65536 wide at Gemma 4's tp 4)
// and decode has few rows, so one block per row leaves most CUs idle (~85 us at one row on an
// R9700). Elementwise, so the split changes no bits.
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, float cap, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = blockIdx.y * kThreads + threadIdx.x; i < n; i += gridDim.y * kThreads)
        yr[i] = f32_to_bf16(cap * tanhf(bf16_to_f32(xr[i]) / cap));
}

/* The f32 leg, for logits the gemm writes as f32. Same shape as the bf16 leg, with no rounding. */
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel_f32(
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t n, float cap, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const float* __restrict__ xr = x + (size_t)m * x_ld;
    float* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = blockIdx.y * kThreads + threadIdx.x; i < n; i += gridDim.y * kThreads)
        yr[i] = cap * tanhf(xr[i] / cap);
}

/* Blocks per softcap row: eight elements a thread, at most 64 blocks. */
static uint32_t softcap_blocks_per_row(int n) {
    uint32_t b = (uint32_t)((n + kThreads * 8 - 1) / (kThreads * 8));
    return b < 1 ? 1 : (b > 64 ? 64 : b);
}

// ---------------------------------------------------------------------------- scale
// ref_scale: y = x * s, s a [1] weight broadcast over every element (Gemma's per-layer scalar on
// the residual stream). Row-wise with a pitch like sigmoid. The scalar is loaded once per block;
// it is f32 or bf16 (the checkpoint's layer_scalar is bf16), selected by `s_dt`.
__global__ __launch_bounds__(kThreads) void scale_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ s, uint32_t s_dt,
    uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const float sv = (s_dt == RAD_F32) ? ((const float*)s)[0] : bf16_to_f32(((const uint16_t*)s)[0]);
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * sv);
}

// ---------------------------------------------------------------------------- silu_mul
// ref_silu_mul: y[.., i] = silu(gu[.., i]) * gu[.., n + i]. The gate is the first half, the order in
// which rad-convert fuses gate_proj and up_proj.
//
// A row is split over grid.y, kSiluPer elements per thread, so decode's few rows use more than a
// few workgroups. A thread's loads are unconditional (a column past the row reads the last element
// and its store is dropped), so all are in flight at once.
constexpr uint32_t kSiluPer = 4;
__global__ __launch_bounds__(kThreads) void silu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    const uint32_t i0 = blockIdx.y * (kSiluPer * kThreads) + threadIdx.x;
    uint16_t gv[kSiluPer], uv[kSiluPer];
#pragma unroll
    for (uint32_t u = 0; u < kSiluPer; ++u) {
        const uint32_t i = i0 + u * kThreads, ic = i < n ? i : n - 1u;
        gv[u] = gr[ic];
        uv[u] = gr[n + ic];
    }
#pragma unroll
    for (uint32_t u = 0; u < kSiluPer; ++u) {
        const uint32_t i = i0 + u * kThreads;
        const float g = bf16_to_f32(gv[u]);
        if (i < n) yr[i] = f32_to_bf16((g / (1.0f + expf(-g))) * bf16_to_f32(uv[u]));
    }
}

// ---------------------------------------------------------------------------- gelu_mul
// ref_gelu_mul: y[.., i] = gelu_tanh(gu[.., i]) * gu[.., n + i], gate first as in silu_mul. The tanh
// GELU (gemma's gelu_pytorch_tanh), not erf, with the same constants as the gemm_nt_bias
// "gelu_tanh" epilogue so the two agree bit for bit.
__global__ __launch_bounds__(kThreads) void gelu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = bf16_to_f32(gr[i]);
        const float k0 = 0.7978845608028654f, k1 = 0.044715f;
        const float gelu = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
        yr[i] = f32_to_bf16(gelu * bf16_to_f32(gr[n + i]));
    }
}

// ---------------------------------------------------------------------------- rope
// ref_rope, in place over a fused [n_head + 2*n_head_kv] head row; the trailing V heads are left
// alone, which is why the op takes the fused projection.
//
// NeoX pairs element i with i + rot/2; GPT-J pairs 2i with 2i+1. The checkpoint decides, and the
// wrong choice yields fluent text with broken positions. With one position component per token,
// mrope is bit-identical to neox (rad_block_attn_gated.h), so the shim maps it.
//
// Partial rotary: `rot` may be less than head_dim (64 of 256 here) and the tail is untouched. The
// base ladder's exponent divides by `den`: `rot` for `rotary` (default), `head_dim` for `head`
// (the proportional variant).
//
// One thread per pair: two loads and two stores, no reduction, no LDS.
//
// Multi-component modes (ref_rope's rope_freqs): `mode` 2 mrope, 3 imrope, 4 axial, up to three
// sections, with component c of token m at pos[c * plane + m]. `plane` 0 is the one-component
// form, where every mode is neox.
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
    uint32_t head_dim, uint32_t n_rope_heads, uint32_t rot, uint32_t nfreq, int den, int gptj,
    float theta, float rscale, uint32_t row_pitch,
    int mode, int s0, int s1, int s2, int n_sec, uint32_t plane, RadYarn yarn, RadLlama3 l3) {
    const uint32_t m = blockIdx.x;
    const uint32_t half = rot / 2;
    const float div = rscale != 0.0f ? rscale : 1.0f;
    const float p0 = (float)pos[m] / div;
    const int sec[3] = { s0, s1, s2 };

    for (uint32_t t = threadIdx.x + blockIdx.y * kThreads; t < n_rope_heads * half;
         t += kThreads * gridDim.y) {
        const uint32_t h = t / half, i = t % half;
        uint16_t* __restrict__ hd = qkv + (size_t)m * row_pitch + (size_t)h * head_dim;
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)den);
        float p = p0;
        if (plane) {
            uint32_t j = i;
            const int c = rope_comp(mode, i, sec, n_sec, &j);
            p = (float)pos[(size_t)c * plane + m] / div;
            if (mode == 4)
                inv = 1.0f / powf(theta, (float)(2 * j) / (float)(2 * sec[c]));
        }
        inv = yarn.frequency(inv, (int)i);
        inv = l3.frequency(inv);
        /* The proportional ladder rotates the whole head with every frequency past nfreq zero
         * (HF's _compute_proportional_rope_parameters): pair i is (i, i + head_dim/2), and pairs
         * past nfreq turn by 0 (cos 1, sin 0), returning the value exactly. */
        if (i >= nfreq) inv = 0.0f;
        const float ang = p * inv;
        float c, s;
        sincosf(ang, &s, &c);
        if (yarn.factor != 0.0f) { c *= yarn.magnitude; s *= yarn.magnitude; }
        const uint32_t i0 = gptj ? 2 * i : i;
        const uint32_t i1 = gptj ? 2 * i + 1 : i + half;
        const float x0 = bf16_to_f32(hd[i0]), x1 = bf16_to_f32(hd[i1]);
        hd[i0] = f32_to_bf16(x0 * c - x1 * s);
        hd[i1] = f32_to_bf16(x0 * s + x1 * c);
    }
}

// ---------------------------------------------------------------------------- kv_store
// ref_kv_store. The paged layout every kernel in this project shares:
//
//     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]
//
// K then V inside a slot: key element d at [..., d], value element d at [..., head_dim + d]. A flat
// slot index (`slot_mapping`) splits into block = slot / block_size, offset = slot % block_size. A
// negative slot is padding and stores nothing.
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

    for (uint32_t j = threadIdx.x; j < kv_heads * head_dim; j += kThreads) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint16_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        c[d]            = k[(size_t)t * k_ld + j];
        c[head_dim + d] = v[(size_t)t * v_ld + j];
    }
}

// The same scatter into an E4M3 cache: the write side of the fp8kv attention family and its
// `k_descale` / `v_descale` operands.
//
// The scale is a caller constant: the reader's descale is indexed `[seq * kv_heads + kv_head]`
// (r4d.h), one number per sequence, so it cannot depend on the token written. vLLM has the same
// contract and defaults both scales to 1.0, where the store is a plain E4M3 round of the bf16.
//
// Four columns per thread: two v_cvt_pk_fp8_f32 fill one dword, stored at once. The entry point
// requires head_dim % 4 == 0, so a group never straddles two heads.
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
            kf[e] = bf16_to_f32(k[(size_t)t * k_ld + j + (uint32_t)e]) * k_inv;
            vf[e] = bf16_to_f32(v[(size_t)t * v_ld + j + (uint32_t)e]) * v_inv;
        }
        // Clamped because v_cvt_pk_fp8_f32 returns NaN for out-of-range input instead of
        // saturating (as in quant_act_fp8).
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

// The fp8_v4 store: K as kv_store_fp8 writes it, V as int4 in groups of 32 with an f16 scale
// (absmax / 7). Four columns a thread as above, so a group of 32 is 8 lanes; head_dim is a multiple
// of 32, so a group never straddles two heads. A token's row is `row` bytes. K6 is the k6_v4 store:
// K as int6 in groups of 32 (absmax / 31) in the row layout of abi/rad_builder.h's RAD_KVQ_K6V4, V
// as above after it.
template <int K6>
__global__ __launch_bounds__(kThreads) void kv_store_fp8v4_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld, float k_inv, float v_inv, uint32_t row) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * row;
    const size_t head_step = (size_t)block_size * row;

    for (uint32_t j = threadIdx.x * 4u; j < kv_heads * head_dim; j += kThreads * 4u) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint8_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        float kf[4], vf[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kf[e] = bf16_to_f32(k[(size_t)t * k_ld + j + (uint32_t)e]) * k_inv;
            vf[e] = bf16_to_f32(v[(size_t)t * v_ld + j + (uint32_t)e]) * v_inv;
        }
        const uint32_t voff = K6 ? head_dim * 3 / 4 : head_dim;
        const uint32_t vsoff = K6 ? head_dim * 21 / 16 : head_dim * 3 / 2;
        if constexpr (K6) {
            // A group of 32 is 8 lanes; its scale absmax / 31 in f16, codes 6-bit two's complement.
            float kmax = fmaxf(fmaxf(fabsf(kf[0]), fabsf(kf[1])), fmaxf(fabsf(kf[2]), fabsf(kf[3])));
            for (int m = 4; m >= 1; m >>= 1) kmax = fmaxf(kmax, __shfl_xor(kmax, m, 32));
            // Clamped so 31 times it stays an f16 (65504): a group gone inf
            // or NaN decodes finite, as the E4M3 form's clamp to 448 does.
            const __fp16 k16 = (__fp16)fminf(kmax / 31.0f, 2112.0f);
            const float ksc = (float)k16;
            uint32_t q[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float r = ksc > 0.0f ? rintf(kf[e] / ksc) : 0.0f;
                q[e] = (uint32_t)((int)fminf(fmaxf(r, -31.0f), 31.0f)) & 0x3Fu;
            }
            *(uint16_t*)(c + d / 2) = (uint16_t)((q[0] & 15u) | ((q[1] & 15u) << 4) |
                                                 ((q[2] & 15u) << 8) | ((q[3] & 15u) << 12));
            // The high bits of dims x..x+3 of a 16-dim chunk (x = d % 16 in 0, 4, 8, 12): bits
            // 0-3 of bytes x/2 and x/2+1 for x < 8, bits 4-7 of bytes (x-8)/2 and (x-8)/2+1 for
            // x >= 8. The lane two over holds the dims eight on, so the lower lane writes both.
            const uint32_t hb = ((q[0] >> 4) | ((q[1] >> 4) << 2)) | ((((q[2] >> 4) | ((q[3] >> 4) << 2))) << 8);
            const uint32_t up = (uint32_t)__shfl_xor((int)hb, 2, 32);
            if ((d & 15u) < 8u)
                *(uint16_t*)(c + head_dim / 2 + (d / 16) * 4 + (d & 7u) / 2) = (uint16_t)(hb | (up << 4));
            if ((d & 31u) == 0)
                *(uint16_t*)(c + head_dim * 5 / 4 + (d / 32) * 2) = __builtin_bit_cast(uint16_t, k16);
        } else {
            int pk = 0;
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[0], -448.0f), 448.0f),
                                                 fminf(fmaxf(kf[1], -448.0f), 448.0f), pk, false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[2], -448.0f), 448.0f),
                                                 fminf(fmaxf(kf[3], -448.0f), 448.0f), pk, true);
            *(uint32_t*)(c + d) = (uint32_t)pk;
        }

        float amax = fmaxf(fmaxf(fabsf(vf[0]), fabsf(vf[1])), fmaxf(fabsf(vf[2]), fabsf(vf[3])));
        for (int m = 4; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        // Clamped so 7 times it stays an f16: see the K scale above.
        const __fp16 s16 = (__fp16)fminf(amax / 7.0f, 8188.0f);
        const float sc = (float)s16;
        uint32_t q[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float r = sc > 0.0f ? rintf(vf[e] / sc) : 0.0f;
            q[e] = (uint32_t)((int)fminf(fmaxf(r, -7.0f), 7.0f)) & 0xFu;
        }
        *(uint16_t*)(c + voff + d / 2) = (uint16_t)(q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12));
        if ((d & 31u) == 0)
            *(uint16_t*)(c + vsoff + (d / 32) * 2) = __builtin_bit_cast(uint16_t, s16);
    }
}

}  // namespace

// ================================================================================ entry points
//
// Pitches are in elements; 0 means tight. Pointers are bf16 except the index arrays, which are i32
// (the core's derived buffers are all i32, core/sched/batch.cpp).

/* The plugin entry and r4d.h's long-stream entry share this body. */
static int embed_lookup_bf16_emit(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                  int vocab_off, int x_ld, float mult, RadStream stream) {
    if (!tok || !wte || !x) return -1;
    if (M <= 0 || n_embd <= 0 || n_vocab <= 0 || vocab_off < 0) return -2;
    if (!x_ld) x_ld = n_embd;
    return R4D_EMIT(stream, embed_lookup_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const int32_t*)tok, (const uint16_t*)wte,
                    (uint16_t*)x, (uint32_t)n_embd, (int32_t)n_vocab, (int32_t)vocab_off,
                    (uint32_t)x_ld, mult) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_embed_lookup_bf16(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                     int vocab_off, int x_ld, long stream) {
    return embed_lookup_bf16_emit(tok, wte, x, M, n_embd, n_vocab, vocab_off, x_ld, 1.0f,
                                  (RadStream)(uintptr_t)stream);
}

/* The quad walk when width, pitch and both bases allow its 4- and 8-byte accesses (a bf16 quad is
 * 8 bytes); otherwise one entry per thread. */
static int embed_lookup_q_emit(const int32_t* tok, const void* wte, bool q, const void* sc,
                               bool sc_f32, uint16_t* x, long long M, int n_embd, int n_vocab,
                               int vocab_off, long long x_ld, RadStream stream) {
    const bool quad = n_embd % 4 == 0 && x_ld % 4 == 0 && r4d_aligned(x, 8) &&
                      r4d_aligned(wte, q ? 4 : 8);
    const uint64_t threads = (uint64_t)M * (uint64_t)(quad ? n_embd / 4 : n_embd);
    const dim3 grid((uint32_t)((threads + kThreads - 1) / kThreads));
#define R4D_ELQ(V, Q)                                                                             \
    R4D_EMIT(stream, (embed_lookup_q_kernel<V, Q>), grid, dim3(kThreads), 0, tok, wte, sc,      \
             (uint32_t)(sc_f32 ? 1u : 0u), x, (uint32_t)M, (uint32_t)n_embd, (int32_t)n_vocab,   \
             (int32_t)vocab_off, (uint32_t)x_ld)
    const int rc = quad ? (q ? R4D_ELQ(4, true) : R4D_ELQ(4, false))
                        : (q ? R4D_ELQ(1, true) : R4D_ELQ(1, false));
#undef R4D_ELQ
    return rc == RAD_OK ? 0 : -4;
}

/* ------------------------------------------------------------------ the PLE VRAM row cache
 *
 * Opt-in, off by default: RADIANCE_PLE_VRAM_CACHE_SETS names the set count (rounded down to a power
 * of two, capped at kPleCacheMaxSets); 0 or absent allocates nothing and every gather runs the
 * generic kernel above, unchanged. 65536 sets is 262144 rows of 320 B, 80 MiB of payload plus 3 MiB
 * of tags, generations and heat counters; at the cap of 262144 sets it is 1,048,576 rows, 320 MiB
 * plus 12 MiB. The init prints what it took.
 *
 * Why it can pay: with --ngram-placement ram the table is pinned host memory, so the gather reads
 * 320 B a row over the link and dequantises it, per row and per rank. A row that recurs (a prompt
 * re-read across turns, a head whose hash collides) is then free. Whether a workload's hit rate
 * repays the insert traffic is a measurement this change does not make; there is no default on and
 * no speed claim here (archive PR #2's 16x is an unverified ABI-16 number).
 *
 * Lifetime, identity, budget:
 *   - one per resolved instance, from init() to fini() (rad_abi.h), which is one per PLE gather in
 *     the program and one model load: a reload re-inits and starts empty, with no table to
 *     identify.
 *   - the table a launch passes is bound on the first launch. A launch handing a different table
 *     bypasses the cache rather than rebinding, so one table's rows can never answer for another
 *     and no launch has to clear the payload. A cache whose instance outlives its table leaks
 *     nothing, since the payload is never dereferenced again.
 *   - the allocation happens in init, which runs before the VRAM budget reads the card, so it is
 *     part of that budget's "already held" (core/mem/vram_budget.h) and not an unaccounted claim
 *     on a pool; a failed allocation only leaves the cache off.
 *
 * A way is a seqlock rather than a status word: generation 0 is empty, odd is owned by one writer,
 * even is published. A reader keeps the generation it found at lookup and demands the same one
 * after its payload load, so a way reclaimed and rewritten for the same id in between -- which a
 * READY word cannot distinguish from a row that stayed put -- reads as a miss.
 *
 * The mechanism is the device kernel above (ple_embed_lookup_q_kernel). Its hit and miss paths
 * write the same bytes as embed_lookup_q_kernel, which r4d_selftest's `ple_cache` checks against a
 * host gather at every stage: cold, warm, colliding, after a table change, after a reset, and from
 * four concurrent streams. That case also attaches the r4d_ple_probe handshake, which forces the
 * reclaim/rewrite window on one reader and counts the rechecks that land in it.
 */
struct PleVramCache {
    int32_t*  tags = nullptr;    /* row id per way, valid under an even generation */
    uint32_t* gen  = nullptr;    /* 0 empty, odd owned by a writer, even published (the seqlock) */
    uint32_t* heat = nullptr;    /* hit counter per way: the replacement policy's age */
    uint8_t*  data = nullptr;    /* kPleCacheRowBytes rows, decoded bf16 */
    uint32_t* probe = nullptr;   /* a test's r4d_ple_probe words; null everywhere else */
    uint32_t  sets = 0;
    int       device = -1;
    std::atomic<const void*> table{nullptr};
};

static void ple_cache_free(PleVramCache* c) {
    if (!c) return;
    (void)hipFree(c->data);
    (void)hipFree(c->heat);
    (void)hipFree(c->gen);
    (void)hipFree(c->tags);
    delete c;
}

constexpr uint32_t kPleCacheWays     = 4;
constexpr uint32_t kPleCacheRowBytes = 320;          /* 160 bf16, kPleRow above */
constexpr uint32_t kPleCacheMaxSets  = 1u << 18;     /* 1,048,576 rows: 320 MiB of payload */

static uint32_t ple_cache_sets_from_env() {
    const char* e = std::getenv("RADIANCE_PLE_VRAM_CACHE_SETS");
    if (!e || !*e) return 0;
    char* end = nullptr;
    const long want = std::strtol(e, &end, 10);
    if (end == e) return 0;
    if (want <= 0) return 0;
    long sets = 1;
    while (sets * 2 <= want && (uint32_t)(sets * 2) <= kPleCacheMaxSets) sets *= 2;
    return (uint32_t)sets;
}

extern "C" int r4d_rad_embed_lookup_q_init(const RadParam* p, int n_p, int rank, int world,
                                           void** out) {
    (void)p; (void)n_p; (void)rank; (void)world;
    if (out) *out = nullptr;
    const uint32_t sets = ple_cache_sets_from_env();
    if (!sets) return RAD_OK;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return RAD_OK;
    PleVramCache* c = new (std::nothrow) PleVramCache();
    if (!c) return RAD_OK;
    c->sets = sets;
    c->device = dev;
    const size_t ways = (size_t)sets * kPleCacheWays;
    const bool got = hipMalloc((void**)&c->tags, ways * sizeof(int32_t)) == hipSuccess &&
                     hipMalloc((void**)&c->gen, ways * sizeof(uint32_t)) == hipSuccess &&
                     hipMalloc((void**)&c->heat, ways * sizeof(uint32_t)) == hipSuccess &&
                     hipMalloc((void**)&c->data, ways * kPleCacheRowBytes) == hipSuccess;
    if (!got || hipMemset(c->gen, 0, ways * sizeof(uint32_t)) != hipSuccess) {
        ple_cache_free(c);
        std::fprintf(stderr, "[r4d] ple cache: device %d could not allocate %u sets; the gathers "
                             "read the table as before\n", dev, sets);
        return RAD_OK;
    }
    std::fprintf(stderr, "[r4d] ple cache: device %d holds %u sets x %u ways, %.1f MiB "
                         "(%zu rows)\n", dev, sets, kPleCacheWays,
                 (double)(ways * (kPleCacheRowBytes + 12)) / (1024.0 * 1024.0), ways);
    if (out) *out = c;
    return RAD_OK;
}

extern "C" void r4d_rad_embed_lookup_q_fini(void* instance) {
    ple_cache_free((PleVramCache*)instance);
}

/* Test hook, not part of the plugin ABI (r4d.h): the probe words the kernel's handshake uses. The
 * test allocates them; a cache no test touched keeps `probe` null and its launches pass null. */
extern "C" int r4d_ple_probe(void* instance, unsigned* ctl) {
    if (!instance) return -1;
    ((PleVramCache*)instance)->probe = ctl;
    return 0;
}

/* True when this launch may use `c` for `table`: the first table seen binds the instance, and any
 * other leaves the cache alone. */
static bool ple_cache_binds(PleVramCache* c, const void* table) {
    const void* bound = c->table.load(std::memory_order_acquire);
    if (!bound &&
        c->table.compare_exchange_strong(bound, table, std::memory_order_acq_rel))
        return true;
    return bound == table;
}

static int embed_lookup_q_cached_emit(const int32_t* tok, const uint8_t* wte, const void* sc,
                                      bool sc_f32, uint16_t* x, long long M, int32_t n_vocab,
                                      int32_t vocab_off, long long x_ld, PleVramCache* c,
                                      RadStream stream) {
    return R4D_EMIT(stream, (ple_embed_lookup_q_kernel), dim3((uint32_t)M),
                    dim3(kPleCacheThreads), 0, tok, wte, sc, (uint32_t)(sc_f32 ? 1u : 0u), x,
                    (uint32_t)x_ld, n_vocab, vocab_off, c->tags, c->gen, c->heat, c->data,
                    c->sets, c->probe) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_embed_lookup_bf16_scaled(long tok, long wte, long x, int M, int n_embd,
                                           int n_vocab, int vocab_off, int x_ld, float scale,
                                           long stream) {
    return embed_lookup_bf16_emit(tok, wte, x, M, n_embd, n_vocab, vocab_off, x_ld, scale,
                                  (RadStream)(uintptr_t)stream);
}

static int gather_rows_bf16_emit(long x, long idx, long y, int M, int n, int src_rows,
                                 int x_ld, int y_ld, RadStream stream) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    return R4D_EMIT(stream, gather_rows_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)x, (const int32_t*)idx,
                    (uint16_t*)y, (uint32_t)n, (int32_t)src_rows,
                    (uint32_t)x_ld, (uint32_t)y_ld) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_bf16(long x, long idx, long y, int M, int n, int src_rows,
                                    int x_ld, int y_ld, long stream) {
    return gather_rows_bf16_emit(x, idx, y, M, n, src_rows, x_ld, y_ld,
                                 (RadStream)(uintptr_t)stream);
}

/* Byte rows. `n`, `x_ld` and `y_ld` are in bytes here and divided down to uchar4 inside. */
static int gather_rows_u8_emit(long x, long idx, long y, int M, int n, int src_rows,
                               int x_ld, int y_ld, RadStream stream) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if ((n | x_ld | y_ld) & 3) return -2;        /* the registry constrains n; pitches too */
    return R4D_EMIT(stream, gather_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)x, (const int32_t*)idx, (uchar4*)y,
                    (uint32_t)(n / 4), (int32_t)src_rows,
                    (uint32_t)(x_ld / 4), (uint32_t)(y_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_u8(long x, long idx, long y, int M, int n, int src_rows,
                                  int x_ld, int y_ld, long stream) {
    return gather_rows_u8_emit(x, idx, y, M, n, src_rows, x_ld, y_ld,
                               (RadStream)(uintptr_t)stream);
}

static int scatter_rows_u8_emit(long v, long idx, long x, int M, int n, int dst_rows,
                                int v_ld, int x_ld, RadStream stream) {
    if (!v || !idx || !x) return -1;
    if (M <= 0 || n <= 0 || dst_rows <= 0) return -2;
    if (!v_ld) v_ld = n;
    if (!x_ld) x_ld = n;
    if ((n | v_ld | x_ld) & 3) return -2;
    return R4D_EMIT(stream, scatter_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)v, (const int32_t*)idx, (uchar4*)x,
                    (uint32_t)(n / 4), (int32_t)dst_rows,
                    (uint32_t)(v_ld / 4), (uint32_t)(x_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_scatter_rows_u8(long v, long idx, long x, int M, int n, int dst_rows,
                                   int v_ld, int x_ld, long stream) {
    return scatter_rows_u8_emit(v, idx, x, M, n, dst_rows, v_ld, x_ld,
                                (RadStream)(uintptr_t)stream);
}

// `w_f32` selects an f32 or bf16 gain; a null `w` is a gain of one. `mult` scales the row before
// its single rounding.
extern "C" int r4d_rmsnorm_mult_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                     float mult, int w_f32, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    // One workgroup per row, so decode rows are latency-bound. r4d_act_threads is the measured
    // width table; r4d_rmsnorm_quant_fp8 must use the same one, because the width decides which
    // columns a thread folds into the sum of squares and the fused kernel must be byte-identical
    // to this one followed by the quantiser.
    const int thr = r4d_act_threads(M, n);
    // A drafter's 256-row head norm at 256 threads is exactly a full card; step past the cliff.
#define R4D_RN_T(wf, T)                                                                         \
    hipLaunchKernelGGL((rmsnorm_kernel<wf, T>), dim3(r4d_cliff_rows((uint32_t)M, 1, T / kWave)),  \
                       dim3(T), 0,                                                              \
                       (hipStream_t)stream, (const uint16_t*)x, (const void*)w, (uint16_t*)y,   \
                       (uint32_t)n, eps, wadd, mult, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M)
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

/* The unscaled form (mult = 1 is exact). */
extern "C" int r4d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                int w_f32, int x_ld, int y_ld, long stream) {
    return r4d_rmsnorm_mult_bf16(x, w, y, M, n, eps, wadd, 1.0f, w_f32, x_ld, y_ld, stream);
}

// The weightless norm. Unlike rmsnorm it is not templated on width: rmsnorm's width is pinned to
// match the fused rmsnorm_quant_fp8, and rmsnorm_p has no fused counterpart, so it uses kThreads.
extern "C" int r4d_rmsnorm_p_bf16(long x, long y, int M, int n, float eps,
                                  int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(rmsnorm_p_kernel, dim3(r4d_cliff_rows((uint32_t)M, 1, kThreads / kWave)),
                       dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n, eps,
                       (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `op` is 0 for a sum, 1 for a product. `b_ld` 0 broadcasts one row of n down the rows; -1 is a
// one-element b, broadcast as a scalar.
extern "C" int r4d_binary_bf16(long a, long b, long y, int M, int n, int op,
                               int a_ld, int b_ld, int y_ld, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!a_ld) a_ld = n;
    if (!y_ld) y_ld = n;
    const uint32_t b_step = b_ld < 0 ? 0u : 1u;
    if (b_ld < 0) b_ld = 0;
    hipLaunchKernelGGL(binary_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, op, b_step, 1.0f);
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

// ---------------------------------------------------------------------------- fill
// y[m, i] = v over an [M, n] extent with a row pitch, for a rank with nothing to compute whose
// downstream reader (an all-reduce, a residual) still needs a defined buffer. A memset cannot
// handle the pitch or a two-byte value.
// T is the element's bits: uint16_t for bf16, uint32_t for f32.
template<typename T>
__global__ __launch_bounds__(kThreads) void fill_kernel(T* __restrict__ y, uint32_t n,
                                                        uint32_t y_ld, T v) {
    T* __restrict__ yr = y + (size_t)blockIdx.x * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) yr[i] = v;
}

// Host form of r4d_common.h's f32_to_bf16 (which is __device__), round-to-nearest-even.
static inline uint16_t f32_to_bf16_host(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

extern "C" int r4d_fill_bf16(long y, int M, int n, int y_ld, float value, long stream) {
    if (!y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(fill_kernel<uint16_t>, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream, (uint16_t*)y, (uint32_t)n, (uint32_t)y_ld,
                       f32_to_bf16_host(value));
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_fill_f32(long y, int M, int n, int y_ld, float value, long stream) {
    if (!y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!y_ld) y_ld = n;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    hipLaunchKernelGGL(fill_kernel<uint32_t>, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream, (uint32_t*)y, (uint32_t)n, (uint32_t)y_ld, bits);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_logit_softcap_bf16(long x, long y, int M, int n, float cap,
                                      int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!(cap > 0.0f)) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(logit_softcap_kernel, dim3((uint32_t)M, softcap_blocks_per_row(n)),
                       dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n, cap,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_logit_softcap_f32(long x, long y, int M, int n, float cap,
                                     int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!(cap > 0.0f)) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(logit_softcap_kernel_f32, dim3((uint32_t)M, softcap_blocks_per_row(n)),
                       dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const float*)x, (float*)y, (uint32_t)n, cap,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_scale_bf16(long x, long s, uint32_t s_dt, long y, int M, int n,
                              int x_ld, int y_ld, long stream) {
    if (!x || !s || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (s_dt != RAD_F32 && s_dt != RAD_BF16) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(scale_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)x, (const void*)s, s_dt, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_silu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream) {
    if (!gu || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    const uint32_t chunks = ((uint32_t)n + kSiluPer * kThreads - 1u) / (kSiluPer * kThreads);
    hipLaunchKernelGGL(silu_mul_kernel, dim3((uint32_t)M, chunks), dim3(kThreads), 0,
                       (hipStream_t)stream, (const uint16_t*)gu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// GELU-gated product, the gemma FFN activation. Same shape and gate order as silu_mul; tanh GELU to
// match ref_gelu_mul's act_gelu_tanh.
extern "C" int r4d_gelu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream) {
    if (!gu || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(gelu_mul_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)gu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `n_rope_heads` is n_head + n_head_kv: the Q heads and the K heads of a fused projection, or just
// the one buffer's heads when q and k are separate tensors (which is what the gated block does).
extern "C" int r4d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                             int rot, int gptj, float theta, float rscale, int row_pitch,
                             long stream) {
    return r4d_rope_mc_bf16(qkv, pos, M, head_dim, n_rope_heads, rot, rot, gptj, theta, rscale,
                            row_pitch, 0, nullptr, 0, 0, stream);
}

// The single launch of rope_kernel. `yarn` with factor 0 is the plain rotary (the YaRN blend
// returns the frequency unchanged and no magnitude is applied), and so is `l3` with factor 0 (Llama
// 3's scaling, after YaRN's, which no caller combines with it). `den` is the ladder width (`rot`
// for `rotary`, `head_dim` for `head`); the plain entry passes `rot`.
static int rope_launch(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int den,
                       int gptj, float theta, float rscale, int row_pitch, int mode,
                       const int* sec, int n_sec, int plane, RadYarn yarn, long stream,
                       int nfreq = -1, RadLlama3 l3 = RadLlama3{}) {
    if (!qkv || !pos) return -1;
    if (M <= 0 || head_dim <= 0 || n_rope_heads <= 0) return -2;
    if (rot <= 0 || rot > head_dim || (rot & 1)) return -3;
    if (n_sec < 0 || n_sec > 3 || (n_sec && !sec)) return -3;
    if (!row_pitch) row_pitch = n_rope_heads * head_dim;
    const uint32_t pairs = (uint32_t)n_rope_heads * (uint32_t)(rot / 2);
    const uint32_t gy = (pairs + kThreads - 1) / kThreads;
    const int s0 = n_sec > 0 ? sec[0] : 0, s1 = n_sec > 1 ? sec[1] : 0, s2 = n_sec > 2 ? sec[2] : 0;
    if (nfreq < 0) nfreq = rot / 2;
    hipLaunchKernelGGL(rope_kernel, dim3((uint32_t)M, gy), dim3(kThreads), 0, (hipStream_t)stream,
                       (uint16_t*)qkv, (const int32_t*)pos, (uint32_t)head_dim,
                       (uint32_t)n_rope_heads, (uint32_t)rot, (uint32_t)nfreq, (int)den, gptj, theta,
                       rscale,
                       (uint32_t)row_pitch, mode, s0, s1, s2, n_sec, (uint32_t)plane, yarn, l3);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// The multi-component form: `mode` 2 mrope, 3 imrope, 4 axial, `sec[0..n_sec)` the sections and
// `plane` the component pitch of `pos`; plane 0 is the one-component rotary above.
extern "C" int r4d_rope_mc_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                                int rot, int den, int gptj, float theta, float rscale,
                                int row_pitch, int mode, const int* sec, int n_sec, int plane,
                                long stream) {
    return rope_launch(qkv, pos, M, head_dim, n_rope_heads, rot, den, gptj, theta, rscale,
                       row_pitch, mode, sec, n_sec, plane, RadYarn{}, stream);
}

extern "C" int r4d_kv_store_bf16(long k, long v, long slot, long kv, int M, int head_dim,
                                 int kv_heads, int block_size, int n_blocks,
                                 int k_ld, int v_ld, long stream) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    /* The fp8 cache is a separate entry point (r4d_kv_store_fp8), selected by --kv-cache-dtype. */
    hipLaunchKernelGGL(kv_store_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                       (uint16_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                       (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// bf16 k and v in, E4M3 cache out. `k_scale` and `v_scale` are the reader's descales, so this
// divides by them; 1.0 is the vLLM default and what the attention kernels assume when their
// descale pointers are null.
static int kv_store_fp8_emit(long k, long v, long slot, long kv, int M, int head_dim,
                             int kv_heads, int block_size, int n_blocks,
                             int k_ld, int v_ld, float k_scale, float v_scale,
                             RadStream stream) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    // Four columns per thread, so a head must be whole groups or a group would straddle two heads.
    if (head_dim % 4) return -3;
    if (!(k_scale > 0.0f) || !(v_scale > 0.0f)) return -5;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    return R4D_EMIT(stream, kv_store_fp8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                    (uint8_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                    (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld,
                    1.0f / k_scale, 1.0f / v_scale) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_kv_store_fp8(long k, long v, long slot, long kv, int M, int head_dim,
                                int kv_heads, int block_size, int n_blocks,
                                int k_ld, int v_ld, float k_scale, float v_scale, long stream) {
    return kv_store_fp8_emit(k, v, slot, kv, M, head_dim, kv_heads, block_size, n_blocks,
                             k_ld, v_ld, k_scale, v_scale, (RadStream)(uintptr_t)stream);
}

// ================================================================================ logits_gemm
//
// The output is f32, which is why this kernel exists: `logits_gemm` is r4d_gemm_bf16_nt_m16's
// arithmetic, but that kernel writes bf16 and the engine's logits buffer is f32 (eight mantissa
// bits are too few for nucleus cuts between neighbouring probabilities).
//
// One workgroup per vocabulary column, 256 threads striding K. At n_vocab 248320 and K 5120 the
// 2.5 GiB weight read per step is the cost. Untuned: one correct variant.
//
// Rows are processed in chunks of kLgRows so accumulators stay in registers. M is one row per
// sampled position (`gather_rows` runs first), so one chunk usually covers it.

namespace {

constexpr uint32_t kLgRows = 8;    // rows held in registers at once
constexpr uint32_t kLgVec  = 8;    // bf16 per 128-bit load

__global__ __launch_bounds__(kThreads) void logits_gemm_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ w, float* __restrict__ c,
    uint32_t M, uint32_t K, uint32_t N, uint32_t a_ld, uint32_t c_ld) {
    __shared__ float lds[kWaves][kLgRows];
    const uint32_t j = blockIdx.x;
    if (j >= N) return;
    const uint16_t* __restrict__ wj = w + (size_t)j * K;

    for (uint32_t r0 = 0; r0 < M; r0 += kLgRows) {
        const uint32_t nr = (M - r0) < kLgRows ? (M - r0) : kLgRows;
        float acc[kLgRows];
#pragma unroll
        for (uint32_t i = 0; i < kLgRows; ++i) acc[i] = 0.0f;

        for (uint32_t k = threadIdx.x * kLgVec; k < K; k += kThreads * kLgVec) {
            uint16_t wv[kLgVec];
#pragma unroll
            for (uint32_t p = 0; p < kLgVec; ++p) wv[p] = wj[k + p];
            for (uint32_t i = 0; i < nr; ++i) {
                const uint16_t* __restrict__ ar = a + (size_t)(r0 + i) * a_ld + k;
                float s = 0.0f;
#pragma unroll
                for (uint32_t p = 0; p < kLgVec; ++p) s += bf16_to_f32(ar[p]) * bf16_to_f32(wv[p]);
                acc[i] += s;
            }
        }

        for (uint32_t i = 0; i < nr; ++i) {
            float v = acc[i];
            v = r4d_wave_sum(v);
            const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
            if (lane == 0) lds[wave][i] = v;
        }
        __syncthreads();
        if (threadIdx.x < nr) {
            float v = 0.0f;
#pragma unroll
            for (uint32_t q = 0; q < kWaves; ++q) v += lds[q][threadIdx.x];
            c[(size_t)(r0 + threadIdx.x) * c_ld + j] = v;
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
    hipLaunchKernelGGL(logits_gemm_kernel, dim3((uint32_t)N), dim3(kThreads), 0,
                       (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)w, (float*)c,
                       (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)c_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// Unpacking RadArgs and deriving row pitches belongs to the kernel library (libr4d/r4d_args.h).
// The kernels take raw pointers and one pitch, so anything unpitched must be tight (spec 2.1).
// Before each launch an op checks:
//
//   * the dtype, from the operand itself: a row's `dtype` constraint is a necessary condition on
//     the geometry, not a promise about the tensors, and an f32 operand read as bf16 is garbage.
//   * the row pitch: r4d_rows_pitch() returns elements, or -1 (RAD_E_STRIDE) for a strided inner
//     axis.
//   * the row count, from the operand, never the `M` parameter: M is ranged, so the args hold the
//     band's upper bound (libref/ref_ops.h).
/* bf16 is the only activation width these kernels read. */
static inline int dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

/* Opaque bytes: KV tier traffic, whose contents are whatever the attention kernel wrote. */
static inline int dt_u8(const RadTensor* t) {
    return (t && (t->dtype == RAD_U8 || t->dtype == RAD_I8)) ? RAD_OK : RAD_E_DTYPE;
}

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

/* ------------------------------------------------------------------ embed_lookup */

enum { EL_TOK = 0, EL_WTE, EL_X };

extern "C" int r4d_rad_embed_lookup(const RadArgs* a, RadStream stream) {
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

    const float mult = (float)r4d_getf(a, "mult", 1.0);
    return embed_lookup_bf16_emit(r4d_addr(tok), r4d_addr(wte), r4d_addr(x),
                                  (int)M, (int)n_embd, (int)n_vocab, (int)voff, (int)x_ld,
                                  mult, stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ embed_lookup_q */

/* Operand 4, `ahead`, is not read: it carries upcoming prompt ids for a host reader that
 * prefetches table rows from a file, and a card reads its table in place. */
enum { ELQ_TOK = 0, ELQ_WTE, ELQ_SCALE, ELQ_X };

extern "C" int r4d_rad_embed_lookup_q(const RadArgs* a, RadStream stream) {
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

    /* The cached path needs the model's row width, a 16-byte output row and the eight-byte table
     * read its lanes do; anything else (and the default, where no cache was allocated) takes the
     * generic kernel above. */
    PleVramCache* cache = (PleVramCache*)a->instance;
    if (cache && cache->sets && q && n_embd == (long long)kPleRow && x_ld >= (long long)kPleRow &&
        x_ld % 8 == 0 && M <= 0x7fffffffLL && r4d_aligned(x->data, 16) &&
        r4d_aligned(wte->data, 8) && ple_cache_binds(cache, wte->data))
        return embed_lookup_q_cached_emit((const int32_t*)r4d_addr(tok), (const uint8_t*)r4d_addr(wte),
                                          sc ? (const void*)r4d_addr(sc) : nullptr,
                                          sc && sc->dtype == RAD_F32, (uint16_t*)r4d_addr(x), M,
                                          (int)n_vocab, (int)voff, x_ld, cache, stream) == 0
                   ? RAD_OK : RAD_E_DEVICE;

    return embed_lookup_q_emit((const int32_t*)r4d_addr(tok), (const void*)r4d_addr(wte), q,
                               sc ? (const void*)r4d_addr(sc) : nullptr,
                               sc && sc->dtype == RAD_F32, (uint16_t*)r4d_addr(x), M,
                               (int)n_embd, (int)n_vocab, (int)voff, x_ld, stream) == 0
               ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ gather_rows */

enum { GR_X = 0, GR_IDX, GR_Y };

extern "C" int r4d_rad_gather_rows(const RadArgs* a, RadStream stream) {
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
                                 stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ byte rows */
/* The u8 forms. `n` is a byte count, so r4d_rows_pitch's pitches are byte pitches. */
extern "C" int r4d_rad_gather_rows_u8(const RadArgs* a, RadStream stream) {
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
                               stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* Operands (v, idx, x), x inout: rows the call does not name are left unchanged. */
enum { SR_V = 0, SR_IDX, SR_X };

extern "C" int r4d_rad_scatter_rows_u8(const RadArgs* a, RadStream stream) {
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
                                stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* The same copy for rows declared bf16 (e.g. an encoder's rows over a language model's
 * embeddings): a bf16 row of n is 2n bytes. */
extern "C" int r4d_rad_scatter_rows_bf16(const RadArgs* a, RadStream stream) {
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
                                stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ cast (bf16 -> bf16) */
/* A strided row copy, which is what `cast` reduces to when both ends are the same dtype. A
 * drafter conditioned on the trunk's intermediate hidden states must copy rows out of the residual
 * stream before the next layer overwrites them, and ref's host implementation is refused in a
 * serve. libr4d's 2D copy takes byte pitches, so taps can land side by side as column slices of
 * one [T, 5 * n_embd] row. */
enum { CS_X = 0, CS_Y };

extern "C" int r4d_rad_cast(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, CS_X);
    const RadTensor* y = r4d_opt(a, CS_Y);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    /* A copy at the dtype both sides share: bf16, or f32 (a latent kept aside and put back). */
    if (x->dtype != y->dtype || (x->dtype != RAD_BF16 && x->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const long long es = x->dtype == RAD_F32 ? 4 : 2;

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* The row count is the destination's: a tap's destination is sized for the step's ceiling
     * while the source holds the live rows, so the source need only have at least as many. */
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_p2p_copy2d(r4d_addr(y), r4d_addr(x), y_ld * es, x_ld * es, n * es, M,
                          (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rmsnorm */

enum { RN_X = 0, RN_W, RN_Y };

/* The norm over f32 rows into bf16 (a video decoder's f32 residual stream), at the bf16 form's
 * width so the two fold a row's sum of squares alike. */
static int rmsnorm_x_f32(const float* x, const void* w, uint16_t* y, int M, int n, float eps,
                         float wadd, float mult, int w_f32, int x_ld, int y_ld, RadStream stream) {
    const int thr = r4d_act_threads(M, n);
#define R4D_RNF_T(wf, T)                                                                        \
    hipLaunchKernelGGL((rmsnorm_kernel<wf, T, float>), dim3((uint32_t)M), dim3(T), 0,          \
                       (hipStream_t)stream, x, w, y, (uint32_t)n, eps, wadd, mult,              \
                       (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M)
#define R4D_RNF(wf)                                                                             \
    do { if (thr == 1024)     R4D_RNF_T(wf, 1024);                                              \
         else if (thr == 512) R4D_RNF_T(wf, 512);                                               \
         else                 R4D_RNF_T(wf, 256); } while (0)
    if (w_f32) R4D_RNF(true);
    else       R4D_RNF(false);
#undef R4D_RNF
#undef R4D_RNF_T
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, RN_X);
    const RadTensor* w = r4d_opt(a, RN_W);
    const RadTensor* y = r4d_opt(a, RN_Y);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    /* The row bf16, or f32 (read as it is); the output bf16. */
    if (x->dtype != RAD_BF16 && x->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    R4D_NEED(dt_bf16(y));
    const bool x_f32 = x->dtype == RAD_F32;
    /* An absent gain is one (Gemma 4's value norm, its multimodal embedder's pre-projection norm).
     * `wadd` without a gain is refused. */
    if (!w) {
        if (x_f32) return R4D_NO(RAD_E_DTYPE);
        const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
        if (n <= 0) return R4D_NO(RAD_E_SHAPE);
        if (r4d_getf(a, "wadd", 0.0) != 0.0) return R4D_NO(RAD_E_INVAL);
        long long M = 0, My = 0;
        const long long x_ld = r4d_rows_pitch(x, n, &M);
        const long long y_ld = r4d_rows_pitch(y, n, &My);
        if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
        return r4d_rmsnorm_mult_bf16(r4d_addr(x), 0, r4d_addr(y), (int)M, (int)n,
                                (float)r4d_getf(a, "eps", 1e-6), 0.0f,
                                (float)r4d_getf(a, "mult", 1.0), 0, (int)x_ld, (int)y_ld,
                                (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
    }
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    /* The gain may be f32 (as plugins declare norm gains) or bf16; its width is read off the
     * operand rather than assumed. */
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
    if (x_f32)
        return rmsnorm_x_f32((const float*)x->data, w->data, (uint16_t*)y->data, (int)M, (int)n,
                             eps, wadd, (float)r4d_getf(a, "mult", 1.0), w_f32, (int)x_ld,
                             (int)y_ld, s);
    return r4d_rmsnorm_mult_bf16(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n, eps, wadd,
                            (float)r4d_getf(a, "mult", 1.0),
                            w_f32, (int)x_ld, (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                                 : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ add_rmsnorm */

enum { AN_X = 0, AN_A, AN_W, AN_Y };

extern "C" int r4d_rad_add_rmsnorm(const RadArgs* a, RadStream s) {
    int emit_rc = RAD_OK;
    const RadTensor* x  = r4d_opt(a, AN_X);
    const RadTensor* ad = r4d_opt(a, AN_A);
    const RadTensor* w  = r4d_opt(a, AN_W);
    const RadTensor* y  = r4d_opt(a, AN_Y);
    if (!x || !ad || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(ad));
    R4D_NEED(dt_bf16(y));
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const bool w_f32 = w->dtype == RAD_F32;
    const long long n = r4d_geti_or(a, "n", r4d_dim(w, (int)w->rank - 1));
    if (n <= 0 || r4d_numel(w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long a_ld = r4d_rows_pitch(ad, n, &Ma);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || a_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || My < M) return R4D_NO(RAD_E_SHAPE);
    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    /* rmsnorm's width, from the same table (see r4d_rmsnorm_mult_bf16) */
    const int thr = r4d_act_threads((int)M, (int)n);
#define R4D_AN_T(wf, T)                                                                         \
    emit_rc = R4D_EMIT(s, (add_rmsnorm_kernel<wf, T>),                                          \
                       dim3(r4d_cliff_rows((uint32_t)M, 1, T / kWave)), dim3(T), 0,              \
                       (uint16_t*)r4d_wp(x), (const uint16_t*)r4d_p(ad), r4d_p(w),              \
                       (uint16_t*)r4d_wp(y), (uint32_t)n, eps, wadd, mult, (uint32_t)x_ld,       \
                       (uint32_t)a_ld, (uint32_t)y_ld, (uint32_t)M)
#define R4D_AN(wf)                                                                              \
    do { if (thr == 1024)     R4D_AN_T(wf, 1024);                                               \
         else if (thr == 512) R4D_AN_T(wf, 512);                                                \
         else                 R4D_AN_T(wf, 256); } while (0)
    if (w_f32) R4D_AN(true);
    else       R4D_AN(false);
#undef R4D_AN
#undef R4D_AN_T
    return emit_rc;
}

/* ------------------------------------------------------------------ rmsnorm_p */

extern "C" int r4d_rad_rmsnorm_p(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(x, (int)x->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const float eps = (float)r4d_getf(a, "eps", 1e-6);
    const int rc = r4d_rmsnorm_p_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, eps,
                                      (int)x_ld, (int)y_ld, (long)(uintptr_t)s);
    return rc == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ add / mul */

enum { BI_A = 0, BI_B, BI_Y };

/* Both rows of each op share this: every operand bf16, or (add_f32) every operand f32. */
static int binary(const RadArgs* a, int mul, RadStream s) {
    const RadTensor* A = r4d_opt(a, BI_A);
    const RadTensor* B = r4d_opt(a, BI_B);
    const RadTensor* Y = r4d_opt(a, BI_Y);
    if (!A || !B || !Y) return R4D_NO(RAD_E_INVAL);
    const bool f32 = Y->dtype == RAD_F32;
    const uint32_t dt = f32 ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16;
    if (A->dtype != dt || B->dtype != dt || Y->dtype != dt) return R4D_NO(RAD_E_DTYPE);
    /* `div` is add's alone (mul's schema has none); 0 and non-finite values are refused, as libref
     * refuses them. */
    const double div = mul ? 1.0 : r4d_getf(a, "div", 1.0);
    if (!(div != 0.0) || !std::isfinite(div)) return R4D_NO(RAD_E_INVAL);

    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long a_ld = r4d_rows_pitch(A, n, &Ma);
    if (y_ld < 0 || a_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M) return R4D_NO(RAD_E_SHAPE);

    /* The one broadcast rule: exactly n elements against M rows broadcasts along rows (pitch 0).
     * Anything between n and M*n is a shape bug and is refused. */
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

    if (f32) {
        const uint64_t total = (uint64_t)M * (uint64_t)n;
        const uint64_t blocks = std::min<uint64_t>((total + kThreads - 1) / kThreads, 65536u);
        hipLaunchKernelGGL(binary_f32_kernel, dim3((uint32_t)blocks), dim3(kThreads), 0,
                           (hipStream_t)s, (const float*)A->data, (const float*)B->data,
                           (float*)Y->data, (uint32_t)n, (uint32_t)a_ld, (uint32_t)b_ld,
                           (uint32_t)y_ld, mul, 1u, (float)div, total);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (div != 1.0) {
        hipLaunchKernelGGL(binary_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)s,
                           (const uint16_t*)A->data, (const uint16_t*)B->data, (uint16_t*)Y->data,
                           (uint32_t)n, (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, mul, 1u,
                           (float)div);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    return r4d_binary_bf16(r4d_addr(A), r4d_addr(B), r4d_addr(Y), (int)M, (int)n, mul,
                           (int)a_ld, (int)b_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_add(const RadArgs* a, RadStream s) { return binary(a, 0, s); }
extern "C" int r4d_rad_mul(const RadArgs* a, RadStream s) { return binary(a, 1, s); }

/* mul_weight: y = x * w, `w` a bf16 weight of one element (pitch -1, a scalar) or one row (pitch
 * 0, broadcast down the rows). The f32 row (mul_weight_f32) writes an f32 `y` from a bf16 or f32
 * `x`; the bf16 row a bf16 `y` from a bf16 `x`. */
extern "C" int r4d_rad_mul_weight(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* W = r4d_opt(a, 1);
    const RadTensor* Y = r4d_opt(a, 2);
    if (!X || !W || !Y) return R4D_NO(RAD_E_INVAL);
    const bool y_f32 = Y->dtype == RAD_F32;
    if (y_f32 ? (X->dtype != RAD_BF16 && X->dtype != RAD_F32) : X->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    /* The weight bf16 or f32 (a layer scale a model keeps f32; libref reads either). */
    if (W->dtype != RAD_BF16 && W->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!y_f32 && Y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    long long w_ld;
    if (r4d_numel(W) == 1) w_ld = -1;
    else if (r4d_numel(W) == n && r4d_contig(W)) w_ld = 0;
    else return R4D_NO(RAD_E_SHAPE);
    if (y_f32) {
        const uint64_t total = (uint64_t)M * (uint64_t)n;
        const dim3 grid((uint32_t)std::min<uint64_t>((total + kThreads - 1) / kThreads, 65536u));
        const uint32_t ws = w_ld < 0 ? 0u : 1u;
        const bool xf = X->dtype == RAD_F32, wf = W->dtype == RAD_F32;
#define R4D_MWF(TX, TW)                                                                          \
        hipLaunchKernelGGL((mul_weight_f32_kernel<TX, TW>), grid, dim3(kThreads), 0,             \
                           (hipStream_t)s, (const TX*)X->data, (const TW*)W->data,               \
                           (float*)Y->data, (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, ws,     \
                           total)
        if (xf && wf)      R4D_MWF(float, float);
        else if (xf)       R4D_MWF(float, uint16_t);
        else if (wf)       R4D_MWF(uint16_t, float);
        else               R4D_MWF(uint16_t, uint16_t);
#undef R4D_MWF
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (W->dtype == RAD_F32) {
        hipLaunchKernelGGL(mul_wf32_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)s,
                           (const uint16_t*)X->data, (const float*)W->data, (uint16_t*)Y->data,
                           (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, w_ld < 0 ? 0u : 1u);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    return r4d_binary_bf16(r4d_addr(X), r4d_addr(W), r4d_addr(Y), (int)M, (int)n, 1,
                           (int)x_ld, (int)w_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ sigmoid */

extern "C" int r4d_rad_sigmoid(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    /* Row-wise with a pitch: the input is usually a column slice (the gate half of an interleaved
     * [tokens, heads, 2*head_dim] projection, `n` wide inside a stride of 2n). */
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

/* ------------------------------------------------------------------ fill */

extern "C" int r4d_rad_fill(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    if (!y) return R4D_NO(RAD_E_INVAL);
    if (y->dtype != RAD_BF16 && y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const double v = r4d_getf(a, "value", 0.0);
    const int e = (y->dtype == RAD_F32 ? r4d_fill_f32 : r4d_fill_bf16)(
        r4d_addr(y), (int)M, (int)n, (int)y_ld, (float)v, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ logit_softcap */

extern "C" int r4d_rad_logit_softcap(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    /* Two legs: bf16 logits, or f32 logits as the gemm hands the sampler. */
    const bool leg_bf16 = (x->dtype == RAD_BF16) && (y->dtype == RAD_BF16);
    const bool leg_f32  = (x->dtype == RAD_F32)  && (y->dtype == RAD_F32);
    if (!leg_bf16 && !leg_f32) return R4D_NO(RAD_E_DTYPE);

    /* Gemma 4 issues cap 30.0; the default serves direct calls. A zero cap would divide by zero
     * in every lane. */
    const float cap = (float)r4d_getf(a, "cap", 30.0);
    if (!(cap > 0.0f)) return R4D_NO(RAD_E_INVAL);

    /* Row-wise with a pitch, as sigmoid. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const int rc = leg_bf16
        ? r4d_logit_softcap_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, cap,
                                 (int)x_ld, (int)y_ld, (long)(uintptr_t)s)
        : r4d_logit_softcap_f32(r4d_addr(x), r4d_addr(y), (int)M, (int)n, cap,
                                (int)x_ld, (int)y_ld, (long)(uintptr_t)s);
    return rc == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ scale */

extern "C" int r4d_rad_scale(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* sc = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !sc || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    /* `s` is a [1] weight, f32 or bf16. An empty scalar is a shape error, not a zero. It is read
     * on the device, once per block, to avoid a device-to-host sync on the step path. */
    if (sc->dtype != RAD_F32 && sc->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(sc) < 1) return R4D_NO(RAD_E_SHAPE);

    /* Row-wise with a pitch, as sigmoid. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_scale_bf16(r4d_addr(x), r4d_addr(sc), sc->dtype, r4d_addr(y), (int)M, (int)n,
                          (int)x_ld, (int)y_ld,
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

/* ------------------------------------------------------------------ gelu_mul */

extern "C" int r4d_rad_gelu_mul(const RadArgs* a, RadStream s) {
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

    const int rc = r4d_gelu_mul_bf16(r4d_addr(gu), r4d_addr(y), (int)M, (int)n, (int)gu_ld, (int)y_ld,
                                     (long)(uintptr_t)s);
    return rc == 0 ? RAD_OK : RAD_E_DEVICE;
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

    /* `ladder`: `rotary` (default) measures the base ladder in `rot`; `head` in `head_dim` (the
     * proportional variant). */
    const char* lad = r4d_gets(a, "ladder");
    long long den = rot;
    if (lad) {
        if (!std::strcmp(lad, "head")) den = head_dim;
        else if (std::strcmp(lad, "rotary")) return R4D_NO(RAD_E_UNSUPPORTED);
    }
    /* `head` is HF's proportional RoPE: the rotation spans the whole head (pair i is
     * (i, i + head_dim/2)) and only the first rotary_dim/2 frequencies are non-zero. Pairing
     * (i, i + rot/2) instead turns the wrong channels (Gemma 4's global layers, 128 of 512) and
     * degrades long-range attention. */

    /* The mode sets how many components a position has. With [M] positions, mrope and imrope are
     * neox. With [C, M] positions and `sections`, each frequency reads its section's component
     * (docs/OPS.md, `rope`). */
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
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn)) return R4D_NO(RAD_E_INVAL);
    /* No model uses YaRN with the axial vision rotary or a linear position divide. */
    if (yarn.factor != 0.0f && (mc == 4 || rscale != 1.0f)) return R4D_NO(RAD_E_UNSUPPORTED);
    /* Llama 3's scaling: one-component positions only, and never beside YaRN (libref refuses
     * both alike). */
    const RadLlama3 l3 = rad_llama3_args(a);
    if (!rad_llama3_valid(l3) || (l3.factor != 0.0f && yarn.factor != 0.0f))
        return R4D_NO(RAD_E_INVAL);
    if (l3.factor != 0.0f && (mc == 4 || multi)) return R4D_NO(RAD_E_UNSUPPORTED);
    /* The Q and K heads rotate; the trailing V heads do not. */
    long long rot_l = rot, nfreq = rot / 2;
    if (den == head_dim && rot < head_dim && !multi && !gptj && mc == 0) rot_l = head_dim;
    return rope_launch(r4d_addr(qkv), r4d_addr(pos), (int)Mp, (int)head_dim,
                       (int)(n_head + n_head_kv), (int)rot_l, (int)den, gptj, theta, rscale,
                       (int)pitch, mc, sec, n_sec, plane, yarn, (long)(uintptr_t)s,
                       (int)nfreq, l3) == 0
        ? RAD_OK
        : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ kv_store */

enum { KS_K = 0, KS_V, KS_SLOT, KS_CACHE };

/* Geometry shared by both stores, which differ only in the cache dtype, so the paged-layout check
 * exists once. */
struct KvStoreGeom {
    const RadTensor *k, *v, *sm, *kv;
    long long head_dim, block_size, kv_heads, n_blocks, k_ld, v_ld, T;
};

static int kv_store_geom(const RadArgs* a, KvStoreGeom* g, long long row_per_hd16 = 32) {
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
    /* The shared paged layout is [n_blocks, kv_heads, block_size, 2*head_dim]; refuse a cache that
     * disagrees. */
    if (g->kv->rank != 4) return R4D_NO(RAD_E_SHAPE);
    if (g->kv->shape[1] != g->kv_heads || g->kv->shape[2] != g->block_size ||
        g->kv->shape[3] != g->head_dim * row_per_hd16 / 16) return R4D_NO(RAD_E_SHAPE);
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

/* The E4M3 store. `k_scale` and `v_scale` are the attention side's descales (one per sequence and
 * head there, a constant here); 1.0 is vLLM's default and the fp8kv kernels' assumption when the
 * descale operands are absent. */
extern "C" int r4d_rad_kv_store_fp8(const RadArgs* a, RadStream stream) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    /* Four columns per thread, so a head must be whole groups. */
    if (g.head_dim % 4) return R4D_NO(RAD_E_SHAPE);

    const double ks = r4d_getf(a, "k_scale", 1.0);
    const double vs = r4d_getf(a, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);

    return kv_store_fp8_emit(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                             (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                             (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld,
                             (float)ks, (float)vs, stream) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* The fp8_v4 and k6_v4 stores: the cache's last extent is a token's row, 25 or 22 * head_dim / 16
 * bytes. */
static int kv_store_q4(const RadArgs* a, RadStream stream, int k6) {
    const long long r16 = k6 ? 22 : 25;
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g, r16);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    if (g.head_dim % 32) return R4D_NO(RAD_E_SHAPE);
    const double ks = r4d_getf(a, "k_scale", 1.0);
    const double vs = r4d_getf(a, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);
    long long k_ld = g.k_ld ? g.k_ld : g.kv_heads * g.head_dim;
    long long v_ld = g.v_ld ? g.v_ld : g.kv_heads * g.head_dim;
    if (k6)
        return R4D_EMIT(stream, kv_store_fp8v4_kernel<1>, dim3((uint32_t)g.T), dim3(kThreads), 0,
                        (const uint16_t*)r4d_addr(g.k), (const uint16_t*)r4d_addr(g.v),
                        (const int32_t*)r4d_addr(g.sm), (uint8_t*)r4d_addr(g.kv),
                        (uint32_t)g.head_dim, (uint32_t)g.kv_heads, (uint32_t)g.block_size,
                        (uint32_t)g.n_blocks, (uint32_t)k_ld, (uint32_t)v_ld, (float)(1.0 / ks),
                        (float)(1.0 / vs), (uint32_t)(g.head_dim * r16 / 16)) == RAD_OK ? RAD_OK : RAD_E_DEVICE;
    return R4D_EMIT(stream, kv_store_fp8v4_kernel<0>, dim3((uint32_t)g.T), dim3(kThreads), 0,
                    (const uint16_t*)r4d_addr(g.k), (const uint16_t*)r4d_addr(g.v),
                    (const int32_t*)r4d_addr(g.sm), (uint8_t*)r4d_addr(g.kv),
                    (uint32_t)g.head_dim, (uint32_t)g.kv_heads, (uint32_t)g.block_size,
                    (uint32_t)g.n_blocks, (uint32_t)k_ld, (uint32_t)v_ld, (float)(1.0 / ks),
                    (float)(1.0 / vs), (uint32_t)(g.head_dim * r16 / 16)) == RAD_OK ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r4d_rad_kv_store_fp8v4(const RadArgs* a, RadStream stream) { return kv_store_q4(a, stream, 0); }
extern "C" int r4d_rad_kv_store_k6v4(const RadArgs* a, RadStream stream) { return kv_store_q4(a, stream, 1); }

/* ------------------------------------------------------------------ logits_gemm
 *
 * gemm_nt's arithmetic with `n_vocab`/`n_embd` for N/K, but writing f32 (see the kernel above).
 * M is one row per sampled position: `gather_rows` runs first, since ungathered logits at
 * max_tok 8192 and a 248K vocabulary would be 8 GiB. */

enum { LG_X = 0, LG_W, LG_Y };

extern "C" int r4d_rad_logits_gemm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(w));
    /* f32 output, checked. */
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
