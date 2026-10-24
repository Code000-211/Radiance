// r4d_fused_quant_fp8.hip - the two producers that already have an activation row in registers,
// with r4d_quant_act_fp8 riding along on the same pass.
//
// WHY FUSE, AND WHY THESE TWO.  The fp8a8 GEMMs need an E4M3 activation with an f32 scale per
// (row, 128 columns) in front of EVERY quantised linear, and `activation_scheme: dynamic` means
// there is no shipped scale to skip it with.  A decode step of this model issues ~300 of those
// quantisers -- one per activation per layer, on both ranks -- each costing microseconds on a step
// budgeted in tens of milliseconds.  That is not bandwidth: the row is 8 tokens by 5120 channels,
// 82 KiB, which is 0.13 us of a 638 GB/s card.  It is a launch, a grid fill and a second pass over
// a row some other kernel had in registers a moment earlier.
//
// Two of the three producers hand the quantiser a row they have just computed:
//
//   rmsnorm -> quant   the input of every attention, GDN and MLP linear.  ~128 a step.
//   silu_mul -> quant  down_proj's input.  ~64 a step.
//
// The third (a GEMM's own output feeding the next linear) is not fusable this way and stays a
// launch of its own.  The int8 path carries the same fusion -- r4d_rmsnorm_had_quant_i8 and
// r4d_gated_had_quant_i8 -- and this is the fp8 form of it, minus the rotation, which is what the
// int8 grid needs to survive one scale per row and a 128-wide block scale does not.
//
// BIT-IDENTICAL TO THE PAIR IT REPLACES, and that is a hard requirement rather than a nicety,
// because the engine's generated text is compared byte for byte across builds.  The unfused pair
// writes a BF16 norm output and the quantiser then reads those bf16 bytes back, so the amax and
// the codes are taken over
// bf16-ROUNDED values.  Both kernels here round to bf16 first and quantise the round-tripped float,
// which costs one v_cvt per element and is the difference between "the same text" and "very nearly
// the same text".
//
// THE NORM RE-READS ITS ROW FOR THE SECOND PASS rather than keeping it in registers or LDS, and
// that stays true WITH the residual, which is where this differs from r4d_rmsnorm_had_quant_i8.
// That kernel keeps the row because it normalises the f32 sum, and re-reading would normalise a
// bf16-rounded one, which costs thousands of +/-1 codes.  This one rounds the sum to bf16 and
// stores it FIRST, because that is what the pair it replaces does: `add` writes a bf16 residual and
// `rmsnorm` reads those bytes back.  So the value normalised here is the value normalised there,
// the re-read is L1-hot, and the output does not move.  Less accurate than the int8 path's choice,
// and bit-identical to the pair it replaces, which is the property that matters here.
//
// THE RESIDUAL MAY ALIAS `out_bf16`, and every caller in this tree makes it do so: a block reads
// the PREVIOUS block's delta out of the shared `h` buffer and writes its own norm output back into
// it.  Pass one reads every `x` element before the block_sum barrier and pass two writes `y` only
// after it, so the two never race.
//
// `wadd` IS ADDED TO EVERY GAIN IN f32, for the reason r4d_rmsnorm_had_quant_i8 gives at length:
// Qwen3.5's norms are GemmaRMSNorm over a zero-centred gamma, and forming `1 + w` in bf16 first is
// a systematic 0.3% per channel.  Pass 0 for a plain RMS norm.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_ar_seq.h"   // the all-reduce sequence step the fused all-reduce + norm takes
#include "r4d_fp8_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"   // r4d_act_threads: the workgroup width, shared with the int8 path
#include "r4d_fwht.h"     // the in-register rotation the `_had_` forms put before the absmax
#include "r4d_q8.h"       // the 8-bit codes: E4M3, or int8 for an int8 trunk linear

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kGroup = 128;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// Round-to-nearest-even, the same three lines every bf16 store in this library uses.
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Both reductions exchange through DPP and permlanex16 (r4d_xor_lane, r4d_wave_sum) rather than
// `__shfl_xor`, which gfx1201 lowers to ds_bpermute_b32 -- an LDS round trip a stage, five a
// reduction, with nothing else in flight on the wave to hide them. Same partners in the same order,
// so the same bits.
__device__ __forceinline__ float wave_max(float v) {
    v = fmaxf(v, r4d_xor_lane<16>(v));
    v = fmaxf(v, r4d_xor_lane<8>(v));
    v = fmaxf(v, r4d_xor_lane<4>(v));
    v = fmaxf(v, r4d_xor_lane<2>(v));
    v = fmaxf(v, r4d_xor_lane<1>(v));
    return v;
}

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

// One 128-column group's worth of quantisation, given the four bf16-rounded values a lane owns.
// Lifted out because both kernels below end the same way, and because getting the clamp wrong is
// the failure r4d_quant_act_fp8.hip spends a paragraph on: v_cvt_pk_fp8_f32 answers an
// out-of-range input with a NaN rather than saturating, and `amax * (1/scale)` is 448 in exact
// arithmetic but can land an ulp above it in float -- so without the fmin every block containing
// its own maximum produces a NaN.
//
// I8 writes int8 codes at amax / 127 instead (r4d_q8.h), the codes quant_act_i8g would take over
// the same bf16 values: an int8 linear reads them where an fp8 one reads the E4M3 pair.
template <bool I8 = false>
__device__ __forceinline__ void quant_group(const float v[4], float* __restrict__ s,
                                            uint8_t* __restrict__ q, size_t s_at, uint32_t c0,
                                            uint32_t n, uint32_t lane) {
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    m = wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    if (lane == 0) s[s_at] = sc;
    const float inv = 1.0f / sc;
    const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
    // A ragged last group is written zero-padded rather than partially, exactly as
    // r4d_quant_act_fp8 does, and the caller owns `n` rounded up to 4 bytes of `q`.
    if (c0 < n) *(uint32_t*)(q + c0) = pk;
}

// Pass two, shared by every kernel in this file that ends in a block-scaled fp8 activation: give
// each wave a whole 128-column group and each lane four consecutive columns of it, which is what
// makes the group's amax a wave reduction and the E4M3 store a single dword.  `nr` is the row the
// norm was taken OF, already in memory -- rounding it to bf16 and reading it back is the whole
// bit-identity argument at the top of this file, and it is written once here so that the fused
// all-reduce below cannot drift from the standalone norm above it.
template <bool WF32, uint32_t THR>
__device__ __forceinline__ void norm_quant_row(
    const uint16_t* __restrict__ nr, const void* __restrict__ w, uint16_t* __restrict__ yr,
    uint8_t* __restrict__ qr, float* __restrict__ s, size_t s_row, uint32_t n, uint32_t nkb,
    float rs, float wadd) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    for (uint32_t kb = wave; kb < nkb; kb += THR / kWave) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // ONE BRANCH FOR ALL FOUR COLUMNS. Every launcher refuses an `n` that is not a multiple of
        // four, so a lane's columns are all in the row or all past it. Guarded one column at a
        // time, each load sits in its own branch with its use straight after it and the compiler
        // waits for every one before issuing the next; under one branch they all issue first.
        if (c0 < n) {
            float g[4];
            uint16_t x[4];
            const uint64_t* __restrict__ nr64 = (const uint64_t*)(const void*)nr;
            const uint64_t x_raw = nr64[c0 >> 2];
            __builtin_memcpy(x, &x_raw, sizeof(x));

            if constexpr (WF32) {
                const float4* __restrict__ w4 = (const float4*)w;
                const float4 gw = w4[c0 >> 2];
                g[0] = gw.x; g[1] = gw.y; g[2] = gw.z; g[3] = gw.w;
            } else {
                const uint64_t* __restrict__ w64 = (const uint64_t*)w;
                const uint64_t gw_raw = w64[c0 >> 2];
                uint16_t gw[4];
                __builtin_memcpy(gw, &gw_raw, sizeof(gw));
#pragma unroll
                for (int j = 0; j < 4; ++j) g[j] = bf2f(gw[j]);
            }

            uint16_t hb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // THE BF16 ROUND TRIP IS THE POINT.  The unfused quantiser reads these bytes, not
                // the f32 that produced them.
                const uint16_t h = f2bf(bf2f(x[j]) * rs * (g[j] + wadd));
                hb[j] = h;
                v[j] = bf2f(h);
            }
            if (yr) {
                uint64_t* __restrict__ yr64 = (uint64_t*)(void*)yr;
                uint64_t h_raw;
                __builtin_memcpy(&h_raw, hb, sizeof(h_raw));
                yr64[c0 >> 2] = h_raw;
            }
        }
        quant_group(v, s, qr, s_row + kb, c0, n, lane);
    }
}

// ---------------------------------------------------------------------------- rmsnorm -> quant
// One block a row.  Pass one is the sum of squares across all 256 threads; pass two gives each wave
// a whole 128-column group and each lane four consecutive columns of it, which is what makes the
// group's amax a wave reduction and the E4M3 store a single dword.
template <bool WF32, bool RES, uint32_t THR>
__global__ __launch_bounds__(THR) void rmsnorm_quant_fp8_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    uint32_t nkb, float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld,
    uint32_t q_ld, uint32_t s_ld) {
    __shared__ float lds[THR / kWave];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
    // What the norm is taken OF: the residual once there is one, the input otherwise.
    const uint16_t* __restrict__ nr = RES ? rr : xr;

    // FOUR STRIDES OF THE ROW ARE LOADED BEFORE ANY IS USED. A plain `i += THR` loop waits for
    // each stride's load before issuing the next, one round trip per stride; here a stride past
    // the row reloads this thread's first column, which is always in the row, and is skipped
    // below. The sum still takes this thread's columns in the order `i = tid; i += THR` does,
    // which is the order every kernel sharing r4d_act_threads' partition takes them in.
    constexpr uint32_t kU = 4;
    float ss = 0.0f;
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += kU * THR) {
        uint16_t xa[kU], ra[kU];
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR < n ? i0 + u * THR : i0;
            xa[u] = xr[i];
            if (RES) ra[u] = rr[i];
        }
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR;
            if (i < n) {
                float v;
                if (RES) {
                    const uint16_t h = f2bf(bf2f(ra[u]) + bf2f(xa[u]));
                    rr[i] = h;
                    v = bf2f(h);
                } else {
                    v = bf2f(xa[u]);
                }
                ss += v * v;
            }
        }
    }
    const float rs = rsqrtf(block_sum<THR / kWave>(ss, lds) / (float)n + eps);

    norm_quant_row<WF32, THR>(nr, w, y ? y + (size_t)row * y_ld : nullptr, q + (size_t)row * q_ld,
                              s, (size_t)row * s_ld, n, nkb, rs, wadd);
}

// ---------------------------------------------------------------------------- gate -> quant
// silu(gate) * up, or sigmoid(gate) * up, then the same quantiser.  No reduction across the row, so
// there is no first pass and no barrier: one wave, one group, done.
//
// ROT INSERTS THE WIDTH-128 HADAMARD between the product and the absmax, which is where it has to
// go: the benefit of a rotation is entirely that the maximum the grid is scaled by gets smaller.
// It is free here for the same reason it is free in r4d_quant_act_fp8 -- this kernel's mapping,
// one wave a 128-group and four consecutive values a lane, IS r4d_fwht_wave<128>'s, so the seven
// stages are register arithmetic and five shuffles on a launch already being paid.
//
// `y` STILL RECEIVES THE UNROTATED PRODUCT, and that is not an inconsistency. `y` exists for a
// bf16 consumer of the silu output; the rotation is a property of the QUANTISED pair alone, and
// the weight that undoes it never sees `y`.
//
// A WAVE A GROUP, AND NO IDLE WAVES. With no reduction across the row the workgroup's width changes
// nothing but how many groups run at once, so the launcher sizes it to the row's groups, capped at
// r4d_act_threads' width, and the kernel reads the width from blockDim. A fixed width leaves most
// of the waves of a narrow row with nothing to do: at 1024 threads and three groups, 29 of 32.
template <int ACT, bool ROT = false, bool I8 = false>
__global__ __launch_bounds__(1024) void gated_quant_fp8_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t nkb, uint32_t gu_ld, uint32_t y_ld,
    uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t*  __restrict__ qr = q + (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += blockDim.x / kWave) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // One branch for all four columns, so all eight loads issue before any is waited on; see
        // norm_quant_row.
        if (c0 < n) {
            uint16_t gb[4], ub[4];
            const uint64_t* __restrict__ gr64 = (const uint64_t*)(const void*)gr;
            const uint64_t g_raw = gr64[c0 >> 2];
            const uint64_t u_raw = gr64[(n + c0) >> 2];
            __builtin_memcpy(gb, &g_raw, sizeof(gb));
            __builtin_memcpy(ub, &u_raw, sizeof(ub));

            uint16_t hb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g = bf2f(gb[j]);
                const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
                const uint16_t h = f2bf(a * bf2f(ub[j]));
                hb[j] = h;
                v[j] = bf2f(h);
            }
            if (yr) {
                uint64_t* __restrict__ yr64 = (uint64_t*)(void*)yr;
                uint64_t h_raw;
                __builtin_memcpy(&h_raw, hb, sizeof(h_raw));
                yr64[c0 >> 2] = h_raw;
            }
        }
        // The rotated launcher refuses an n that is not a multiple of the group, so no block that
        // reaches this line mixes real values with the padding zeros above.
        if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v, (int)lane);
        quant_group<I8>(v, s, qr, (size_t)row * s_ld + kb, c0, n, lane);
    }
}

// -------------------------------------------------- gate . x -> quant, two source buffers
//
// THE SAME EPILOGUE WITH THE GATE SOMEWHERE ELSE.  A gated attention's output gate is not packed
// beside the thing it scales: `x` is the attention output [T, n_head * head_dim] and the gate is a
// COLUMN SLICE of the [q|gate] projection, [T * n_head, 2 * head_dim] with the gate half at
// `g_off`.  So the row is a (token, head) pair rather than a token, which is the whole trick: at
// that shape `x` is contiguous, the gate is a fixed stride apart, and the fp8 scale plane of
// [T * n_head, head_dim / 128] is byte-for-byte the [T, n_head * head_dim / 128] the out
// projection reads.  Three launches an attention layer -- sigmoid, the multiply, the quantiser --
// become one, and at a decode batch the dispatch gap between those three is comparable to the work
// they do.
template <int ACT, bool I8 = false>
__global__ __launch_bounds__(1024) void gate_quant_fp8_kernel(
    const uint16_t* __restrict__ g, const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n, uint32_t nkb, uint32_t g_ld,
    uint32_t x_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = g + (size_t)row * g_ld;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t*  __restrict__ qr = q + (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += blockDim.x / kWave) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // One branch for all four columns, so all eight loads issue before any is waited on; see
        // norm_quant_row.
        if (c0 < n) {
            uint16_t gb[4], xb[4];
            const uint64_t* __restrict__ gr64 = (const uint64_t*)(const void*)gr;
            const uint64_t* __restrict__ xr64 = (const uint64_t*)(const void*)xr;
            const uint64_t g_raw = gr64[c0 >> 2];
            const uint64_t x_raw = xr64[c0 >> 2];
            __builtin_memcpy(gb, &g_raw, sizeof(gb));
            __builtin_memcpy(xb, &x_raw, sizeof(xb));

            uint16_t hb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = bf2f(gb[j]);
                // THE ACTIVATION IS ROUNDED BEFORE THE MULTIPLY, which is the one place this
                // differs from gated_quant_fp8 above and it is not an oversight.  That kernel's
                // unfused reference is silu_mul_bf16, ONE kernel that keeps the activation in f32
                // through the product.  This one's is `sigmoid` then `mul`, TWO kernels with a
                // bf16 buffer between them -- so the round trip is part of the arithmetic being
                // replaced, and skipping it changes the model's output: the emitted text moves and
                // the drafter's acceptance rate falls.
                const uint16_t ab =
                    f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
                const uint16_t h = f2bf(bf2f(ab) * bf2f(xb[j]));
                hb[j] = h;
                v[j] = bf2f(h);
            }
            if (yr) {
                uint64_t* __restrict__ yr64 = (uint64_t*)(void*)yr;
                uint64_t h_raw;
                __builtin_memcpy(&h_raw, hb, sizeof(h_raw));
                yr64[c0 >> 2] = h_raw;
            }
        }
        quant_group<I8>(v, s, qr, (size_t)row * s_ld + kb, c0, n, lane);
    }
}

// -------------------------------------------------------- all-reduce -> rmsnorm -> quant
//
// The same norm again, with the tensor-parallel all-reduce that always precedes it folded in.
// Every row-parallel linear's output is all-reduced and then immediately normalised -- o_proj,
// out_proj and down_proj, 128 of each on a 64-layer decode step, always adjacent and always in
// that order. The norm's own body is a small fraction of what the pair costs; the rest is a launch,
// a dispatch gap and a round trip through memory for a tensor that exists for one kernel. The int8
// path carries the same fusion (r4d_ar_ln_had_quant_i8); this is the fp8 form of it, minus the
// rotation.
//
// THE PARTITION IS BY ROW, for the reason that file gives at length: r4d_ar_oneshot_2rank_exact
// hands block b a contiguous chunk of 16B WORDS with an independent flag and no grid barrier
// anywhere, and a norm cannot use that partition because a sum of squares closes only over a whole
// row -- but rows are contiguous, so a contiguous range of ROWS is a contiguous range of words.
// Both ranks compute the same range from (M, gridDim.x), so the handshake stays per-block.
//
// BIT-IDENTICAL TO THE PAIR IT REPLACES, and by construction rather than by inspection. Pass two
// is norm_quant_row above, the same code the standalone kernel runs. Pass one reproduces the
// all-reduce's own arithmetic exactly -- an f32 sum STORED AS BF16 (r4d_ar_oneshot_2rank_exact's
// vadd) and then read as the norm's input -- and stores it back into `x`, which is not bookkeeping:
// it is what the kernel this replaces wrote, so a reader of `x` that is not this norm still finds
// the same bytes. The column partition of the sum of squares is `i = tid; i += THR` on both, which
// is the agreement r4d_act_threads exists to keep.
//
// BOTH WIRES ARE SERVED, and the compressed one is why this kernel is worth its size. Above
// --tp-wire-min-kb the message goes out 6-bit rotated, and that payload is worth more than the
// launch a fusion saves. A kernel that served only the exact wire would hand every message above
// the floor back to the standalone pair, and the fusion would run at one sequence and nowhere
// else -- which is the wrong band, because a step carries n_seq * (1 + n_spec) rows, so the message
// crosses the floor at the SECOND concurrent sequence and stays across it. Serving WIRE == 6 here
// is what lets the two wins compose.
//
// The wire is r4d_ar_oneshot_2rank_wht6's format and arithmetic unchanged -- orthonormal
// Walsh-Hadamard over a group of 64, amax-scaled, 6 uniform bits plus a bf16 scale -- packed at a
// CHUNK of 512 elements instead of 2048. The chunk size is not numerics: the rotation and the amax
// close over a GROUP, and a group is 64 elements at either chunking, so the codes and the scales
// are the same bytes. What it decides is who owns what. A chunk may not straddle a row, or a
// block's rows would not be a whole number of chunks and the two ranks would disagree about which
// bytes a block owns; and a wave owns a whole chunk, so a smaller chunk puts more waves to work,
// which is why a chunk is 8 groups and not 32 (see r4d_ar_ln_had_quant_i8.hip).
//
// THE REDUCED ROW GOES BACK TO GLOBAL `x`, not to an LDS staging slice. The int8 twin stages it,
// because its norm needs the row again for a rotation it performs in place; this one does not --
// pass two re-reads the row it just wrote, which is the same re-read the standalone norm does and
// the same L1-hot argument the top of this file makes. It costs a bf16 round trip that the int8
// form avoids and buys back all of the LDS.
//
// AND THIS RANK'S OWN PACKED HALF GOES TO GLOBAL TOO, in the side buffer the standalone kernel
// already allocates (`loc_pack`), rather than to LDS. LDS cannot hold it, and the failure only a
// long enough prompt shows: LDS is 64 KiB and a row's packed half is ~4 KiB at n = 5120, so a block
// holds fifteen rows where a PREFILL chunk is two thousand. The launch is then refused with a shape
// error, which in this engine aborts the step and fails the requests in it. A bound that depends on
// the block count is not a bound this op gets to have: the same op serves M = 8 and
// M = 2048. The side buffer costs 0.75 bytes an element written and read straight back, which is
// under half a microsecond at decode width and stays in L2.
#define R4D_ARQ_SPIN_MAX 4000000000ULL

#define ARQ_GROUP 64
#define ARQ_BITS 6
#define ARQ_EPL 8
#define ARQ_LPG (ARQ_GROUP / ARQ_EPL)                    /* 8 lanes span one group */
#define ARQ_WV 32
#define ARQ_GPW (ARQ_WV * ARQ_EPL / ARQ_GROUP)           /* 4 groups a wave-iteration */
#define ARQ_CG 8                                         /* groups a chunk */
#define ARQ_CE (ARQ_CG * ARQ_GROUP)                      /* 512 elements a chunk */
#define ARQ_ITERS (ARQ_CE / (ARQ_WV * ARQ_EPL))          /* 2 iterations to cover a chunk */
#define ARQ_NWORD (ARQ_ITERS * ARQ_EPL * ARQ_BITS / 32)  /* 3 dwords a lane holds for a chunk */
#define ARQ_CBYTES (ARQ_CE * ARQ_BITS / 8)               /* 384 wire bytes a chunk */
#define ARQ_QMAX ((1 << (ARQ_BITS - 1)) - 1)
#define ARQ_QMIN (-(1 << (ARQ_BITS - 1)))
#define ARQ_MASK ((1u << ARQ_BITS) - 1u)

// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`: the sign flip is one XOR against a mask
// uniform over the stage, so the select disappears. Bit-identical -- IEEE negation is exact.
template <int M>
__device__ __forceinline__ void arq_xstage(float* v, int lig) {
    const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
    for (int i = 0; i < ARQ_EPL; ++i) {
        const float o = r4d_xor_lane<M>(v[i]);
        v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
    }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this also undoes it.
__device__ __forceinline__ void arq_wht(float* v, int lig) {
    static_assert(ARQ_LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
    for (int h = 1; h < ARQ_EPL; h <<= 1)
#pragma unroll
        for (int i = 0; i < ARQ_EPL; ++i)
            if ((i & h) == 0) { const float a = v[i], c = v[i ^ h]; v[i] = a + c; v[i ^ h] = a - c; }
    arq_xstage<1>(v, lig);
    arq_xstage<2>(v, lig);
    arq_xstage<4>(v, lig);
    const float nrm = rsqrtf((float)ARQ_GROUP);
#pragma unroll
    for (int i = 0; i < ARQ_EPL; ++i) v[i] *= nrm;
}

__device__ __forceinline__ void arq_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arq_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <bool WF32, bool RES, uint32_t THR, int WIRE>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_fp8_kernel(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint8_t* __restrict__ q, float* __restrict__ s,
    int4* peer_base, const int4* my_base, unsigned int* peer_flags, unsigned int* my_flags,
    unsigned int* seq_ctrs, char* loc_pack, long loc_scale_off, int M, uint32_t n, uint32_t nkb,
    float eps, float wadd, uint32_t r_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld,
    int slot_stride16, int drain, int acq, unsigned int nbw) {
    __shared__ float lds[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    /* Integer arithmetic only, so both ranks land on the same split. */
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    /* Chunks a row spans, and this block's window into the message's global chunk numbering. Both
     * ranks derive them from the same (M, nb, n), so neither has to be told. */
    const int cpr = WIRE ? (int)(n / ARQ_CE) : 0;
    const int c00 = r0 * cpr, cpb = (r1 - r0) * cpr;
    const uint32_t lane = tid & (ARQ_WV - 1), wave = tid / ARQ_WV;
    const int lig = (int)(lane & (ARQ_LPG - 1)), gw = (int)(lane / ARQ_LPG);

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* const ms = my_base + (size_t)s_slot * slot_stride16;
    /* The slot is split exactly as the standalone compressed all-reduce splits it: packed symbols
     * from the base, bf16 group scales from half way up. Group indices are GLOBAL over the whole
     * [M][n] message, so a block simply addresses the groups its rows cover. */
    char* const p_pay = (char*)ps;
    const char* const m_pay = (const char*)ms;
    unsigned short* const p_sc =
        (unsigned short*)((char*)ps + (size_t)(slot_stride16 >> 1) * 16);
    const unsigned short* const m_sc =
        (const unsigned short*)((const char*)ms + (size_t)(slot_stride16 >> 1) * 16);
    /* This rank's own packed half, in the side buffer split exactly as the peer slot is: symbols
     * from the base, scales from `loc_scale_off`. Indexed by the GLOBAL chunk and group number, so
     * a block writes and reads only what its own rows cover and no two blocks overlap. */
    unsigned int* const Lpk = (unsigned int*)loc_pack;
    float* const Lsc = (float*)(loc_pack + loc_scale_off);

    if constexpr (WIRE == 0) {
        /* Push my rows into the peer's slot. `x` is a tight [M][n] of bf16 -- the launcher refuses
         * a pitched one -- so a row range is a 16B-word range. */
        const int wpr = (int)(n >> 3);
        const int wend = r1 * wpr;
        const int4* const in4 = (const int4*)x;
        for (int k = r0 * wpr + (int)tid; k < wend; k += (int)THR) ps[k] = in4[k];
    } else {
        /* Rotate, amax, quantise and pack straight into the peer's slot, keeping the same words in
         * LDS. A wave owns whole chunks, so every lane emits exactly ARQ_NWORD dwords a chunk and
         * the flush is lane-uniform. No group is ever partial: n is a whole number of chunks and a
         * block owns whole rows, so the `live` guard the standalone kernel needs is not needed
         * here -- which is also why every cross-lane stage below is wholly active. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQ_WV)) {
            const int c = c00 + ci;
            unsigned int acc[ARQ_NWORD];
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi) acc[wi] = 0u;
#pragma unroll
            for (int t = 0; t < ARQ_ITERS; ++t) {
                const int g = c * ARQ_CG + t * ARQ_GPW + gw;
                const long off = (long)g * ARQ_GROUP + (long)lig * ARQ_EPL;
                float v[ARQ_EPL];
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) v[i] = bf2f(x[off + i]);
                arq_wht(v, lig);
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
                a = fmaxf(a, r4d_xor_lane<1>(a));
                a = fmaxf(a, r4d_xor_lane<2>(a));
                a = fmaxf(a, r4d_xor_lane<4>(a));
                float sc = (a > 0.0f) ? bf2f(f2bf(a * (1.0f / (float)ARQ_QMAX))) : 1.0f;
                if (!(sc > 0.0f)) sc = 1.0f;
                if (lig == 0) {
                    p_sc[g] = (unsigned short)(__float_as_uint(sc) >> 16);
                    Lsc[g] = sc;
                }
                const float inv = 1.0f / sc;
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) {
                    const unsigned int sym = (unsigned int)(
                        min(max((int)rintf(v[i] * inv), ARQ_QMIN), ARQ_QMAX) - ARQ_QMIN);
                    const int bit = (t * ARQ_EPL + i) * ARQ_BITS, wi = bit >> 5, bo = bit & 31;
                    acc[wi] |= sym << bo;
                    if (bo + ARQ_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
                }
            }
            /* Flush lane-contiguously, widest region first, so each is one coalesced store. */
            char* const pb = p_pay + (size_t)c * ARQ_CBYTES;
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQ_NWORD; w4 += 4)
                *(uint4*)(pb + w4 * 4 * ARQ_WV + lane * 16) =
                    make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]);
            if constexpr (ARQ_NWORD % 4 >= 2)
                *(uint2*)(pb + (ARQ_NWORD & ~3) * 4 * ARQ_WV + lane * 8) =
                    make_uint2(acc[ARQ_NWORD & ~3], acc[(ARQ_NWORD & ~3) + 1]);
            if constexpr (ARQ_NWORD % 2 == 1)
                *(unsigned int*)(pb + (ARQ_NWORD - 1) * 4 * ARQ_WV + lane * 4) =
                    acc[ARQ_NWORD - 1];
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi)
                Lpk[((size_t)c * ARQ_NWORD + wi) * ARQ_WV + lane] = acc[wi];
        }
    }
    /* The drain modes are r4d_ar_oneshot_2rank_exact's, and the variant table picks between them
     * for the same reasons; the kernel only has to honour them identically. */
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arq_store_rel(&peer_flags[b], sq);
        unsigned long long z = 0;
        /* A handshake that cannot complete is a lockstep violation: norming a slot the peer never
         * filled would hand stale bytes on as a sum, so it traps instead. */
        while (r4d_ar_seq_behind(arq_load_acq(&my_flags[b]), sq)) {
            if (++z > R4D_ARQ_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    if (acq) __threadfence_system();

    if constexpr (WIRE == 6) {
        /* REDUCE: unpack both halves of every chunk this block owns, add in fp32, inverse-rotate,
         * and store the bf16 sum back into `x` -- which is what the standalone compressed
         * all-reduce wrote, so pass two below reads the bytes it would have read and a reader of
         * `x` that is not this norm still finds the same ones. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQ_WV)) {
            const int c = c00 + ci;
            const char* const pb = m_pay + (size_t)c * ARQ_CBYTES;
            unsigned int pac[ARQ_NWORD], lac[ARQ_NWORD];
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQ_NWORD; w4 += 4) {
                const uint4 pq = *(const uint4*)(pb + w4 * 4 * ARQ_WV + lane * 16);
                pac[w4] = pq.x; pac[w4 + 1] = pq.y; pac[w4 + 2] = pq.z; pac[w4 + 3] = pq.w;
            }
            if constexpr (ARQ_NWORD % 4 >= 2) {
                const uint2 pu = *(const uint2*)(pb + (ARQ_NWORD & ~3) * 4 * ARQ_WV + lane * 8);
                pac[ARQ_NWORD & ~3] = pu.x; pac[(ARQ_NWORD & ~3) + 1] = pu.y;
            }
            if constexpr (ARQ_NWORD % 2 == 1)
                pac[ARQ_NWORD - 1] =
                    *(const unsigned int*)(pb + (ARQ_NWORD - 1) * 4 * ARQ_WV + lane * 4);
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi)
                lac[wi] = Lpk[((size_t)c * ARQ_NWORD + wi) * ARQ_WV + lane];
#pragma unroll
            for (int t = 0; t < ARQ_ITERS; ++t) {
                const int gl = t * ARQ_GPW + gw;
                const int g = c * ARQ_CG + gl;
                const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
                const float lsc = Lsc[g];
                float v[ARQ_EPL];
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) {
                    const int bit = (t * ARQ_EPL + i) * ARQ_BITS, wi = bit >> 5, bo = bit & 31;
                    unsigned int psy = pac[wi] >> bo, lsy = lac[wi] >> bo;
                    if (bo + ARQ_BITS > 32) {
                        psy |= pac[wi + 1] << (32 - bo);
                        lsy |= lac[wi + 1] << (32 - bo);
                    }
                    /* The two ranks fold the identical pair in OPPOSITE order; fp32 add commutes
                     * exactly, so they stay bit-identical as long as neither product is contracted
                     * into an FMA. libr4d/CMakeLists.txt passes -ffp-contract=off for exactly this. */
                    v[i] = (float)((int)(psy & ARQ_MASK) + ARQ_QMIN) * psc
                         + (float)((int)(lsy & ARQ_MASK) + ARQ_QMIN) * lsc;
                }
                arq_wht(v, lig);
                const long off = (long)g * ARQ_GROUP + (long)lig * ARQ_EPL;
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) x[off + i] = f2bf(v[i]);
            }
        }
        /* Pass two reads `x` across the whole row and a wave only wrote its own chunks of it. */
        __syncthreads();
    }

    /* The peer's rows are now in MY scratch, in the same [M][n] layout as x -- WIRE == 0 only;
     * the compressed wire has already reduced into `x` above. */
    const uint16_t* __restrict__ P = (const uint16_t*)ms;

    for (int row = r0; row < r1; ++row) {
        uint16_t* __restrict__ xr = x + (size_t)row * n;
        const uint16_t* __restrict__ pr = P + (size_t)row * n;
        uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
        const uint16_t* __restrict__ nr = RES ? rr : xr;

        (void)pr;
        float ss = 0.0f;
        for (uint32_t i = tid; i < n; i += THR) {
            /* The exact wire reduces HERE, on the norm's own pass; the compressed one has already
             * reduced into `x` and this reads what it stored. */
            uint16_t a;
            if constexpr (WIRE == 0) { a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a; }
            else                     { a = xr[i]; }
            float v;
            if (RES) { const uint16_t h = f2bf(bf2f(rr[i]) + bf2f(a)); rr[i] = h; v = bf2f(h); }
            else     { v = bf2f(a); }
            ss += v * v;
        }
        /* block_sum's barrier is also what publishes pass one's stores to pass two's reads. */
        const float rs = rsqrtf(block_sum<THR / kWave>(ss, lds) / (float)n + eps);
        norm_quant_row<WF32, THR>(nr, w, y ? y + (size_t)row * y_ld : nullptr,
                                  q + (size_t)row * q_ld, s, (size_t)row * s_ld, n, nkb, rs, wadd);
        /* lds is re-used by the next row this block owns, and a thread still reading it for this
         * row would otherwise race the next row's write. The one-row-a-block kernel above never
         * needed this barrier and that is the only structural difference between them. */
        __syncthreads();
    }
}

}  // namespace

// `w` is the norm gain, f32 when `w_f32` and bf16 otherwise -- every gain in this project's
// architecture plugins is declared f32, and the width is the caller's to state for the reason
// r4d_model_bf16's rmsnorm gives.  `y` may be null on a layer with no bf16 consumer of the
// post-norm tensor; every caller in this tree has one, because the fp8a16 matvec reads it at M = 1.
//
// `res`, when given, is the fused_add_rms_norm form: `res += x` in f32, ROUNDED TO BF16 and stored,
// and the norm is then taken of the stored value -- which is bit-for-bit the `add` + `rmsnorm` pair
// this replaces. Pass 0 for the plain form. It may alias `y`; see the note at the top of the file.
//
//   x    [M][n]              bf16, `x_ld` elements a row (0 = n)
//   res  [M][n]              bf16, `r_ld` elements a row (0 = n), optional, READ AND WRITTEN
//   w    [n]                 f32 or bf16
//   y    [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q    [M][n]              E4M3, `q_ld` bytes a row (0 = n), rounded up to 4
//   s    [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
extern "C" int r4d_rmsnorm_quant_fp8(long x, long res, long w, long y, long q, long s, int M, int n,
                                     float eps, float wadd, int w_f32, int x_ld, int r_ld,
                                     int y_ld, int q_ld, int s_ld, long stream) {
    if (!x || !w || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;              // a lane owns four columns; see the kernel
    if (!x_ld) x_ld = n;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int nkb = (n + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;

    // ONE WORKGROUP A ROW, so at a decode batch the grid is 8-72 workgroups on a 64-CU part and
    // the kernel is latency bound rather than bandwidth bound: it moves ~205 KiB, which is far less
    // work than the ~2.6 us launch floor costs. Widening the workgroup is the only knob that adds
    // parallelism without changing the algorithm. r4d_act_threads holds the crossover table, and
    // the reasoning behind it is written out with it.
    //
    // rmsnorm_bf16 PICKS THE SAME WIDTH FROM THE SAME FUNCTION, and it has to: the width decides
    // which columns a thread accumulates into the sum of squares, so two kernels that disagree
    // about it produce sums that differ in the last bits. This fusion is required to be
    // byte-identical to the pair it replaces, which is exactly that agreement.
    const int thr = r4d_act_threads(M, n);
#define R4D_RQ_T(wf, rs, T)                                                                     \
    hipLaunchKernelGGL((rmsnorm_quant_fp8_kernel<wf, rs, T>), dim3((uint32_t)M), dim3(T), 0,    \
                       (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)res, (const void*)w,  \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)nkb, eps,    \
                       wadd, (uint32_t)x_ld, (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)q_ld,     \
                       (uint32_t)s_ld)
#define R4D_RQ(wf, rs)                                                                          \
    do { if (thr == 1024)     R4D_RQ_T(wf, rs, 1024);                                           \
         else if (thr == 512) R4D_RQ_T(wf, rs, 512);                                            \
         else                 R4D_RQ_T(wf, rs, 256); } while (0)
    if (w_f32) { if (res) R4D_RQ(true, true);  else R4D_RQ(true, false); }
    else       { if (res) R4D_RQ(false, true); else R4D_RQ(false, false); }
#undef R4D_RQ
#undef R4D_RQ_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `mode` is 0 for silu (down_proj's input) and 1 for sigmoid (a gated attention o_proj's), the same
// spelling r4d_gated_had_quant_i8 takes.
//
//   gu [M][2n]             bf16, gate first then up, `gu_ld` elements a row (0 = 2n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3 (int8 when `i8`), `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
//
// `rot` is a STRICTER call, not a flag on the same one: a ragged last group is legal unrotated
// (it is written zero-padded and the GEMM's bounds guard stops the padding reaching anything) and
// is not legal rotated, because a butterfly that mixed real values with padding has no matching
// weight block on the other side.
static int gated_quant_fp8_launch(long gu, long y, long q, long s, int M, int n, int mode,
                                  int gu_ld, int y_ld, int q_ld, int s_ld, const R4dSink& sk,
                                  bool rot, bool i8 = false) {
    int emit_rc = RAD_OK;
    if (!gu || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (rot && (n % (int)kGroup)) return -6;
    // The rotated codes pair with a rotated four-bit expert, which reads E4M3 only.
    if (rot && i8) return -7;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int nkb = (n + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;
    // A wave a group, capped at r4d_act_threads' width; see the kernel. There is no cross-row
    // reduction here, so the width changes nothing numerically -- only how many waves share the
    // groups. The ROTATION is wave-local too (r4d_fwht.h), so that stays true with it.
    const int thr = std::min(nkb, r4d_act_threads(M, n) / (int)kWave) * (int)kWave;
#define R4D_GQ_R(a, R, I)                                                                       \
    emit_rc = R4D_EMIT(sk, (gated_quant_fp8_kernel<a, R, I>), dim3((uint32_t)M), dim3(thr), 0,  \
                       (const uint16_t*)gu, (uint16_t*)y, (uint8_t*)q,                         \
                       (float*)s, (uint32_t)n, (uint32_t)nkb, (uint32_t)gu_ld, (uint32_t)y_ld,  \
                       (uint32_t)q_ld, (uint32_t)s_ld)
#define R4D_GQ(a)                                                                               \
    do { if (rot) R4D_GQ_R(a, true, false);                                                     \
         else if (i8) R4D_GQ_R(a, false, true);                                                 \
         else R4D_GQ_R(a, false, false); } while (0)
    if (mode == 0) R4D_GQ(0);
    else           R4D_GQ(1);
#undef R4D_GQ
#undef R4D_GQ_R
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_gated_quant_fp8(long gu, long y, long q, long s, int M, int n, int mode,
                                   int gu_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return gated_quant_fp8_launch(gu, y, q, s, M, n, mode, gu_ld, y_ld, q_ld, s_ld,
                                  r4d_stream_sink((RadStream)(uintptr_t)stream), false);
}

extern "C" int r4d_gated_had_quant_fp8(long gu, long y, long q, long s, int M, int n, int mode,
                                       int gu_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return gated_quant_fp8_launch(gu, y, q, s, M, n, mode, gu_ld, y_ld, q_ld, s_ld,
                                  r4d_stream_sink((RadStream)(uintptr_t)stream), true);
}

// act . x, with the gate and the thing it scales in DIFFERENT buffers -- see the kernel above for
// why the row is a (token, head) pair.  `mode` is spelled as r4d_gated_quant_fp8 spells it.  The
// activation is ROUNDED TO BF16 before the product, because the pair this replaces is two kernels
// with a bf16 buffer between them -- unlike r4d_gated_quant_fp8, whose reference is one.
//
//   g  [M][.]              bf16, `g_ld` elements a row; the gate starts at element 0 of the row,
//                          so a caller with a [q|gate] projection passes g already offset
//   x  [M][n]              bf16, `x_ld` elements a row (0 = n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3 (int8 when `i8`), `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
static int gate_quant_fp8_emit(long g, long x, long y, long q, long s, int M, int n, int mode,
                               int g_ld, int x_ld, int y_ld, int q_ld, int s_ld,
                               const R4dSink& sk, bool i8 = false) {
    int emit_rc = RAD_OK;
    if (!g || !x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (!g_ld) g_ld = n;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int nkb = (n + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;
    // A wave a group, capped at r4d_act_threads' width, as gated_quant_fp8 above.
    const int thr = std::min(nkb, r4d_act_threads(M, n) / (int)kWave) * (int)kWave;
#define R4D_GTQ_I(a, I)                                                                         \
    emit_rc = R4D_EMIT(sk, (gate_quant_fp8_kernel<a, I>), dim3((uint32_t)M), dim3(thr), 0,      \
                       (const uint16_t*)g, (const uint16_t*)x,                                 \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)nkb,        \
                       (uint32_t)g_ld, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)q_ld,          \
                       (uint32_t)s_ld)
#define R4D_GTQ(a) do { if (i8) R4D_GTQ_I(a, true); else R4D_GTQ_I(a, false); } while (0)
    if (mode == 0) R4D_GTQ(0);
    else           R4D_GTQ(1);
#undef R4D_GTQ
#undef R4D_GTQ_I
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_gate_quant_fp8(long g, long x, long y, long q, long s, int M, int n, int mode,
                                  int g_ld, int x_ld, int y_ld, int q_ld, int s_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gate_quant_fp8_emit(g, x, y, q, s, M, n, mode, g_ld, x_ld, y_ld, q_ld, s_ld, sk);
}

// The fused all-reduce. The IPC arguments are r4d_ar_oneshot_2rank_exact's -- the same
// double-buffered scratch, per-block flags and device-resident sequence counters -- and everything
// after them is r4d_rmsnorm_quant_fp8's, with two differences the caller has to know about:
//
//   * `x` HAS NO PITCH. It is the all-reduce's message as well as the norm's input, and a block
//     pushes its rows as a range of 16B words; a pitched x would make the two ranks disagree about
//     which bytes a block owns. Refused rather than silently strided.
//   * `nblocks` IS CLAMPED TO M, because a block owns whole rows. At a decode batch that is 8 to
//     64 blocks, which is the range the standalone all-reduce asks for anyway.
//
// `res` is the fused_add_rms_norm form and behaves exactly as it does there. Returns 0, or a
// negative code for a shape it does not serve.
extern "C" int r4d_ar_rmsnorm_quant_fp8_chunk_elems(void) { return ARQ_CE; }

extern "C" int r4d_ar_rmsnorm_quant_fp8(long peer_scratch, long my_scratch, long peer_flags,
                                        long my_flags, long seq_ctrs, long slot_stride16,
                                        long x, long res, long w, long y, long q, long s,
                                        int M, int n, float eps, float wadd, int w_f32,
                                        int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                        int nblocks, int nthreads, int drain, int acq, int wire,
                                        long loc_pack, long loc_scale_off) {
    if (!x || !w || !q || !s) return -1;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;              // a lane owns four columns; see norm_quant_row
    if (n % 8) return -5;              // n bf16 is a whole number of 16B words; see the push
    if (wire != 0 && wire != 6) return -7;
    // A chunk may not straddle a row, or a block's rows would not be a whole number of chunks and
    // the two ranks would disagree about which bytes a block owns.
    if (wire == 6 && (n % ARQ_CE)) return -8;
    if (wire == 6 && (!loc_pack || loc_scale_off <= 0)) return -9;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int nkb = (n + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    const int nbw = r4d_ar_max_blocks();       // the counter capacity; see r4d_ar_seq.h
    if (nb > nbw) nb = nbw;
    // The width is r4d_act_threads for the reason the standalone norm gives: it decides which
    // columns a thread accumulates into the sum of squares, so a fused kernel that picked a
    // different one would produce a different sum in the last bits and a different token stream.
    const int thr = nthreads ? nthreads : r4d_act_threads(M, n);
    if (thr != 256 && thr != 512 && thr != 1024) return -6;
#define R4D_ARQ_T(wf, rs, T, W)                                                                  \
    hipLaunchKernelGGL((ar_rmsnorm_quant_fp8_kernel<wf, rs, T, W>), dim3((uint32_t)nb), dim3(T),  \
                       0,                                                                        \
                       (hipStream_t)stream, (uint16_t*)x, (uint16_t*)res, (const void*)w,        \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (int4*)peer_scratch,                \
                       (const int4*)my_scratch, (unsigned int*)peer_flags,                       \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, (char*)loc_pack,        \
                       (long)loc_scale_off, M, (uint32_t)n,                                      \
                       (uint32_t)nkb, eps, wadd, (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)q_ld, \
                       (uint32_t)s_ld, (int)slot_stride16, drain, acq, (unsigned)nbw)
#define R4D_ARQ_W(wf, rs, T)                                                                     \
    do { if (wire == 6) R4D_ARQ_T(wf, rs, T, 6); else R4D_ARQ_T(wf, rs, T, 0); } while (0)
#define R4D_ARQ(wf, rs)                                                                          \
    do { if (thr == 1024)     R4D_ARQ_W(wf, rs, 1024);                                           \
         else if (thr == 512) R4D_ARQ_W(wf, rs, 512);                                            \
         else                 R4D_ARQ_W(wf, rs, 256); } while (0)
    if (w_f32) { if (res) R4D_ARQ(true, true);  else R4D_ARQ(true, false); }
    else       { if (res) R4D_ARQ(false, true); else R4D_ARQ(false, false); }
#undef R4D_ARQ
#undef R4D_ARQ_W
#undef R4D_ARQ_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. See libr4d/r4d_args.h for why there is no adapter
// directory between the core and these launches.
/* ================================================================== the two fused quantisers
 *
 * The shape rules are the standalone quantiser's, applied to the operand that PRODUCES the row
 * rather than to the row itself, plus the one thing the fusion adds: the bf16 output is optional
 * because a layer with no bf16 consumer of the post-norm tensor should not pay for the store, and
 * every caller in this tree has one anyway (the fp8a16 matvec reads it at M = 1). */

extern "C" int r4d_rad_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    const RadTensor* x  = r4d_opt(a, RQ_X);
    const RadTensor* rs = r4d_opt(a, RQ_RES);
    const RadTensor* w  = r4d_opt(a, RQ_W);
    const RadTensor* q  = r4d_opt(a, RQ_Q);
    const RadTensor* sc = r4d_opt(a, RQ_S);
    const RadTensor* y  = r4d_opt(a, RQ_Y);
    if (!x || !w || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* Rank 2 on the input too, unlike the standalone quantiser: a norm has a gain per channel, so
     * "everything after dim 0 is the row" would put one gain across several heads. The GDN block's
     * rank-3 activation is quantised by the standalone op, and its gated norm by its own kernel. */
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = x->shape[1];
    const long long grp = r4d_quant_act_fp8_group();
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_geti_or(a, "group", grp) != grp) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(w) != K) return R4D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    /* The residual is the same extent as the input and is READ AND WRITTEN. It is allowed to alias
     * `out_bf16` -- every caller in this tree makes it do so -- which is safe because pass one
     * reads all of `x` before the kernel's only barrier and pass two writes `y` after it. */
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = r4d_rmsnorm_quant_fp8(r4d_addr(x), rs ? r4d_addr(rs) : 0, r4d_addr(w),
                                        y ? r4d_addr(y) : 0, r4d_addr(q), r4d_addr(sc),
                                        (int)M, (int)K,
                                        (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
                                        w->dtype == RAD_F32, (int)xp, (int)rp, (int)yp, (int)qp,
                                        (int)sp, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_gated_quant_fp8(const RadArgs* a, const R4dSink& sk, bool rot) {
    const RadTensor* gu = r4d_opt(a, GQ_GU);
    const RadTensor* q  = r4d_opt(a, GQ_Q);
    const RadTensor* sc = r4d_opt(a, GQ_S);
    const RadTensor* y  = r4d_opt(a, GQ_Y);
    if (!gu || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype picks the quantiser: E4M3, or int8 for an int8 down projection. */
    if (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (rot && q->dtype == RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (gu->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = gu->shape[0], K = q->shape[1];
    const long long grp = r4d_quant_act_fp8_group();
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_geti_or(a, "group", grp) != grp) return R4D_NO(RAD_E_SHAPE);
    /* The gated operand is [M][2K] and the halves are gate then up -- one buffer, because that is
     * how the fused gate_up projection writes it and splitting would be a copy. */
    if (gu->shape[1] != 2 * K) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    const long long gp = r4d_row_pitch(gu), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    if (gp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    /* `act` is the vocabulary's name and `mode` is libr4d's integer spelling of the same choice;
     * `act` wins where both are given, which is what r4d_gated_had_quant_i8's row promises. */
    const char* act = r4d_gets(a, "act");
    int mode = (int)r4d_geti_or(a, "mode", 0);
    if (act) {
        if (!std::strcmp(act, "silu")) mode = 0;
        else if (!std::strcmp(act, "sigmoid")) mode = 1;
        else return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const int e = gated_quant_fp8_launch(r4d_addr(gu), y ? r4d_addr(y) : 0, r4d_addr(q),
                                         r4d_addr(sc), (int)M, (int)K, mode, (int)gp, (int)yp,
                                         (int)qp, (int)sp, sk, rot, q->dtype == RAD_I8);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_gated_quant_fp8_body(const RadArgs* a, const R4dSink& sk) {
    return rad_gated_quant_fp8(a, sk, false);
}
R4D_ENTRY(r4d_rad_gated_quant_fp8, rad_gated_quant_fp8_body)

/* THE ROTATED FORM, a different op for the same reason had_quant_act_fp8 is: the codes it writes
 * are not the codes the other one writes, so a graph with one substituted for the other is a
 * different model that still runs and still answers. It pairs with a down-projection weight
 * stored at dtype w4a8h and with nothing else. */
static int rad_gated_had_quant_fp8_body(const RadArgs* a, const R4dSink& sk) {
    return rad_gated_quant_fp8(a, sk, true);
}
R4D_ENTRY(r4d_rad_gated_had_quant_fp8, rad_gated_had_quant_fp8_body)

/* Elements between consecutive `width`-wide rows, the same rule r4d_gdn.cpp's gn_row_pitch uses:
 * a [.., heads, width] operand steps a row with the stride of the axis above the last, and a
 * flattened one is contiguous and steps by `width`. It is the only way to read a column view of
 * an interleaved projection, which is what a gated attention's gate is. */
static long long tq_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    /* AND THE ROWS HAVE TO BE UNIFORMLY THAT FAR APART, which the width test alone does not say.
     * A [T, n_head, head_dim] column view of the [q|gate] projection flattens to T * n_head rows
     * only if T's stride is exactly n_head of them -- true here, because the narrowing shrank the
     * last axis and left every stride alone, and false for anything that also sliced an axis
     * above it. Refusing is the only safe answer: a wrong pitch reads the next head's gate. */
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* THE SAME EPILOGUE WITH THE GATE SOMEWHERE ELSE. A gated attention's output gate is not packed
 * beside the thing it scales -- `x` is the attention output [M][n] and the gate is a COLUMN SLICE
 * of the [q|gate] projection -- so this takes two operands where gated_quant_fp8 takes one. The
 * caller narrows the gate to its own columns with a column view and the shim reads the offset off
 * the tensor, which is why there is no `gate_offset` parameter. */
static int rad_gate_quant_fp8_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* g  = r4d_opt(a, TQ_G);
    const RadTensor* x  = r4d_opt(a, TQ_X);
    const RadTensor* q  = r4d_opt(a, TQ_Q);
    const RadTensor* sc = r4d_opt(a, TQ_S);
    const RadTensor* y  = r4d_opt(a, TQ_Y);
    if (!g || !x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    /* M IS COUNTED OFF `x`, NOT READ OUT OF THE PARAMETERS. `M` is the row's RANGED parameter, so
     * what arrives here is the band's declared maximum and not the issue -- a shim that trusts it
     * refuses every real step. And shape[0] is no better: the row here is a (token, head) pair and
     * neither operand spells it that way, the gate being a [T, n_head, head_dim] column view of
     * the [q|gate] projection and x a flat [T, n_head * head_dim]. The element count over the
     * known width is the one reading both agree on. */
    const long long K = r4d_geti_or(a, "n", 0);
    const long long grp = r4d_quant_act_fp8_group();
    if (K <= 0 || r4d_numel(x) % K) return R4D_NO(RAD_E_SHAPE);
    const long long M = r4d_numel(x) / K;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_geti_or(a, "group", grp) != grp) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(g) < M * K) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) < M * K || r4d_numel(sc) < M * nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) < M * K) return R4D_NO(RAD_E_SHAPE);
    const long long gp = tq_pitch(g, K), xp = tq_pitch(x, K);
    const long long qp = tq_pitch(q, K), sp = tq_pitch(sc, nkb);
    const long long yp = y ? tq_pitch(y, K) : 0;
    if (gp < 0 || xp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    const char* act = r4d_gets(a, "act");
    int mode = (int)r4d_geti_or(a, "mode", 1);
    if (act) {
        if (!std::strcmp(act, "silu")) mode = 0;
        else if (!std::strcmp(act, "sigmoid")) mode = 1;
        else return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const int e = gate_quant_fp8_emit(r4d_addr(g), r4d_addr(x), y ? r4d_addr(y) : 0,
                                      r4d_addr(q), r4d_addr(sc), (int)M, (int)K, mode,
                                      (int)gp, (int)xp, (int)yp, (int)qp, (int)sp, sk,
                                      q->dtype == RAD_I8);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_gate_quant_fp8, rad_gate_quant_fp8_body)

