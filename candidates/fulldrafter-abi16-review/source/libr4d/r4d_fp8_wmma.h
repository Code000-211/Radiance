// Shared device-side machinery for the BLOCK-SCALED FP8 family: the 16x16x16 fp8 WMMA fragment
// map, the E4M3 decoders, and the bf16 block-scale load. Named for what it is rather than for a
// kernel, like r4d_common.h and r4d_gdn_wmma.h, because it provides none.
//
// ---- the lane->element mapping, which AMD does not document ------------------------------------
//
// AMD does not publish it, and ROCm issue #6025, which asks for it, is closed without an answer.
// With lane `l`, `c = l % 16`, `h = l / 16` it is:
//
//   A, row-major A[m][k]     lane l holds A[c][h*8 + j],  j = 0..7
//   B, row-major B[k][n]     lane l holds B[h*8 + j][c],  j = 0..7
//   C/D accumulator          acc[j] IS D[h*8 + j][c]
//
// The accumulator is COLUMN-distributed: a lane owns one column and eight consecutive rows, lanes
// 0-15 rows 0-7 and lanes 16-31 rows 8-15. The intuitive "a lane owns a row" is wrong, produces a
// silently transposed tile, and gives no compile or runtime error.
//
// IDENTITY, DIAGONAL AND SYMMETRIC TEST MATRICES CANNOT TELL THE TWO APART -- both give 0 errors.
// Any test of fragment code here must use asymmetric operands.
//
// ---- the K labelling ---------------------------------------------------------------------------
//
// The canonical hardware K-order is not contiguous: lanes 0-15 really hold k = 0,1,2,3,8,9,10,11.
// Calling slot j of lane-half h "k = 8h + j" above is a permutation of K applied IDENTICALLY to A
// and B, and D = sum_k A[m][k] B[k][n] is invariant under it, so the product is bit-identical. Two
// rules keep it that way:
//
//   1. BOTH OPERANDS must use this labelling. If one ever comes from GLOBAL_LOAD_TR_B128 (which
//      emits the canonical order) and the other from a plain load, the K labels do not line up and
//      the result is silently wrong. Every kernel in this family stages both operands through the
//      same `stage8` and reads both through `r4d_fp8_frag`, which is what makes that safe.
//   2. A SPARSE path may NOT use this labelling: SWMMAC's index bits are tied to the hardware K
//      groups. Nothing in this family is sparse.
//
// ---- hazards -----------------------------------------------------------------------------------
//
// A WMMA whose A, B or index overlaps the PREVIOUS WMMA's D needs at least one independent VALU
// instruction in between -- required for correct function, not a perf hint. The compiler inserts it
// for these intrinsics. Feeding D straight back as C of the same opcode is the fast path and is
// exempt, which is why these kernels keep one long accumulator chain of a single opcode rather than
// mixing WMMA types in the inner loop.
//
// And WMMA does NOT co-issue with VALU on RDNA4, for every opcode. Dequantisation and address math
// inside a wave are serial with its matrix work. That is the whole argument for an fp8 GEMM whose
// inner loop converts nothing: v_wmma_f32_16x16x16_fp8_fp8 issues at 412 TF/s (2.99 cycles), the
// same rate as v_wmma_i32_16x16x16_iu8, and here it is reached because both operands are already
// E4M3 bytes and nothing is unpacked between the LDS read and the mma.
#pragma once
#include "r4d_common.h"
#include "r4d_fp8_frag.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_fp8 {

// A/B operand widths are the RDNA4 ones -- half of RDNA3's, because RDNA4 dropped the cross-half
// replication. Passing a gfx11-width vector compiles for gfx1100 and fails here, which is the
// intended way to find out the unsuffixed builtin was used.
typedef int   i2v __attribute__((ext_vector_type(2)));   // fp8 A or B, 2 VGPRs
typedef float f8v __attribute__((ext_vector_type(8)));   // f32 accumulator, 8 VGPRs

constexpr uint32_t kM = 16, kN = 16, kK = 16;

// Always the _gfx12-suffixed builtin. The unsuffixed name is the RDNA3 form with different operand
// widths; it does not compile for gfx1201, which is the good outcome.
__device__ __forceinline__ f8v mma(i2v a, i2v b, f8v c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// THE INT8 TWIN, for the same kernels at i8 codes: v_wmma_i32_16x16x16_iu8, both operands signed,
// at the fp8 instruction's rate. Its int32 accumulator rides in the f8v registers as bits, so one
// accumulator type serves both instantiations; acc_val reads a slot back as the number it holds.
// A 128-K scale block of products is at most 128 * 127 * 127, far inside int32.
typedef int i8v __attribute__((ext_vector_type(8)));
template <bool I8>
__device__ __forceinline__ f8v mma8(i2v a, i2v b, f8v c) {
    if constexpr (I8)
        return __builtin_bit_cast(f8v, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                           true, a, true, b, __builtin_bit_cast(i8v, c), false));
    else
        return mma(a, b, c);
}
template <bool I8>
__device__ __forceinline__ float acc_val(float x) {
    if constexpr (I8) return (float)__builtin_bit_cast(int, x);
    else return x;
}

__device__ __forceinline__ uint32_t frag_c (uint32_t lane) { return lane & 15u; }
__device__ __forceinline__ uint32_t frag_k0(uint32_t lane) { return (lane >> 4) * 8u; }
// Row of D that accumulator slot `j` holds; the column is frag_c(lane).
__device__ __forceinline__ uint32_t acc_row(uint32_t lane, uint32_t j) {
    return (lane >> 4) * 8u + j;
}

__device__ __forceinline__ f8v acc_zero() {
    f8v z;
#pragma unroll
    for (int j = 0; j < 8; ++j) z[j] = 0.0f;
    return z;
}

// ---- E4M3 -> f32 -------------------------------------------------------------------------------
//
// ONE INSTRUCTION on gfx1201: v_cvt_f32_fp8. The software form is ~11 VALU ops and is the fallback
// for a target without the builtin. Over all 256 byte values the two agree on 254. The two that
// differ are 0x7F and 0xFF, E4M3-fn's NaN encodings -- the hardware returns NaN, as the format
// says, and the software form returns +-480. Nothing here writes those bytes (both quantisers
// clamp to +-448 = 0x7E), so the difference is inert on real data and turns a silent 480 into a
// visible NaN if that ever stops being true.
//
// There is no subnormal-free fast path here. That path is only
// correct where the encoder clamps entries to exponent 1..7 by construction, as an expert codebook
// can; an arbitrary checkpoint weight has no such precondition, and for a subnormal that path
// silently returns 2^-7 * 1.MMM instead of MMM * 2^-9. The hardware instruction handles them.
__device__ __forceinline__ float e4m3(uint32_t b) {
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
    return __builtin_amdgcn_cvt_f32_fp8(b, 0);
#else
    const uint32_t sign = (b & 0x80u) << 24;
    const uint32_t ex   = (b >> 3) & 0x0Fu;
    const uint32_t mn   = b & 0x07u;
    const float nrm = __uint_as_float(((ex + 120u) << 23) | (mn << 20));
    const float sub = (float)mn * (1.0f / 512.0f);
    return __uint_as_float(__float_as_uint(ex ? nrm : sub) | sign);
#endif
}

// TWO E4M3 bytes of a packed dword to two f32 in ONE instruction -- v_cvt_pk_f32_fp8, which gfx1201
// has because RDNA4 carries the fp8 WMMA path. `HI` selects bytes 2,3 over bytes 0,1 and must be a
// template parameter because the builtin's word select is a constant. Returns r4d_common.h's `v2f`,
// which is what the builtin actually returns.
template <bool HI>
__device__ __forceinline__ v2f e4m3_pk(uint32_t q) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    return __builtin_amdgcn_cvt_pk_f32_fp8(q, HI);
#else
    constexpr uint32_t s = HI ? 16u : 0u;
    v2f r;
    r.x = e4m3((q >> s) & 0xFFu);
    r.y = e4m3((q >> (s + 8u)) & 0xFFu);
    return r;
#endif
}

// ---- THE BLOCK SCALE: why the checkpoint's bf16 scales are read natively ------------------------
//
// Qwen3.8-27B-FP8 carries a BF16 `weight_scale_inv`, one per 128x128 tile in row-major order, and
// the decode is `__uint_as_float(s << 16)` -- ONE SHIFT, ONE BITCAST.
//
// The alternative encoding is one UE8M0 exponent BYTE per tile, decoded
// `__uint_as_float(e ? (e << 23) : 0u)`: the same shift and bitcast at a different
// distance. Converting to it would round every tile's gain to a power of two, and correcting for
// that means requantising the E4M3 bytes against the new scale -- half a bit off a 3-bit mantissa,
// everywhere, to save nothing. The scale plane is 21.8 KiB against an 89 MiB weight at gate_proj's
// shape.
//
// No zero test, unlike the UE8M0 form: exponent 0 encodes 2^-127 in that format's convention and
// needs one, while bf16 encodes zero as zero and `0u << 16` is +0.0f.
__device__ __forceinline__ float bscale(uint16_t s) {
    return __uint_as_float((uint32_t)s << 16);
}

// ---- LDS staging, shared by both fp8a8 instantiations -------------------------------------------
//
// An fp8 fragment is 8 bytes a lane, so the granule is 8 B and a 64 B row holds 8 of them. A
// fragment read is ONE granule column over SIXTEEN CONSECUTIVE ROWS, and the whole question is
// which bank each of those rows lands in.
//
// THE ROW PITCH IS NOT THE TILE WIDTH, and that is the point. LDS has 32 banks of 4 B, so bank =
// (byte address / 4) % 32 and a row pitch of P bytes puts row r at bank (r * P/4) % 32. With the
// tile width itself as the pitch -- 64 or 256, both multiples of 128 B -- that is 16r % 32 or
// 0, so rows 8 apart (or all of them) collide and the read serialises.
//
// An XOR granule swizzle -- `(k >> 3) ^ (row & 7)` -- does not solve it. It
// permutes only WITHIN a row, and a 64 B row holds 8 granules, so it can separate at most 8 rows:
// rows r and r+8 still land together and the sixteen lanes of a fragment read take two bank cycles
// rather than one. Fragment reads are a large share of this kernel's time, so the second cycle is
// not a small cost.
//
// A PADDED PITCH separates all sixteen instead, and needs no permutation at all. P = FK + 8 gives
// P/4 = 18 words at FK = 64, and 18r % 32 over r = 0..15 is
//
//     0 18 4 22 8 26 12 30 16 2 20 6 24 10 28 14
//
// -- sixteen distinct even banks, so the sixteen lanes cover all 32 banks exactly once and the
// read is one bank cycle. At FK = 256 it is 66 words and 66r % 32 = 2r % 32, distinct likewise.
// The pad must be a multiple of 8 to keep the 8 B granule aligned for ds_read_b64; +8 is the
// smallest that works, and a larger pad only costs LDS without separating any better -- +16 is
// four words, and 4r % 32 has period 8, so it collides again.
//
// Cost is 8 B a row of LDS: the prefill tile is 36 KiB a block rather than 32, which leaves room
// for 3 resident blocks a WGP rather than 4. The bank cycle saved outweighs the lost block.
template <uint32_t FK>
__device__ __forceinline__ constexpr uint32_t lds_pitch() { return FK + 8u; }

template <uint32_t FK>
__device__ __forceinline__ i2v frag(const uint8_t* __restrict__ lds, uint32_t row0, uint32_t k0,
                                    uint32_t lane) {
    const uint32_t row = row0 + frag_c(lane);
    const uint32_t k = k0 + frag_k0(lane);              // 8-aligned, so (k & 7) == 0
    const uint2 v = *(const uint2*)(lds + (size_t)row * lds_pitch<FK>() + k);
    i2v f;
    f[0] = (int)v.x;
    f[1] = (int)v.y;
    return f;
}

// The 32 bytes a thread stages, held in registers between the global load and the LDS store.
//
// They are SPLIT into a load and a store because of what sits between them. Issued as one call, the
// load is followed immediately by the barrier that publishes it, so a thread has exactly one
// 32-byte request in flight and then waits -- and the K loop becomes a chain of memory latencies
// rather than a stream. The symptom is a block sweep that saturates well below the card's coalesced
// ceiling and then does not move when the block count doubles.
struct Stage8 { uint4 v, w; };

// One of the FOUR fragments held in a 32-byte fragment-order load. `t` selects which: bytes 8t,
// and a Stage8 is two uint4, so fragment 0 and 1 come out of `v` and 2 and 3 out of `w`. `t` is a
// constant at every call site after unrolling, so this is a register rename and not a branch.
__device__ __forceinline__ i2v frag_of(const Stage8& s, uint32_t t) {
    const uint2 g[4] = { make_uint2(s.v.x, s.v.y), make_uint2(s.v.z, s.v.w),
                         make_uint2(s.w.x, s.w.y), make_uint2(s.w.z, s.w.w) };
    i2v f;
    f[0] = (int)g[t].x;
    f[1] = (int)g[t].y;
    return f;
}

/* `NT` asks the cache to evict this line first. It is for an operand THE KERNEL READS ONCE AND NO
 * OTHER BLOCK READS AT ALL -- the weight -- and it exists to stop that stream from evicting the one
 * operand every block DOES share. `__builtin_nontemporal_load` will not take `uint4`: that is a
 * HIP_vector_type class template, not a native vector, so the hinted load goes through an
 * ext_vector_type and the WHOLE value is bit-cast back (never a subscript of it, which
 * miscompiles). Nothing about the bytes changes -- this is a cache hint and the arithmetic that
 * follows is identical. */
typedef unsigned r4d_f8_ntu4 __attribute__((ext_vector_type(4)));

template <bool NT = false>
__device__ __forceinline__ Stage8 stage8_load(const uint8_t* __restrict__ src, bool live) {
    Stage8 r;
    r.v = make_uint4(0, 0, 0, 0);
    r.w = make_uint4(0, 0, 0, 0);
    if (live) {
        if constexpr (NT) {
            r.v = __builtin_bit_cast(uint4,
                      __builtin_nontemporal_load((const r4d_f8_ntu4*)src));
            r.w = __builtin_bit_cast(uint4,
                      __builtin_nontemporal_load((const r4d_f8_ntu4*)(src + 16)));
        } else {
            r.v = *(const uint4*)src;
            r.w = *(const uint4*)(src + 16);
        }
    }
    return r;
}

// The fragment-order counterpart: the 32 bytes a lane holds in a (16 row, 64 k) tile are four
// granules of ONE row at k `k0 + t*16`, not four consecutive ones. So a fragment-order operand is
// de-permuted on the way INTO LDS and everything downstream -- the padded pitch, the bank layout,
// frag() itself -- is untouched. That matters: the pitch is what makes a fragment read one bank
// cycle, and reading fragment order straight out of LDS would put eight lanes on one bank.
template <uint32_t FK>
__device__ __forceinline__ void stage8_store_frag(uint8_t* __restrict__ lds, uint32_t row,
                                                 uint32_t k0, Stage8 r) {
    const uint4 v = r.v, w = r.w;
    const uint2 gr[4] = { make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y),
                          make_uint2(w.z, w.w) };
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        *(uint2*)(lds + (size_t)row * lds_pitch<FK>() + k0 + i * kFragK) = gr[i];
}

template <uint32_t FK>
__device__ __forceinline__ void stage8_store(uint8_t* __restrict__ lds, uint32_t row, uint32_t k0,
                                             Stage8 r) {
    const uint4 v = r.v, w = r.w;
    const uint2 gr[4] = { make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y),
                          make_uint2(w.z, w.w) };
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t k = k0 + i * 8;
        *(uint2*)(lds + (size_t)row * lds_pitch<FK>() + k) = gr[i];
    }
}

constexpr uint32_t kMoeFusedDecodeMaxTokens = 16;
constexpr uint32_t kMoeFusedDecodeMaxInter  = 384;

}  // namespace r4d_fp8
