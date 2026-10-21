// DFlash2 grouped dynamic depthwise convolution -- device math only.
//
// Deliberately free of <hip/hip_runtime.h> so the identical source can be compiled standalone to
// gfx1201 assembly with a stock clang (no ROCm install) for ISA review, and as part of r4d.so.
//
// The op, per DFlash2's _grouped_conv with taps=2, group_size=16:
//
//   out[t,h] = (base[0,h] + delta[t,0,g]) * x[t,h]
//            + (base[1,h] + delta[t,1,g]) * x[t-1,h] * ((t % block) >= 1)     g = h / group
//
// The block position is t % block_size for ANY block size (block = 1 + num_speculative_tokens,
// 7 at the 6-draft serve, 8 at 7): one scalar remainder per workgroup, hoisted out of the lane
// work. An earlier build masked with block_size - 1 and refused non-powers of two.
//
// The reference builds this as ~6 full [T,H] elementwise passes plus a materialised
// [T, taps, num_groups, group] coefficient tensor. Every one of those intermediates is consumed
// exactly once, so the fused form reads x, delta and base once and writes out once.
#pragma once

typedef unsigned int   r4d_u32;
typedef unsigned short r4d_u16;
typedef r4d_u32 r4d_u32x4 __attribute__((ext_vector_type(4)));

// A 16-byte load addressed both ways. The subscript must come off a union member and never off a
// vector via __builtin_bit_cast: bit-casting a vector SUBSCRIPT silently yields element 0's bits.
union R4dV8 { r4d_u32x4 v; r4d_u16 h[8]; };

// Under hipcc these are device functions; the standalone ISA harness compiles the same source with
// a plain clang targeting amdgcn, where there is no __device__ and everything is already device code.
#ifndef R4D_DEV
#ifdef __HIP__
#define R4D_DEV __device__ __forceinline__
#else
#define R4D_DEV static __attribute__((always_inline))
#endif
#endif

R4D_DEV float r4d_bf2f(r4d_u16 v) {
    return __builtin_bit_cast(float, (r4d_u32)v << 16);
}

// f32 -> bf16 RTNE in software: gfx1201 has no bf16 convert instruction.
R4D_DEV r4d_u16 r4d_f2bf(float f) {
    r4d_u32 u = __builtin_bit_cast(r4d_u32, f);
    return (r4d_u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One thread owns 8 contiguous channels: a single dwordx4 in and out. group_size 16 divides that
// span, so the group index -- and therefore both delta scalars -- are uniform over the thread's 8
// channels and are loaded once rather than per channel.
template <int TAPS, int GROUP>
R4D_DEV void r4d_dflash_conv_body(const r4d_u16* __restrict__ x,
                                  const r4d_u16* __restrict__ delta,
                                  const r4d_u16* __restrict__ base,
                                  r4d_u16* __restrict__ out,
                                  int H, int dpitch, int NG, int vpr,
                                  int block_size, int t, int hv) {
    if (hv >= vpr) return;
    // Position within the draft block: wave-uniform (t is the workgroup's y index), one scalar
    // remainder. Powers of two lower to a mask by the compiler when block_size is a literal; at
    // runtime it is one s_mul/s_sub pair either way.
    const int bpos = t % block_size;

    // Split every address into a wave-uniform pointer and a lane offset the backend can keep in a
    // single 32-bit VGPR -- gfx12's saddr form. Two things are load-bearing:
    //
    //   * the mask. The backend will only fold a lane offset into saddr if it can prove the
    //     byte-scaled offset stays inside 32 bits. A lane index built from workgroup_id has no
    //     known bound, so it widens the address to a 64-bit vector add and pays a
    //     v_add_co_u32 / v_add_co_ci_u32 pair per load. __builtin_assume does NOT propagate here;
    //     the AND does, for one cheap VALU op.
    //   * pointer + offset, NOT a vector-typed array indexed by lane. The array form makes the
    //     backend materialise a separate 64-bit address per pointer instead of sharing one offset.
    const unsigned off = ((unsigned)hv & 0xFFFFFu) * 8u;
    const unsigned g   = off / (unsigned)GROUP;

    const r4d_u16* __restrict__ xrow = x + (long)t * H;
    r4d_u16*       __restrict__ orow = out + (long)t * H;
    const r4d_u16* __restrict__ drow = delta + (long)t * dpitch;

    R4dV8 x0; x0.v = *(const r4d_u32x4*)(xrow + off);
    R4dV8 b0; b0.v = *(const r4d_u32x4*)(base + off);
    const float d0 = r4d_bf2f(drow[g]);

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc[j] = (r4d_bf2f(b0.h[j]) + d0) * r4d_bf2f(x0.h[j]);

    // Tap `tap` contributes only where the position within the block is at least `tap`. t is the
    // workgroup's y index, so the predicate is wave-uniform and the skip is a scalar branch --
    // and on the rows it skips, the shifted load is never issued at all.
#pragma unroll
    for (int tap = 1; tap < TAPS; ++tap) {
        // Same rule as above, applied to the shifted operands: every uniform term folds into its
        // own pointer so the only thing left varying per lane is the bounded offset. Adding tap*NG
        // into the delta INDEX instead of the pointer is enough to lose saddr again. These are
        // hoisted OUT of the predicate: they are scalar address arithmetic on a path the backend
        // will not fold into saddr from inside a branch, and an address that is merely formed and
        // never dereferenced costs nothing.
        const r4d_u16* __restrict__ xprev = xrow - (long)tap * H;
        const r4d_u16* __restrict__ btap  = base + (long)tap * H;
        const r4d_u16* __restrict__ dtap  = drow + (long)tap * NG;
        if (bpos >= tap) {
            R4dV8 xs; xs.v = *(const r4d_u32x4*)(xprev + off);
            R4dV8 bt; bt.v = *(const r4d_u32x4*)(btap + off);
            const float dt = r4d_bf2f(dtap[g]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] += (r4d_bf2f(bt.h[j]) + dt) * r4d_bf2f(xs.h[j]);
        }
    }

    R4dV8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.h[j] = r4d_f2bf(acc[j]);
    *(r4d_u32x4*)(orow + off) = o.v;
}
