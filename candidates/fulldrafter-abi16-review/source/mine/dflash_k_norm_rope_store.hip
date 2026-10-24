/* dflash_k_norm_rope_store.hip -- the DFlash2 layer's K chain on the device: RMS norm -> NeoX
 * RoPE -> paged BF16 KV store, in one launch per (token, kv head).
 *
 * ============================== THE BASELINE'S OWN PRIMITIVES ARE INCLUDED ==============================
 *
 * `libr4d/r4d_common.h` is header-only, self-contained (hip_runtime + cstdint), and gives
 * `f32_to_bf16`, `bf16_to_f32`, `r4d_wave_sum` and `r4d_act_threads` as `__device__ __forceinline__`
 * functions, so including it needs no libr4d link. mine/dflash_conv_add.hip already takes a libr4d
 * header by relative path for the same reason. RE-SPELLING ANY OF THESE WAS THE BUG: a local
 * conversion that "matched on finite values" is not the baseline's conversion, and a bit-exact
 * fusion does not get to assume its inputs are finite. The rounding rule is now the chain's own
 * bytes, not a transcription of them.
 *
 * ============================== WHY THE LAYOUT IS ONE ELEMENT A LANE ==============================
 *
 * The tempting layout gives a lane several consecutive elements and reduces per wave: no LDS, no
 * barrier. It is also a DIFFERENT FUNCTION. libr4d's rmsnorm_kernel says it outright -- the width
 * decides which columns a thread folds into the sum of squares, and two kernels that disagree about
 * it disagree in the last bits of the norm. A ULP in the mean square is a ULP in the normed value,
 * and after the rotation's subtraction of two products that ULP can surface as a few percent where
 * the terms nearly cancel.
 *
 * THE REFERENCE'S FOLD IS REPRODUCIBLE AT 128 LANES:
 *
 *   * r4d_act_threads(M, 128) is 1024, 512 or 256 (libr4d/r4d_common.h:250-254), never less than
 *     the head width, so `for (i = threadIdx.x; i < n; i += THR)` gives every thread AT MOST one
 *     element -- and the threads beyond column 127 contribute exactly 0.0f.
 *   * block_sum (libr4d/r4d_model_bf16.hip:144-156) folds each wave with r4d_wave_sum and then sums
 *     the wave partials in ASCENDING wave index. Threads 128..THR-1 are whole waves of zeros, and
 *     adding +0.0f to a partial is exact, so the reference reduces to four partials in ascending
 *     order -- exactly what a 128-thread block sums.
 *   * r4d_wave_sum is a fixed DPP sequence (xor 16, 8, 4, 2, 1) and is CALLED, not copied: the same
 *     lanes, the same order, therefore the same bits.
 *
 * ONE ASSUMPTION IS NAMED RATHER THAN HIDDEN: the four-wave reduction is exact only while every
 * partial is finite, because `+0.0f` is exact for a finite addend and an INFINITE partial would be
 * poisoned by a NaN partial from the zero waves. The baseline has the identical exposure with its
 * own zero threads, so the two still agree bit for bit -- what is not claimed is that either is
 * numerically meaningful for a row whose sum of squares overflows. The comparator covers Inf/NaN
 * rows explicitly (see the handoff).
 *
 * ============================== THE TWO BF16 ROUNDINGS ==============================
 *
 * The unfused path stores the norm as bf16 and rope reads it back, then stores the rotation as bf16
 * and kv_store copies those bytes. Both roundings are reproduced here: the value handed to
 * `sincosf`'s products is the ROUNDED norm widened back to f32, and the cache receives the SAME
 * bf16 the store of `k` receives.
 *
 * ============================== RoPE ==============================
 *
 * rope_kernel (libr4d/r4d_model_bf16.hip:544-570) is reproduced at the same arguments: NeoX
 * pairing, `inv = 1.0f / powf(theta, (float)(2*i) / (float)rot)`, the position divided by `scale`
 * in f32, and `sincosf`. There is NO cos/sin table: computing the angles costs two transcendental
 * calls a pair and saves a buffer, a lifetime and a second definition of the angles.
 *
 * A LANE OWNS A PAIR, not an element. With rot == head_dim, element i and i + head_dim/2 are both
 * loaded and both produced by lane i < head_dim/2, so there is no shuffle, no LDS exchange and no
 * read-after-in-place hazard: the partner is in the lane's own registers.
 *
 * ============================== THE STORE ==============================
 *
 * kv_store_kernel's bf16 branch, element for element: the paged layout is
 * [n_blocks, n_head_kv, block_size, 2*head_dim] with K at [.., d] and V at [.., head_dim + d], and
 * a NEGATIVE slot stores nothing. A slot at or past the last block also stores nothing -- that is
 * the device store's rule (libr4d/r4d_model_bf16.hip:596) and it is what this op reproduces; see the
 * header note in the .cpp on why libref can refuse it instead.
 *
 * V IS COPIED AS BITS. kv_store_bf16 stores the operand's uint16 unchanged (r4d_model_bf16.hip:699),
 * so the copy is the contract; the dtype is checked in the shim rather than re-narrowed here.
 */
#include <hip/hip_runtime.h>

#include "../libr4d/r4d_common.h"
#include "dflash_k_norm_rope_store.h"

namespace {

constexpr int kThreads = 128;          /* == head_dim: one element a lane, no tail */
constexpr int kWave    = 32;
constexpr int kWaves   = kThreads / kWave;
constexpr int kHeadDim = 128;

/* THE REDUCTION ARGUMENT DEPENDS ON THESE TWO. A block narrower than the head would fold several
 * columns into one thread's accumulator; a block wider would add extra zero partials, which is
 * exact but would make the four-wave claim above a five-wave one. Both are stated here so a future
 * width cannot quietly invalidate the comment. */
static_assert(kThreads == kHeadDim, "one element a lane: the block IS the head");
static_assert(kWaves == 4, "block_sum's ascending partials are four here, as at the reference width");

struct Args {
    uint16_t*       k;
    const uint16_t* v;
    const void*     w;
    const int32_t*  pos;
    const int32_t*  slot;
    uint16_t*       kv;
    int64_t k_pitch, v_pitch;
    int     n_head_kv, block_size, n_blocks, w_f32;
    float   eps, wadd, theta, rscale;
};

/* ============================== ORDERED VALU ==============================
 *
 * When BOTH operands are NaN, v_mul/v_add/v_sub return the FIRST source operand's NaN (S0). Which
 * operand the compiler places in S0 is register allocation, so two kernels built from the same
 * expression can disagree in a NaN payload -- and here that is a gate failure, not a tolerance.
 * `kstore/exceptional-gain` failed exactly this way, twice.
 *
 * The `_s0` helpers name the operand that must land in S0, so the order is the source text's and
 * not the allocator's. One instruction each: no predicate, no select, nothing extra on the measured
 * path.
 *
 * THE ORDERS ARE THE REFERENCE'S, taken per instantiation from the shipped
 * `libr4d/CMakeFiles/libr4d.dir/r4d_model_bf16.hip.o` (fatbin member
 * `hipv4-amdgcn-amd-amdhsa--gfx1201`), not from one assumed variant:
 *
 *   rmsnorm_kernel<WF32,THR>, all SIX instantiations (f32/bf16 gain x 1024/512/256 threads):
 *     scale in S0 of the first product, `g + wadd` in S0 of the second, `wadd` in S0 of the add.
 *     e.g. <true,1024> `v_dual_mul_f32 v7, s1, v7` then `v_mul_f32_e32 v7, v8, v7`;
 *          <false,256> `v_dual_mul_f32 v13, s1, v13` then `v_dual_mul_f32 v12, v12, v13`.
 *
 *   rope_kernel, ONE instantiation (kThreads 256): trig in S0 of all four products,
 *   `c*x0` in S0 of the sub, `c*x1` in S0 of the add:
 *     v_mul_f32_e32  v8, v0, v1   ; s*x0    v_dual_mul_f32 v0, v0, v6 ; s*x1
 *     v_dual_mul_f32 v1, v7, v1   ; c*x0    v_mul_f32_e32  v6, v7, v6 ; c*x1
 *     v_sub_f32_e32  v0, v1, v0   ; c*x0 - s*x1
 *     v_add_f32_e32  v1, v6, v8   ; c*x1 + s*x0
 *
 * Every one of these is exact for finite and infinite inputs -- mul/add are commutative, sub
 * anti-commutative -- so no already-passing case moves. The orders matter only when both operands
 * are NaN, which is what a NaN norm scale across a whole head produces.
 */
__device__ __forceinline__ float mul_s0(float s0, float s1) {
    float d;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(d) : "v"(s0), "v"(s1));
    return d;
}
__device__ __forceinline__ float add_s0(float s0, float s1) {
    float d;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(d) : "v"(s0), "v"(s1));
    return d;
}
__device__ __forceinline__ float sub_s0(float s0, float s1) {
    float d;
    asm("v_sub_f32_e32 %0, %1, %2" : "=v"(d) : "v"(s0), "v"(s1));
    return d;
}

/* `block_sum`'s tail, transcribed from libr4d/r4d_model_bf16.hip:144-156 -- the wave butterfly and
 * the ascending partial sum. It is the one piece of the reference fold small enough that calling
 * it is clearer than reaching for a template whose launch width we do not otherwise use, and it
 * uses r4d_wave_sum, which IS the baseline's own. */
__device__ __forceinline__ float four_wave_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    if ((threadIdx.x & (kWave - 1)) == 0) lds[threadIdx.x / kWave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int wv = 0; wv < kWaves; ++wv) s += lds[wv];
    return s;
}

__global__ __launch_bounds__(kThreads) void kns_kernel(Args a) {
    __shared__ float lds[kWaves];
    constexpr int HD   = kHeadDim;
    constexpr int HALF = HD / 2;

    const int64_t t   = blockIdx.x;          /* token */
    const int64_t h   = blockIdx.y;          /* kv head */
    const int64_t l   = threadIdx.x;         /* column == lane; the block IS the head */

    uint16_t* kin = a.k + t * a.k_pitch + h * HD;

    /* ---- rmsnorm's sum of squares, folded exactly as libr4d folds it ---- */
    const float vv = bf16_to_f32(kin[l]);
    const float sum = four_wave_sum(vv * vv, lds);
    const float inv = rsqrtf(sum / (float)HD + a.eps);

    /* The gain, at the operand's own width; `+ wadd` in f32 before the multiply, as
     * rmsnorm_kernel does (`g + wadd` inside the f32 expression, never in the weight's dtype). */
    auto gain = [&](int64_t i) -> float {
        return a.w_f32 ? ((const float*)a.w)[i] : bf16_to_f32(((const uint16_t*)a.w)[i]);
    };
    /* THE NORM'S STORE, reproduced as a value with the BASELINE'S conversion: the unfused pair
     * writes bf16 here and the rotation reads it back, so every use of a normed element below is
     * this rounded value. */
    auto normed = [&](int64_t i) -> float {
        const float x = bf16_to_f32(kin[i]);
        /* The reference's S0 roles: scale, then `wadd` in the add, then the gain (see above). */
        const float xs = mul_s0(inv, x);
        const float gw = add_s0(a.wadd, gain(i));
        return bf16_to_f32(f32_to_bf16(mul_s0(gw, xs)));
    };

    /* ---- rope, one lane a pair, exactly rope_kernel's arithmetic ---- */
    const float div = a.rscale != 0.0f ? a.rscale : 1.0f;
    const float p   = (float)a.pos[t] / div;

    uint16_t k0 = 0, k1 = 0;
    if (l < HALF) {
        const float fr = 1.0f / powf(a.theta, (float)(2 * l) / (float)HD);
        float sn, cs;
        sincosf(p * fr, &sn, &cs);
        const float x0 = normed(l);
        const float x1 = normed(l + HALF);
        /* rope_kernel's S0 roles: trig first in every product, `c*x0` first in the sub, `c*x1`
         * first in the add (see the ORDERED VALU note above). */
        k0 = f32_to_bf16(sub_s0(mul_s0(cs, x0), mul_s0(sn, x1)));
        k1 = f32_to_bf16(add_s0(mul_s0(cs, x1), mul_s0(sn, x0)));
    }

    /* The reduction barrier completed all original K reads. Each active lane now owns
     * its disjoint pair, so the in-place stores need no second block barrier. */
    if (l < HALF) {
        kin[l]        = k0;
        kin[l + HALF] = k1;
    }

    /* ---- kv_store's bf16 branch ---- */
    const int32_t sl = a.slot[t];
    if (sl < 0 || sl / a.block_size >= a.n_blocks) return;
    if (l >= HALF) return;

    const int64_t blk = sl / a.block_size;
    const int64_t off = sl % a.block_size;
    uint16_t* c = a.kv + (((int64_t)blk * a.n_head_kv + h) * a.block_size + off) * (2 * HD);
    const uint16_t* vrow = a.v + t * a.v_pitch + h * HD;

    c[l]        = k0;
    c[HALF + l] = k1;
    c[HD + l]   = vrow[l];
    c[HD + HALF + l] = vrow[HALF + l];
}

}  /* namespace */

/* THE NARROWING BOUNDS ARE CHECKED HERE AS WELL AS AT ISSUE, and this is not belt-and-braces: a
 * launch may be reached with a band-resolved parameter list whose `M` is the BAND and whose operand
 * carries the real row count, and a tool may call the entry point directly. Every value the kernel
 * narrows to uint32 for the grid is bounded here, and the pitches are bounded so the kernel's own
 * `t * k_pitch` cannot wrap. */
int rad_mine_dflash_k_store_launch(const RadArgs* a, RadStream stream) {
    RadDflashKStoreCall call{};
    const int rc = rad_dflash_k_store_args(a, &call);
    if (rc != RAD_OK) return rc;

    if (call.rows <= 0 || call.rows > (int64_t)UINT32_MAX) return RAD_E_SHAPE;
    if (call.n_head_kv <= 0 || call.n_head_kv > (int64_t)UINT32_MAX) return RAD_E_SHAPE;
    if (call.n_blocks <= 0 || call.n_blocks > INT32_MAX) return RAD_E_SHAPE;
    if (call.block_size <= 0 || call.block_size > INT32_MAX) return RAD_E_SHAPE;
    /* The grid is (rows, n_head_kv) and every index is `t*k_pitch + h*head_dim`, so the products
     * are bounded before they are formed rather than trusted. */
    if (call.rows > (int64_t)UINT32_MAX / call.n_head_kv) return RAD_E_SHAPE;
    if (call.k_pitch <= 0 || call.rows > INT64_MAX / call.k_pitch) return RAD_E_SHAPE;
    if (call.v_pitch <= 0 || call.rows > INT64_MAX / call.v_pitch) return RAD_E_SHAPE;

    Args args{};
    args.k = static_cast<uint16_t*>(call.k);
    args.v = static_cast<const uint16_t*>(call.v);
    args.w = call.w;
    args.pos = call.pos;
    args.slot = call.slot;
    args.kv = static_cast<uint16_t*>(call.kv);
    args.k_pitch = call.k_pitch;
    args.v_pitch = call.v_pitch;
    args.n_head_kv = (int)call.n_head_kv;
    args.block_size = (int)call.block_size;
    args.n_blocks = (int)call.n_blocks;
    args.w_f32 = call.w_f32;
    args.eps = call.eps;
    args.wadd = call.wadd;
    args.theta = call.theta;
    args.rscale = call.rscale;

    /* ONE BLOCK A (TOKEN, KV HEAD): the head's reduction is a block reduction and its 128 columns
     * are 128 lanes, so grid.y names the head and nothing in the kernel indexes across heads. */
    void* kargs[] = { &args };
    const hipError_t e = hipLaunchKernel((const void*)kns_kernel,
                                         dim3((unsigned)call.rows, (unsigned)call.n_head_kv),
                                         dim3(kThreads), kargs, 0, (hipStream_t)stream);
    return e == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_dflash_k_store_count(void) { return 1; }

const RadKernelInfo* rad_mine_dflash_k_store_at(int i) {
    static const RadKernelInfo row = rad_dflash_k_store_row(rad_mine_dflash_k_store_launch);
    return i == 0 ? &row : nullptr;
}
