/* tests/r4d_emu: SIMT emulator shim that runs libr4d kernel source on the host. Each GPU thread
 * is a fiber on one OS thread; __syncthreads and cross-lane ops are barriers the scheduler
 * enforces, a barrier some threads never reach is reported as a deadlock, and lanes of a wave
 * that meet at different cross-lane ops as a divergence. Test-only. */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __constant__
#define __forceinline__ inline
#define __launch_bounds__(n)
/* R4D_WAVES (r4d_common.h) names a kernel attribute; on the host it becomes an empty attribute */
#define amdgpu_waves_per_eu(lo, hi)
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
typedef void* hipStream_t;
typedef int hipError_t;
enum { hipSuccess = 0 };
inline hipError_t hipGetLastError() { return 0; }
/* one device, with the R9700's 64 KiB of LDS a workgroup, which launchers size dynamic LDS by */
struct hipDeviceProp_t { size_t sharedMemPerBlock; };
inline hipError_t hipGetDevice(int* d) { *d = 0; return hipSuccess; }
/* no graphs: a stream never captures */
enum hipStreamCaptureStatus { hipStreamCaptureStatusNone = 0 };
inline hipError_t hipStreamIsCapturing(hipStream_t, hipStreamCaptureStatus* s) {
    *s = hipStreamCaptureStatusNone;
    return hipSuccess;
}
/* every launch runs to completion before it returns */
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
/* __device__ variables are host globals, so a symbol copy is a memcpy */
enum hipMemcpyKind { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
#define HIP_SYMBOL(s) (&(s))
inline hipError_t hipMemcpyFromSymbol(void* dst, const void* sym, size_t n, size_t off, hipMemcpyKind) {
    std::memcpy(dst, (const char*)sym + off, n);
    return hipSuccess;
}
inline hipError_t hipMemcpyToSymbol(const void* sym, const void* src, size_t n, size_t off, hipMemcpyKind) {
    std::memcpy((char*)sym + off, src, n);
    return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int) {
    *p = hipDeviceProp_t{};
    p->sharedMemPerBlock = 64 * 1024;
    return hipSuccess;
}

struct EmuIdx { unsigned x, y, z; };
EmuIdx& emu_thread_idx();
EmuIdx& emu_block_idx();
EmuIdx& emu_block_dim();
EmuIdx& emu_grid_dim();
#define threadIdx (emu_thread_idx())
#define blockIdx (emu_block_idx())
#define blockDim (emu_block_dim())
#define gridDim (emu_grid_dim())
void __syncthreads();
/* the wave barrier (__syncwarp): the lanes of one wave, which is 32 here */
void emu_syncwarp();
#define __syncwarp() emu_syncwarp()
/* the block barrier, returning whether any thread passed a nonzero predicate */
int __syncthreads_or(int p);
int emu_bpermute(int addr, int v);
int emu_readlane(int v, int lane);
#define __builtin_amdgcn_ds_bpermute(a, v) emu_bpermute((a), (v))
#define __builtin_amdgcn_readlane(v, l) emu_readlane((v), (l))
unsigned emu_ballot(bool p);
#define __builtin_amdgcn_ballot_w32(p) emu_ballot((p))
/* HIP's __ballot: the wave32 mask, zero-extended */
inline unsigned long long __ballot(int p) { return emu_ballot(p != 0); }
inline int __popcll(unsigned long long v) { return __builtin_popcountll(v); }
/* v_readfirstlane_b32: a cross-lane op is a wave barrier here, so every lane is active, lane 0 first */
#define __builtin_amdgcn_readfirstlane(v) emu_readlane((v), 0)
/* DPP16 row_shl, row_shr and row_xmask with every row and bank written; a source outside the row
 * reads 0 under bound_ctrl, else leaves `old`. Other controls and masks are reported. */
int emu_update_dpp(int old, int src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl);
#define __builtin_amdgcn_update_dpp(o, s, c, rm, bm, bc) emu_update_dpp((o), (s), (c), (rm), (bm), (bc))
/* v_permlanex16_b32: lane L reads the lane of the other row of sixteen that L's selector names */
int emu_permlanex16(int old, int src, unsigned sel_lo, unsigned sel_hi, bool fi, bool bc);
#define __builtin_amdgcn_permlanex16(o, s, lo, hi, fi, bc) emu_permlanex16((o), (s), (lo), (hi), (fi), (bc))
/* v_perm_b32: result byte i is the byte of {a, b} (b low) that selector byte i names; 8-11 a sign
 * byte (bit 15 or 31 of b, then of a), 12 zero, 13 and up 0xff */
inline unsigned emu_perm(unsigned a, unsigned b, unsigned sel) {
    const unsigned long long d = ((unsigned long long)a << 32) | b;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (8 * i)) & 0xffu;
        unsigned v = s >= 13 ? 0xffu : s == 12 ? 0u : s >= 8 ? ((d >> (16 * (s - 8) + 15)) & 1u) * 0xffu
                                                           : (unsigned)(d >> (8 * s)) & 0xffu;
        r |= v << (8 * i);
    }
    return r;
}
#define __builtin_amdgcn_perm(a, b, s) emu_perm((a), (b), (s))
/* HIP's __byte_perm: result byte i is the byte of {y, x} (x low) that the low three bits of
 * nibble i of s name; only the low four nibbles are read */
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    const unsigned long long d = ((unsigned long long)y << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) r |= (unsigned)((d >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
}
/* s_barrier is the workgroup barrier. Fibers share one memory and switch only at barriers, so a
 * fence or a scheduling barrier orders nothing more. */
#define __builtin_amdgcn_s_barrier() __syncthreads()
#define __builtin_amdgcn_fence(...) ((void)0)
#define __builtin_amdgcn_sched_barrier(m) ((void)0)
/* LDS atomics: fibers switch only at barriers, so a plain read-modify-write is indivisible */
template <class T>
inline T atomicAdd(T* p, T v) { const T o = *p; *p = o + v; return o; }
template <class T>
inline T atomicMin(T* p, T v) { const T o = *p; if (v < o) *p = v; return o; }
template <class T>
inline T atomicMax(T* p, T v) { const T o = *p; if (v > o) *p = v; return o; }
template <class T>
inline T atomicCAS(T* p, T expect, T v) { const T o = *p; if (o == expect) *p = v; return o; }
template <class T>
inline T atomicExch(T* p, T v) { const T o = *p; *p = v; return o; }
/* v_wmma_f32_16x16x16_bf16 (gfx12, wave32): A[M][K] lane M + 16 * (K / 8) element K % 8; B[K][N]
 * lane N + 16 * (K / 8) element K % 8; D[M][N] lane N + 16 * (M / 8) element M % 8 */
typedef short emu_v8s __attribute__((ext_vector_type(8)));
typedef float emu_v8f __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c);
/* the __bf16 fragments of r4d_gdn_wmma.h, the same bits */
typedef __bf16 emu_v8bf __attribute__((ext_vector_type(8)));
inline emu_v8f emu_wmma_bf16(emu_v8bf a, emu_v8bf b, emu_v8f c) {
    return emu_wmma_bf16(__builtin_bit_cast(emu_v8s, a), __builtin_bit_cast(emu_v8s, b), c);
}
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) emu_wmma_bf16((a), (b), (c))
typedef _Float16 emu_v8h __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) emu_wmma_f16((a), (b), (c))
/* v_wmma_f32_16x16x16_fp8_fp8 (gfx12, wave32): the bf16 form's layout with eight E4M3 bytes a
 * lane, element e in byte e. Each output is c plus its 16 products in k order: not the hardware's
 * internal order, but a function of the output's own row, column and c alone, as the hardware's
 * is, which is what a comparison of two kernels over the same fragments relies on. */
typedef int emu_v2i __attribute__((ext_vector_type(2)));
emu_v8f emu_wmma_fp8(emu_v2i a, emu_v2i b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c) emu_wmma_fp8((a), (b), (c))
/* v_wmma_i32_16x16x16_iu8: not emulated; reported when reached */
typedef int emu_v8i __attribute__((ext_vector_type(8)));
emu_v8i emu_wmma_iu8(bool sa, emu_v2i a, bool sb, emu_v2i b, emu_v8i c, bool clamp);
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, cl) \
    emu_wmma_iu8((sa), (a), (sb), (b), (c), (cl))
/* global_load_tr_b128: not emulated; reported when reached */
emu_v8s emu_load_tr_b128(const void* p);
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) emu_load_tr_b128((const void*)(p))
/* v_cvt_pk_fp8_f32 (gfx12, OCP E4M3): converts a and b into the low half of `old` (hi false) or
 * its high half, keeping the other half; round to nearest even. Matches the saturating encoder
 * rad_f32_to_fp8e4m3 on inputs a kernel has already clamped to +-448. */
int emu_cvt_pk_fp8(float a, float b, int old, bool hi);
#define __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi) emu_cvt_pk_fp8((a), (b), (old), (hi))
extern int emu_last_block;     /* threads a block of the last grid had: which kernel ran */

/* Test-side scheduling: hold block `block` of the next grid at its nth cross-lane exchange (a
 * __shfl, __syncwarp or shuffle builtin) while the rest of the grid runs, then let it go. Returns
 * a handle for emu_park_hit. A park stays armed until emu_park_clear; a hit one never parks twice. */
int emu_park(int block, unsigned nth);
int emu_park_hit(int park);
void emu_park_clear();

/* Blocks run one after another on one OS thread, so a fence orders nothing that is not already in
 * order and an atomic is a plain read-modify-write. Enough for the split-K arrival pattern
 * (r4d_lora_bf16.hip): the last block of a group in launch order is the one that reduces. */
inline void __threadfence() {}
#define __HIP_MEMORY_SCOPE_AGENT 3
#define __hip_atomic_fetch_add(p, v, order, scope) emu_atomic_fetch_add((p), (v))
template <typename T> inline T emu_atomic_fetch_add(T* p, T v) { const T o = *p; *p = o + v; return o; }
/* The device allocations a unit makes for itself (counter blocks): host memory here. */
inline hipError_t hipMalloc(void** p, size_t n) { *p = std::calloc(n ? n : 1, 1); return *p ? 0 : 2; }
inline hipError_t hipFree(void* p) { std::free(p); return 0; }
inline hipError_t hipMemset(void* p, int v, size_t n) { std::memset(p, v, n); return 0; }
inline hipError_t hipMemsetAsync(void* p, int v, size_t n, hipStream_t) { std::memset(p, v, n); return 0; }
/* Device memory is host memory here, so a copy is one memcpy; the kind is not consulted. */
inline hipError_t hipMemcpy(void* dst, const void* src, size_t n, hipMemcpyKind) {
    std::memcpy(dst, src, n);
    return 0;
}

struct alignas(8)  uint2  { unsigned x, y; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(16) int4   { int x, y, z, w; };
struct alignas(16) float4 { float x, y, z, w; };
struct uchar4 { unsigned char x, y, z, w; };
inline uchar4 make_uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
    return uchar4{a, b, c, d};
}
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return uint2{a, b}; }
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }
inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }   /* v_rsq_f32 is within an ulp of it */
/* The device's fast exponential is not bit-exact to expf; here it is expf, so a test compares two
 * kernels that both take it, never a kernel against a host value of it. */
inline float __expf(float x) { return std::exp(x); }
inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int __float_as_int(float f) { int u; std::memcpy(&u, &f, 4); return u; }
/* The float shuffles over ds_bpermute, for one-dimensional blocks, within groups of `width`
 * lanes: __shfl_xor reads lane L ^ mask, __shfl lane `src`, __shfl_down lane L + delta (its own
 * value past the group's end) */
inline float emu_shfl_from(float v, int src) {
    return __int_as_float(emu_bpermute(src << 2, __float_as_int(v)));
}
inline float __shfl_xor(float v, int mask, int width = 32) {
    const int lane = (int)threadIdx.x & 31;
    return emu_shfl_from(v, (lane & ~(width - 1)) | ((lane ^ mask) & (width - 1)));
}
inline float __shfl(float v, int src, int width = 32) {
    const int lane = (int)threadIdx.x & 31;
    return emu_shfl_from(v, (lane & ~(width - 1)) | (src & (width - 1)));
}
inline float __shfl_down(float v, unsigned delta, int width = 32) {
    const int lane = (int)threadIdx.x & 31, in_group = lane & (width - 1);
    return emu_shfl_from(v, in_group + (int)delta < width ? lane + (int)delta : lane);
}
/* HIP declares the classification functions in the global namespace */
using std::isfinite;
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

/* one block at a time, every thread a fiber; `dyn_lds` bytes of dynamic LDS */
void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx, size_t dyn_lds = 0);
template <typename K, typename A>
struct EmuCall { K k; A a; static void run(void* p) { auto* c = (EmuCall*)p; c->k(c->a); } };
template <typename A>
void emu_launch(void (*k)(A), dim3 g, dim3 b, A a) {
    EmuCall<void (*)(A), A> c{k, a};
    emu_run_grid(g, b, &EmuCall<void (*)(A), A>::run, &c);
}
#define R4D_KLAUNCH(kernel, grid, block, stream, ...) emu_launch(&kernel, grid, block, __VA_ARGS__)

/* R4D_EMIT (r4d_emit.h): every argument converted to its parameter's type, as r4d_emit does, and
 * `lds` bytes of dynamic LDS; the kernel's name is kept in emu_last_kernel. Returns RAD_OK. */
extern const char* emu_last_kernel;
template <class... P, class... A>
int emu_emit(const char* name, void (*k)(P...), dim3 g, dim3 b, size_t lds, const A&... a) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    struct Call {
        void (*k)(P...);
        std::tuple<P...> v;
        static void run(void* p) { Call* c = (Call*)p; std::apply(c->k, c->v); }
    };
    Call c{k, std::tuple<P...>(static_cast<P>(a)...)};
    emu_last_kernel = name;
    emu_run_grid(g, b, &Call::run, &c, lds);
    return 0;
}
#define R4D_EMIT(stream, kernel, grid, block, lds, ...) \
    emu_emit(#kernel, (kernel), (grid), (block), (size_t)(lds), __VA_ARGS__)
/* hipLaunchKernelGGL, for units that launch with it directly: the same call as R4D_EMIT. */
#define hipLaunchKernelGGL(kernel, grid, block, lds, stream, ...) \
    ((void)(stream), (void)emu_emit(#kernel, (kernel), (grid), (block), (size_t)(lds), __VA_ARGS__))
/* R4D_DYN_LDS (r4d_launch.h): the launch's dynamic LDS, poisoned at each block's start and ending
 * at an inaccessible page (its start 16-byte aligned), so a read before any write shows and one
 * past the end faults */
void* emu_dyn_lds();
#define R4D_DYN_LDS(type, name) type* const name = (type*)emu_dyn_lds()
