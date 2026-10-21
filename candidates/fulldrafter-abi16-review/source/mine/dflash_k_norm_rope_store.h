#ifndef RAD_MINE_DFLASH_K_NORM_ROPE_STORE_H
#define RAD_MINE_DFLASH_K_NORM_ROPE_STORE_H

#include "rad_plugin.h"
#include <stdint.h>

/* dflash_k_norm_rope_store -- the DFlash2 layer's K chain in one launch:
 *
 *     rmsnorm(k)      head_dim-wide RMS norm against the K gain, wadd added to the gain in f32,
 *                     result ROUNDED TO BF16 (that rounding is rmsnorm's store, not an accident);
 *     rope(k)         NeoX pairing i <-> i + head_dim/2, `theta`, the position divided by `scale`,
 *                     accurate sincosf, result rounded to bf16 again;
 *     kv_store        the rotated bf16 k and the v row scattered into the paged cache at the
 *                     token's slot: [n_blocks, n_head_kv, block_size, 2*head_dim], k at [.., d]
 *                     and v at [.., head_dim + d].
 *
 * The definition is the UNFUSED DEVICE CHAIN this replaces -- libr4d/r4d_model_bf16.hip's
 * `rmsnorm_kernel` with block_sum, `rope_kernel` and `kv_store_kernel`. Bit-for-bit equality with
 * that chain is the INTENDED TARGET, NOT AN ESTABLISHED RESULT: it has been reasoned from the
 * source (the fold partition, the two bf16 store points, the same transcendental calls) and it has
 * NOT been measured. No build, compile or device comparison has run against this file. Only the
 * dedicated mine-vs-libr4d comparator described below can establish it; until then treat every
 * numerics statement here as an inspection, not a verification.
 *
 * The host reference (libref/ref_elementwise.cpp's ref_rmsnorm, ref_attn.cpp's ref_rope /
 * ref_kv_store) differs from the device chain in two ways that are documented there and are NOT
 * what this kernel reproduces: ref uses 1.0f/sqrtf where the device uses rsqrtf, and ref
 * accumulates the sum of squares one element at a time where the device folds it over a wave.
 * Where they differ the device chain wins, because the device chain is what the engine ran -- and
 * a comparison against libref is therefore expected to show last-bit differences in the norm, by
 * design.
 *
 * THE ROTATION IS COMPUTED FROM theta. There is no cos/sin plane, no new buffer and no lifetime to
 * manage; `powf` and `sincosf` are the same two calls rope_kernel makes at the same arguments.
 *
 * THE GAIN'S WIDTH IS THE OPERAND'S, f32 in this model (rad_block_dflash2.h declares every norm
 * gain RAD_F32) and bf16 accepted because vLLM's is -- exactly the pair r4d_rad_rmsnorm accepts.
 *
 * THE SUPPORTED GEOMETRY IS NARROW ON PURPOSE: head_dim 128, bf16 k / v / cache, i32 positions and
 * slots, NeoX with the full head rotated, kv_store's own paged layout and its slot rule (a negative
 * slot stores nothing, a slot past the last block stores nothing). Everything else is refused at
 * issue rather than approximated. Partial rotary has no parameter here because this block never
 * declares one; adding it would need the tail write and a second rounding site.
 */
enum {
    RAD_DKNS_K    = 0,   /* INOUT: normed and rotated in place, exactly as the pair does */
    RAD_DKNS_V    = 1,
    RAD_DKNS_W    = 2,   /* WEIGHT: the K-norm gain, head_dim wide */
    RAD_DKNS_POS  = 3,
    RAD_DKNS_SLOT = 4,
    RAD_DKNS_KV   = 5,
    RAD_DKNS_OPERANDS = 6
};

/* Validated, device-free view of one call. Nothing here dereferences a pointer: at issue every
 * operand is device memory. */
struct RadDflashKStoreCall {
    void*           k;          /* written in place */
    const void*     v;
    const void*     w;
    const int32_t*  pos;
    const int32_t*  slot;
    void*           kv;
    int64_t         rows, head_dim, n_head_kv, block_size, n_blocks;
    int64_t         k_pitch, v_pitch;   /* stride[0]: elements between consecutive rows */
    int             w_f32;
    float           eps, wadd, theta, rscale;
};

int rad_dflash_k_store_schema_count(void);
const RadOpSchema* rad_dflash_k_store_schema_at(int);
int rad_dflash_k_store_shape(const RadParam*, int, int, RadOpdDesc*);
/* RAD_OK, or the reason this call is not the op: RAD_E_UNSUPPORTED outside the declared geometry,
 * RAD_E_SHAPE for an extent that disagrees with the parameters, RAD_E_DTYPE for a width or a
 * `kv_dtype` that is not the declared one, RAD_E_STRIDE for a column step other than 1 or a cache
 * that is not packed, RAD_E_INVAL for a required operand absent. */
int rad_dflash_k_store_args(const RadArgs*, RadDflashKStoreCall*);
/* THE CHAIN IS A DESCRIPTION, AND STOCK rad-kbench CANNOT EXECUTE IT. It names the conventional
 * ops `rmsnorm` -> `rope` -> `kv_store` this kernel computes in one launch, each at ITS OWN
 * geometry (the norm over rows*n_head_kv rows of width head_dim; the rotation and the store over
 * rows). RAD_E_NOTFOUND past the last step.
 *
 * Executability, stated because it is easy to assume: rad-kbench resolves a `replaces` step ONLY
 * WITHIN THE FUSED ROW'S OWN PLUGIN (tools/rad_kbench.cpp:2682). Plugin `mine` has no
 * `rmsnorm`/`rope`/`kv_store` rows, so a stock `--fuse` pass reports "no mine kernel for step
 * 'rmsnorm'" and counts a SKIP. A skip is not a pass and this hook is NOT evidence of parity.
 * Establishing bit-equality needs the dedicated comparator that launches this row out of `mine`,
 * launches each step from `libr4d` on an independently initialised buffer set, and compares the
 * post-launch `k` and the entire `kv_cache` byte for byte, failing closed on any missing row. */
int rad_dflash_k_store_fuse(const RadParam*, int, uint64_t, int, RadFuseStep*);
RadKernelInfo rad_dflash_k_store_row(RadLaunchFn);
int rad_mine_dflash_k_store_count(void);
const RadKernelInfo* rad_mine_dflash_k_store_at(int);

#endif /* RAD_MINE_DFLASH_K_NORM_ROPE_STORE_H */
