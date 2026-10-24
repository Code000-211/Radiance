// The QSA indexer's COMPRESSED KEY: one 128-dim vector per four tokens of key history.
//
// Qwen3.8-Flash-Next's full-attention layers do not attend to everything. A small MQA indexer
// scores 4-token blocks of the key history and the attention reads only the best 512 of them, so
// attention cost is constant past 2051 tokens instead of linear. This kernel builds the thing
// those scores are taken against. docs/QSA.md is the whole decomposition.
//
// WHAT THE REFERENCE HIDES, AND IT IS THE REASON THIS IS A KERNEL AT ALL. `Qwen4ExpTextQSAIndexer`
// recomputes every block key inside its per-query loop, because it is written for clarity. A block
// key does not depend on the query and does not change once its four tokens exist -- so it is
// computed ONCE, when the fourth token arrives, and cached. That is the difference between the
// indexer costing 64 bytes a token a layer and costing a pass over the whole history every step.
//
// THE ARITHMETIC, in the order the reference does it and with the roundings that order implies:
//
//   pooled = mean over the block's `ratio` raw keys, ACCUMULATED IN f32 and rounded to bf16
//   normed = RMSNorm(pooled) over `n`, gain (wadd + w)
//   out    = rope(normed) at the position of the block's FIRST token
//
// THE MEAN IS ROUNDED TO bf16 BEFORE THE NORM and that is not an accident of transcription: the
// reference is `key_groups.float().mean(dim=1).to(raw_keys.dtype)` and the norm then reads a bf16
// tensor. Accumulating in f32 and carrying f32 into the norm is a DIFFERENT number -- small, and
// systematic on every block of every layer.
//
// THE ROTATION IS r4d_qk_norm_rope_gate's, line for line: `1/powf(theta, 2i/rot)`, the position
// divided by `rscale` in f32, `sincosf`, neox pairing at a stride of `rot/2`. It computes sincos
// rather than reading that kernel's table because the table holds only the positions THIS STEP
// touches, and a block completing at decode has its first token three positions back.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"   // r4d_wave_sum

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMaxN = 256;      // the index head dim; 128 on this model, bounded for the LDS

/* Which position component frequency `i` reads, for the multi-component rotaries: `mode` 2 is
 * contiguous sections, 3 Qwen3-VL's interleaved ones (r4d_model_bf16.hip's rope says the same). */
__device__ __forceinline__ int qbk_comp(int mode, int i, int s0, int s1, int s2) {
    if (mode == 3) {
        if (i % 3 == 1 && i < 3 * s1) return 1;
        if (i % 3 == 2 && i < 3 * s2) return 2;
        return 0;
    }
    return i < s0 ? 0 : (i < s0 + s1 ? 1 : 2);
}

__global__ __launch_bounds__(kMaxN) void qsa_block_key_kernel(
    const uint16_t* __restrict__ k, const float* __restrict__ w,
    const int32_t* __restrict__ page, const int32_t* __restrict__ bpos,
    uint16_t* __restrict__ out,
    int n, int ratio, int rot, float eps, float wadd, float theta, float rscale,
    long long pos0, int k_ld, int o_ld, int mode, int s0, int s1, int s2, int bstep, int mc) {

    __shared__ float sh[kMaxN];
    __shared__ float red[kMaxN / 32];

    const int b = blockIdx.x;
    if (page && page[b] < 0) return;
    const int t = threadIdx.x;

    /* The mean, in f32 over the block's rows, rounded to bf16 exactly once. */
    float acc = 0.0f;
    if (t < n) {
        const size_t base = (size_t)b * ratio;
        for (int j = 0; j < ratio; ++j)
            acc += __bfloat162float(((const __hip_bfloat16*)k)[(base + j) * k_ld + t]);
        acc = __bfloat162float(__float2bfloat16(acc / (float)ratio));
    }

    /* RMSNorm over n. The sum is over the whole row, so it is a block reduction and not a wave
     * one -- n is 128 here and a wave is 32. */
    float ss = t < n ? acc * acc : 0.0f;
    ss = r4d_wave_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    const int nw = (blockDim.x + 31) / 32;
    for (int i = 0; i < nw; ++i) sum += red[i];
    const float inv = rsqrtf(sum / (float)n + eps);

    /* Rounded to bf16 before the rotation, matching r4d_qk_norm_rope_gate for the model's own K
     * so that block key rotations match model K rotations bit-identically. */
    if (t < n) sh[t] = __bfloat162float(__float2bfloat16(acc * inv * (wadd + w[t])));
    __syncthreads();

    if (t >= n) return;
    float res = sh[t];
    if (t < rot) {
        const int half = rot >> 1;
        const bool first = t < half;
        const int pt = first ? t + half : t - half;
        const int ci = first ? t : t - half;
        /* THE BLOCK'S OWN POSITION, from the work list when there is one. `pos0 + b*ratio` is the
         * tiling and serves a prefill chunk, where the completing blocks ARE a contiguous run of
         * one sequence. It does not serve a decode step: the sequences in a batch are at
         * different positions, so the blocks completing in one launch belong to different
         * sequences and are not a run at all. An explicit list makes a launch a set of work items
         * rather than a range, and then one launch a layer serves any mix. */
        /* With `mc`, the list holds a block's three components side by side, blocks `bstep`
         * apart, and each frequency reads the one its section names. */
        const int comp = mc ? qbk_comp(mode, ci, s0, s1, s2) : 0;
        const long long bp = bpos ? (long long)bpos[(size_t)b * bstep + comp]
                                  : (pos0 + (long long)b * ratio);
        const float p = (float)bp / (rscale != 0.0f ? rscale : 1.0f);
        const float iv = 1.0f / powf(theta, (float)(2 * ci) / (float)rot);
        float s, c;
        sincosf(p * iv, &s, &c);
        const float xp = sh[pt];
        res = first ? (res * c - xp * s) : (xp * s + res * c);
    }
    /* THE DESTINATION IS A PAGE AND NOT A ROW when `page` is passed, because the block-key store
     * is the paged KV group described in docs/QSA.md: block b of a sequence lives at whatever page
     * the manager gave it, and consecutive blocks are not consecutive pages. Absent, the write is
     * contiguous -- which is the shape a test drives and the shape a caller with its own scratch
     * wants. A negative entry is a block with no page, and is skipped rather than written
     * somewhere arbitrary. */
    long long orow = b;
    if (page) {
        const int32_t p = page[b];
        if (p < 0) return;
        orow = p;
    }
    ((__hip_bfloat16*)out)[(size_t)orow * o_ld + t] = __float2bfloat16(res);
}

}  // namespace

//   k    [M*ratio][n]  bf16, `k_ld` elements a row (0 = n)
//   w    [n]           f32
//   out  [M][n]        bf16, `o_ld` elements a row (0 = n)
//
// `pos0` is the position of the FIRST token of the FIRST block. Blocks tile the token stream, so
// block i's first token is at `pos0 + i * ratio` and no position operand is needed; a caller with
// a non-contiguous run issues it in runs.
static int r4d_qsa_block_key_bf16_emit(long k, long w, long page, long bpos, long out, int M,
                                      int n, int ratio, int rot, float eps, float wadd,
                                      float theta, float rscale, long long pos0, int k_ld,
                                      int o_ld, int mode, int s0, int s1, int s2, int bstep, int mc,
                                      const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!k || !w || !out) return -1;
    if (M <= 0 || n <= 0 || n > kMaxN || ratio <= 0) return -2;
    if (rot < 0 || rot > n || (rot & 1)) return -3;
    if (pos0 < 0 && !bpos) return -5;
    if (!k_ld) k_ld = n;
    if (!o_ld) o_ld = n;
    /* The launch is n threads rounded up to a wave, because the reduction is a block reduction
     * over exactly n values and a partial wave would fold garbage into it. */
    const int thr = ((n + 31) / 32) * 32;
    emit_rc = R4D_EMIT(sk, qsa_block_key_kernel, dim3((unsigned)M), dim3((unsigned)thr), 0, (const uint16_t*)k, (const float*)w,
                       (const int32_t*)page, (const int32_t*)bpos, (uint16_t*)out,
                       n, ratio, rot, eps, wadd, theta, rscale, pos0, k_ld, o_ld, mode, s0, s1,
                       s2, bstep, mc);
    return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
}

extern "C" int r4d_qsa_block_key_bf16_emit(long k, long w, long page, long bpos, long out, int M,
                                      int n, int ratio, int rot, float eps, float wadd,
                                      float theta, float rscale, long long pos0, int k_ld,
                                      int o_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return r4d_qsa_block_key_bf16_emit(k, w, page, bpos, out, M, n, ratio, rot, eps, wadd, theta, rscale, pos0, k_ld, o_ld, 0, 0, 0, 0, 1, 0, sk);
}

// ============================================================== the plugin ABI entry point

enum { QBK_K = 0, QBK_W = 1, QBK_PAGE = 2, QBK_POS = 3, QBK_OUT = 4 };

static int r4d_rad_qsa_block_key_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* k = r4d_opt(a, QBK_K);
    const RadTensor* w = r4d_opt(a, QBK_W);
    const RadTensor* pg = r4d_opt(a, QBK_PAGE);      /* OPTIONAL: absent is a contiguous write */
    const RadTensor* bp = r4d_opt(a, QBK_POS);       /* OPTIONAL: absent is the pos0 tiling */
    const RadTensor* o = r4d_opt(a, QBK_OUT);
    if (!k || !w || !o) return R4D_NO(RAD_E_INVAL);
    /* THE DESTINATION MAY BE A KV CACHE, WHICH IS RANK FOUR. The block-key store is a KV group
     * whose geometry is a container and not an attention shape -- `kv_heads` 1, `block_size` the
     * compress ratio, `head_dim` n/(2*ratio) -- so a page holds exactly `n` contiguous values and
     * the manager allocates one per `ratio` tokens (docs/QSA.md). Read that way it IS [pages, n],
     * and the check is on the PRODUCT rather than on the rank, because the rank is the engine's
     * business and the width is this op's. */
    if (k->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (o->rank < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long rot   = r4d_geti_or(a, "rotary_dim", 0);
    const long long pos0  = r4d_geti_or(a, "pos0", -1);
    if (n <= 0 || ratio <= 0 || (pos0 < 0 && !bp)) return R4D_NO(RAD_E_INVAL);
    /* `block_size` exists to size the block-key KV group and is the compress block by definition.
     * A caller that set them apart has a group whose page holds a different number of tokens than
     * a block does, which is the one way this whole design stops working. */
    const long long bs = r4d_geti_or(a, "block_size", ratio);
    if (bs != ratio) return R4D_NO(RAD_E_SHAPE);
    /* THE BLOCK COUNT IS THE KEY OPERAND'S, NOT THE OUTPUT'S, once a page list is in play: the
     * store is as tall as the POOL and the launch is as tall as the blocks completing this step.
     * Reading M off `out` would write the whole store on every call. */
    const long long M = pg ? r4d_numel(pg) : o->shape[0];
    const long long ow = o->shape[0] > 0 ? r4d_numel(o) / o->shape[0] : 0;
    if (M < 1 || ow != n) return R4D_NO(RAD_E_SHAPE);
    if (pg && o->shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    /* `pos` is one position a block, or a rank-2 [>= M, 1 or 3] column slice with a block's
     * components side by side and the blocks a row pitch apart (qsa_work's output). */
    int bstep = 1;
    if (bp && bp->rank == 2) {
        if (bp->shape[0] < M || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1])
            return R4D_NO(RAD_E_SHAPE);
        bstep = (int)bp->stride[0];
    } else if (bp && r4d_numel(bp) < M) {
        return R4D_NO(RAD_E_SHAPE);
    }
    /* THE MULTI-COMPONENT ROTARY: `mode` mrope or imrope with `sections`, and three columns of
     * `pos` (qsa_work's output with its `rope` operand). A single column is the one-component
     * form, which every mode reads as NeoX. */
    const char* mode_s = r4d_gets(a, "mode");
    int mode = 0, sec[3] = { 0, 0, 0 }, n_sec = 0, mc = 0;
    if (mode_s && !std::strcmp(mode_s, "mrope")) mode = 2;
    else if (mode_s && !std::strcmp(mode_s, "imrope")) mode = 3;
    else if (mode_s && std::strcmp(mode_s, "neox")) return R4D_NO(RAD_E_SCHEMA);
    for (const char* v = r4d_gets(a, "sections"); v && *v;) {
        char* e = nullptr;
        const long x = std::strtol(v, &e, 10);
        if (e == v) break;
        if (x <= 0 || n_sec >= 3) return R4D_NO(RAD_E_SHAPE);
        sec[n_sec++] = (int)x;
        v = e;
    }
    if (mode && n_sec > 0 && bp && bp->rank == 2 && bp->shape[1] >= 3) mc = 1;
    /* THE KEY OPERAND IS `ratio` TIMES TALLER, and it is checked rather than assumed: an operand
     * narrowed to the block count instead of the token count would make every block pool three
     * rows of the next one, which produces plausible numbers and a different model. */
    if (k->shape[0] != M * ratio || k->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(w) < n) return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k), op = r4d_row_pitch(o);
    if (kp < 0 || op < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = r4d_qsa_block_key_bf16_emit(
        r4d_addr(k), r4d_addr(w), r4d_addr(pg), r4d_addr(bp), r4d_addr(o),
        (int)M, (int)n, (int)ratio, (int)rot,
        (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 1.0),
        (float)r4d_getf(a, "theta", 10000.0), (float)r4d_getf(a, "scale", 1.0),
        pos0, (int)kp, (int)op, mode, sec[0], sec[1], sec[2], bstep, mc, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

R4D_ENTRY(r4d_rad_qsa_block_key, r4d_rad_qsa_block_key_body)
