// r4d_qk_norm_rope_gate.hip -- r4d_qk_norm_rope_gate: the whole gated-attention QKV preamble in one
// pass: split [q|gate] per head, QK RMS norm over the head, partial NeoX RoPE, and the gate copy.
//
// WHY.  vLLM already has this fusion -- `fused_qk_rmsnorm_rope_gate` -- and gates it on
// `current_platform.is_cuda()`.  The kernel behind it is plain Triton with nothing CUDA-specific in
// it, so the gate looks like conservatism, but it is not: on gfx1201 that kernel does not compile.
// Triton's AMD backend fails in `TritonAMDGPUCanonicalizePointers` on its pointer arithmetic.  So
// ROCm runs the unfused branch -- a split, two GemmaRMSNorms, a RoPE and a gate chunk, which
// inductor lands as FOUR kernels per attention layer, 16 layers, 64 dispatches a decode step for
// ~6.3 us of work that is four passes over the same 14 KB.
//
// WHAT IT IS BIT-FOR-BIT.  The reference path stores the normed value to memory as bf16 and then
// RoPEs what it reads back, so the norm output is ROUNDED TO bf16 BEFORE the rotation.  Skipping
// that round trip is more accurate and would be wrong: it is a different function.  The `.to(
// INPUT_DTYPE).to(tl.float32)` in vLLM's own kernel is there for the same reason.
//
// `wadd` IS ADDED TO EVERY GAIN.  Qwen3NextRMSNorm is GemmaRMSNorm: `x_hat * (1 + w)` over a
// zero-centred gamma, formed in fp32.  vLLM's fused kernel takes `weight.float() + 1.0` from the
// caller; this takes the bf16 weight and the offset, and adds it after the load, inside the fp32
// arithmetic -- identical, and it does not need a per-layer fp32 copy of the gain.
//
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

// ONE BLOCK PER (TOKEN, HEAD) AND ONE ELEMENT PER LANE, WHICH IS NOT A LAYOUT CHOICE -- IT IS THE
// BIT-EQUALITY REQUIREMENT.
//
// The tempting layout gives each lane EIGHT CONSECUTIVE elements and reduces one wave per head: no
// LDS, no barrier, and the rotary partner one `__shfl_xor` away, because element i and i + rot/2
// sit in lanes (rot/2)/VPL apart. It is elegant and it is a DIFFERENT FUNCTION.
// `r4d_rmsnorm_bf16` states the rule that kills it: "the width decides which columns a thread
// folds into the sum of squares. Two kernels that disagree about it disagree in the last bits of
// the norm." A ULP in the mean square is a ULP in the normed value, a ROTATED element is a
// difference of two products, and where those nearly cancel the ULP comes out as a few percent --
// enough to move the engine's text within a few dozen steps, which forces the fusion off.
//
// The reference's fold is reproducible exactly, and it is simpler than it looks. `rmsnorm_bf16`
// runs `for (i = threadIdx.x; i < n; i += THR)`, and `r4d_act_threads` never returns fewer than 256
// threads while every head_dim compiled here is at most 256 -- so the reference ALWAYS folds
// exactly one element per thread, whatever width it picks. The widths above the head dim contribute
// zeros, and adding 0.0f to an f32 accumulator is exact, so `block_sum` over 8 wave partials and
// over 32 of them are the same number. One element a lane, 256 threads, `block_sum` in ascending
// wave order: that is the whole of it.
//
// WHAT IT COSTS: the rotary partner is `rot/2` lanes away, which at rot 64 is a different WAVE, so
// no shuffle reaches it. The partner is RE-DERIVED rather than exchanged through LDS and a
// barrier -- the lane loads the partner's input element and norms it itself, which is one L1-hot
// read and the same `inv` the block already agreed on. No shared memory beyond block_sum's eight
// floats, and no barrier the reduction did not already need.
/* THE PAGED CACHE the KV form writes: kv_store's operands and parameters, in one argument so the
 * launch stays inside the emitter's parameter budget. The descales are read by the E4M3 form only. */
struct QkKvStore {
  const __hip_bfloat16* v;
  const int32_t*        slot;
  void*                 cache;
  int   v_pitch, block_size, n_blocks;
  float k_inv, v_inv;
};

/* The cache forms KV selects. */
enum { QK_KV_NONE = 0, QK_KV_FP8 = 1, QK_KV_BF16 = 2 };

/* KV folds kv_store in: a k head's block, holding its token's rotated k one element a lane, also
 * writes that element and the head's v element into the cache at the token's slot -- as E4M3 with
 * kv_store_fp8's arithmetic element for element, or as the bf16 kv_store_bf16 copies (see the store
 * at the end). */
template <int HD, int KV = QK_KV_NONE>
__global__ __launch_bounds__(256) void r4d_qk_norm_rope_gate_kernel(
    const __hip_bfloat16* __restrict__ QG, const __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ QO, __hip_bfloat16* __restrict__ KO,
    __hip_bfloat16* __restrict__ GO,
    const void* __restrict__ WQ, const void* __restrict__ WK,
    const void* __restrict__ CS, const void* __restrict__ POS,
    int nq, int nkv, int rot, float eps, float wadd,
    int qg_pitch, int k_pitch, int cs_f32, int pos_i64, int w_f32, QkKvStore kvs) {
  constexpr int THR = 256;
  __shared__ float lds[THR / 32];
  const int tok = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x;
  const long p = pos_i64 ? ((const long*)POS)[tok] : (long)((const int*)POS)[tok];
  const int half = rot >> 1;
  const bool isk = h >= nq;
  const int lh = isk ? h - nq : h;
  const __hip_bfloat16* in = isk ? (K + (size_t)tok * k_pitch + (size_t)lh * HD)
                                 : (QG + (size_t)tok * qg_pitch + (size_t)lh * 2 * HD);
  // THE GAIN IS f32 IN THIS MODEL AND bf16 IN vLLM's, and both are the dtype the unfused rmsnorm
  // beside it would have read -- so the width is a property of the operand, not of the kernel.
  const void* w = isk ? WK : WQ;
  __hip_bfloat16* out = isk ? (KO + ((size_t)tok * nkv + lh) * HD)
                            : (QO + ((size_t)tok * nq + lh) * HD);
  const float* csf = (const float*)CS;
  const __hip_bfloat16* csb = (const __hip_bfloat16*)CS;

  // `gain` is spelled once so the two sites that need it -- this lane's element and, in the rotary
  // range, its partner's -- cannot pick different widths.
  auto gain = [&](int i) -> float {
    return w_f32 ? ((const float*)w)[i] : __bfloat162float(((const __hip_bfloat16*)w)[i]);
  };
  auto normed = [&](int i, float inv) -> float {
    const float v = __bfloat162float(in[i]);
    // ROUNDED TO bf16 BEFORE THE ROTATION, which is the whole bit-identity argument at the top of
    // this file: the reference stores the norm and RoPEs what it reads back.
    return __bfloat162float(__float2bfloat16(v * inv * (gain(i) + wadd)));
  };

  float ss = 0.0f;
  if (t < HD) { const float v = __bfloat162float(in[t]); ss = v * v; }
  // rmsnorm_bf16's block_sum, transcribed: the wave butterfly, then the wave partials summed in
  // ASCENDING index order. Both halves matter -- an accumulation over the same values in a
  // different order breaks the bit-equality argument above.
  ss = r4d_wave_sum(ss);
  if ((t & 31) == 0) lds[t >> 5] = ss;
  __syncthreads();
  float sum = 0.0f;
#pragma unroll
  for (int wv = 0; wv < THR / 32; ++wv) sum += lds[wv];
  const float inv = rsqrtf(sum / (float)HD + eps);

  float res = 0.0f;
  if (t < HD) {
    res = normed(t, inv);
    if (t < rot) {
      const bool first = t < half;
      const int pt = first ? t + half : t - half;
      const int ci = first ? t : t - half;
      const size_t co = (size_t)p * rot;
      const float c = cs_f32 ? csf[co + ci] : __bfloat162float(csb[co + ci]);
      const float s = cs_f32 ? csf[co + half + ci] : __bfloat162float(csb[co + half + ci]);
      const float xp = normed(pt, inv);
      res = first ? (res * c - xp * s) : (xp * s + res * c);
    }
  }
  // EVERY LANE HAS READ ITS INPUT BEFORE ANY LANE WRITES IT BACK. `k` is rotated IN PLACE by the
  // engine -- one buffer for the operand and the result -- and under the one-element-a-lane layout
  // a lane's partner element is a MEMORY READ rather than a shuffle, so without this barrier the
  // partner may already have been overwritten. A shuffle-based layout does not need the barrier,
  // which is exactly why it is easy to leave out here: the failure is a 1% scatter of elements
  // that reads like a rounding difference and is not one. r4d_selftest's byte-equality row catches
  // it.
  // AG03: Only K-heads (isk) rotate in-place (in == out). For Q-heads (!isk), in is QG
  // and out is QO, so they write to a completely disjunct buffer with zero aliasing risk.
  if (isk) __syncthreads();
  if (t < HD) {
    out[t] = __float2bfloat16(res);
    // `gate_out` is OPTIONAL in the schema, and a caller that already reads the gate straight out
    // of the interleaved projection -- which the fp8 block does, through gate_quant_fp8 -- passes
    // none rather than paying for a copy it will not read.
    if (!isk && GO) GO[((size_t)tok * nq + lh) * HD + t] = in[HD + t];
  }
  // THE CACHE WRITE kv_store WOULD MAKE NEXT, from the same bytes it would read: the rotated k as
  // the bf16 just stored and the v row as the projection left it. The E4M3 form scales each by the
  // inverse descale, clamps (v_cvt_pk_fp8_f32 answers out-of-range with NaN) and converts; the pair
  // conversion there rounds each element on its own, so one element a lane is the same byte. The
  // bf16 form stores the two values as they are, which is all kv_store_bf16 does. The layout is
  // [n_blocks, kv_heads, block_size, 2*head_dim], k then v, in the cache's element; a negative or
  // out-of-range slot stores nothing.
  if constexpr (KV != QK_KV_NONE) {
    if (isk && t < HD) {
      const int32_t sl = kvs.slot[tok];
      if (sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < (uint32_t)kvs.n_blocks) {
        const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
        const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
        const size_t at = ((size_t)blk * nkv * kvs.block_size + off) * 2 * HD +
                          (size_t)lh * kvs.block_size * 2 * HD;
        const __hip_bfloat16 kb = __float2bfloat16(res);
        const __hip_bfloat16 vb = kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t];
        if constexpr (KV == QK_KV_BF16) {
          __hip_bfloat16* c = (__hip_bfloat16*)kvs.cache + at;
          c[t]      = kb;
          c[HD + t] = vb;
        } else {
          unsigned char* c = (unsigned char*)kvs.cache + at;
          const float kf = __bfloat162float(kb) * kvs.k_inv;
          const float vf = __bfloat162float(vb) * kvs.v_inv;
          const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf, -448.0f), 448.0f), 0.0f,
                                                         0, false);
          const int pv = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vf, -448.0f), 448.0f), 0.0f,
                                                         0, false);
          c[t]      = (unsigned char)(pk & 0xFF);
          c[HD + t] = (unsigned char)(pv & 0xFF);
        }
      }
    }
  }
}


// ============================================================ the cos/sin table
//
// The fused kernel above reads a PRECOMPUTED table instead of theta, and this fills it.  One launch
// a step serves every attention layer, against one rope launch a layer -- which is the whole point:
// at head_dim 256 the rotation is ~7 us of dispatch for ~1 us of work, sixteen times over.
//
// IT IS WRITTEN AT THE POSITION'S OWN ROW, so the table has a row per position the model can name
// and only the rows a step actually touches are filled.  A decode step touches at most a few dozen
// of them, so this moves kilobytes and not the six megabytes a whole table would.  Rows nothing
// addresses are never read.
//
// THE ARITHMETIC IS r4d_model_bf16.hip's rope_kernel, LINE FOR LINE, and that is what makes the
// fusion bit-for-bit rather than merely close: `1.0f / powf(theta, 2i/rot)`, the position divided
// by `rscale` in f32, `sincosf`, and the products consumed in f32 by the caller.  Changing any of
// those here silently changes the model's output -- a different rope is a different model.
__global__ __launch_bounds__(64) void r4d_qk_rope_table_kernel(
    const void* __restrict__ POS, void* __restrict__ CS, long rows, int rot, float theta,
    float rscale, int cs_f32, int pos_i64) {
  const int m = blockIdx.x;
  const long pp = pos_i64 ? ((const long*)POS)[m] : (long)((const int*)POS)[m];
  // A padded row addresses no table row, and a position at or past the table's height writes
  // nothing: its row would land in whatever buffer the arena placed after the plane. libref and
  // libavx skip the same rows.
  if (pp < 0 || pp >= rows) return;
  const int half = rot >> 1;
  const float p = (float)pp / (rscale != 0.0f ? rscale : 1.0f);
  const size_t o = (size_t)pp * rot;
  for (int i = threadIdx.x; i < half; i += (int)blockDim.x) {
    const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rot);
    float c, s;
    sincosf(p * inv, &s, &c);
    if (cs_f32) {
      ((float*)CS)[o + i] = c;
      ((float*)CS)[o + half + i] = s;
    } else {
      ((__hip_bfloat16*)CS)[o + i] = __float2bfloat16(c);
      ((__hip_bfloat16*)CS)[o + half + i] = __float2bfloat16(s);
    }
  }
}

// The two launches, as the plugin ABI's RadLaunchFn. See the note at the head of
// r4d_quant_act_i8.hip for why the flattened argument list and the throws are gone.
enum { QK_QGATE = 0, QK_K, QK_COSSIN, QK_POS, QK_WQ, QK_WK, QK_QOUT, QK_KOUT, QK_GOUT,
       QK_V, QK_SLOT, QK_CACHE };

/* The body of both forms; KVF is the kv_store fold and `kvs` its cache, already checked. */
template <int KVF>
static int qk_body(const RadArgs* args, const R4dSink& sk, const QkKvStore& kvs) {
  const RadTensor* qg = r4d_opt(args, QK_QGATE);
  const RadTensor* k  = r4d_opt(args, QK_K);
  const RadTensor* cs = r4d_opt(args, QK_COSSIN);
  const RadTensor* po = r4d_opt(args, QK_POS);
  const RadTensor* wq = r4d_opt(args, QK_WQ);
  const RadTensor* wk = r4d_opt(args, QK_WK);
  const RadTensor* qo = r4d_opt(args, QK_QOUT);
  const RadTensor* ko = r4d_opt(args, QK_KOUT);
  const RadTensor* go = r4d_opt(args, QK_GOUT);
  // THE K-ONLY FORM. `n_head` 0 means the caller has no [q|gate] projection to split -- its heads
  // are already one plain `head_dim` run each -- so `q_gate`, `q_w` and `q_out` are absent and
  // every head is served off `k`. The KERNEL NEEDS NOTHING for this: `isk` is `h >= nq`, so at
  // nq 0 every block takes the k path and the three absent operands are never dereferenced
  // (`r4d_p` and `r4d_wp` already return null for an absent tensor, so the launch needs no case
  // of its own).
  if (!k || !cs || !po || !wk || !ko) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(cs) || !r4d_contig(po) || !r4d_contig(wk)) return R4D_NO(RAD_E_STRIDE);
  if (!r4d_contig(ko)) return R4D_NO(RAD_E_STRIDE);
  if (go && !r4d_contig(go)) return R4D_NO(RAD_E_STRIDE);

  const long long hd  = r4d_geti_or(args, "head_dim", 0);
  const long long nq  = r4d_geti_or(args, "n_head", 0);
  const long long nkv = r4d_geti_or(args, "n_head_kv", 0);
  const long long rot = r4d_geti_or(args, "rot", 0);
  if (nq < 0 || nkv < 1) return R4D_NO(RAD_E_SHAPE);
  if (nq >= 1 && (!qg || !wq || !qo)) return R4D_NO(RAD_E_INVAL);
  if (qg && (!r4d_contig(wq) || !r4d_contig(qo))) return R4D_NO(RAD_E_STRIDE);

  const long long qg_pitch = qg ? r4d_row_pitch(qg) : 0;
  const long long k_pitch  = r4d_row_pitch(k);
  if (qg_pitch < 0 || k_pitch < 0) return R4D_NO(RAD_E_STRIDE);

  // The row count is the OPERAND'S, and which operand carries it is what the form decides.
  const long long M = qg ? qg->shape[0] : k->shape[0];
  if (M < 1) return R4D_NO(RAD_E_SHAPE);
  if (hd != 64 && hd != 128 && hd != 256) return R4D_NO(RAD_E_SHAPE);   // the compiled set
  if (rot < 2 || (rot & 1) || rot > hd) return R4D_NO(RAD_E_SHAPE);

  const float eps  = (float)r4d_getf(args, "eps", 1e-6);
  const float wadd = (float)r4d_getf(args, "wadd", 0.0);
  // The two width flags are read off the OPERANDS rather than declared twice: a cos/sin table
  // built in f32 and a positions vector in int64 are what the tensor SAYS they are.
  const int cs_f32  = (cs->dtype == RAD_F32) ? 1 : 0;
  const int pos_i64 = (po->dtype == RAD_I64) ? 1 : 0;
  // THE NORM GAIN'S WIDTH IS THE OPERAND'S TOO, and it is f32 in every architecture plugin here
  // where vLLM's is bf16. The two gains must agree, because one flag serves both.
  if (wq && wq->dtype != wk->dtype) return R4D_NO(RAD_E_DTYPE);
  if (wk->dtype != RAD_F32 && wk->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
  const int w_f32 = (wk->dtype == RAD_F32) ? 1 : 0;

  const int nh = (int)(nq + nkv);
  // One block a (token, head), 256 threads, one element a lane. The width is FIXED at 256 and not
  // fitted to the head count: it is what makes the sum of squares agree with r4d_rmsnorm_bf16, and
  // the kernel says at length why.
#define R4D_QKG_L(H) emit_rc = R4D_EMIT(sk, (r4d_qk_norm_rope_gate_kernel<H, KVF>), \
    dim3((int)M, nh), dim3(256), 0, (const __hip_bfloat16*)r4d_p(qg), \
    (const __hip_bfloat16*)r4d_p(k), (__hip_bfloat16*)r4d_wp(qo), (__hip_bfloat16*)r4d_wp(ko), \
    (__hip_bfloat16*)r4d_wp(go), r4d_p(wq), r4d_p(wk), r4d_p(cs), r4d_p(po), \
    (int)nq, (int)nkv, (int)rot, eps, wadd, (int)qg_pitch, (int)k_pitch, cs_f32, pos_i64, w_f32, \
    kvs)
  int emit_rc = RAD_OK;
  switch (hd) {
    case 64:  R4D_QKG_L(64); break;
    case 128: R4D_QKG_L(128); break;
    default:  R4D_QKG_L(256); break;
  }
#undef R4D_QKG_L
  return emit_rc;
}

static int rad_qk_norm_rope_gate_body(const RadArgs* args, const R4dSink& sk) {
  return qk_body<QK_KV_NONE>(args, sk, QkKvStore{});
}
R4D_ENTRY(r4d_rad_qk_norm_rope_gate, rad_qk_norm_rope_gate_body)

/* THE PROLOGUE WITH THE KV STORE FOLDED IN: qk_norm_rope_gate's operands, then kv_store's `v`,
 * `slot_mapping` and `kv_cache`, and kv_store's parameters. The cache is E4M3 or bf16, whichever
 * `kv_dtype` names and the tensor is, checked the way kv_store checks it, and one slot a row of
 * the prologue. The k kv_store would read is the prologue's own output, so it is not an operand of
 * its own. */
static int rad_qk_norm_rope_gate_kv_store_body(const RadArgs* args, const R4dSink& sk) {
  const RadTensor* k  = r4d_opt(args, QK_K);
  const RadTensor* qg = r4d_opt(args, QK_QGATE);
  const RadTensor* v  = r4d_opt(args, QK_V);
  const RadTensor* sm = r4d_opt(args, QK_SLOT);
  const RadTensor* kv = r4d_opt(args, QK_CACHE);
  if (!k || !v || !sm || !kv) return R4D_NO(RAD_E_INVAL);
  const char* kvd = r4d_gets(args, "kv_dtype");
  const uint32_t want = kvd ? rad_dtype_parse(kvd) : (uint32_t)RAD_DT_INVALID;
  if (v->dtype != RAD_BF16 || sm->dtype != RAD_I32 || kv->dtype != want ||
      (kv->dtype != RAD_F8E4M3 && kv->dtype != RAD_BF16))
    return R4D_NO(RAD_E_DTYPE);
  if (!r4d_contig(sm) || !r4d_contig(kv)) return R4D_NO(RAD_E_STRIDE);

  const long long hd  = r4d_geti_or(args, "head_dim", 0);
  const long long nkv = r4d_geti_or(args, "n_head_kv", 0);
  const long long bs  = r4d_geti_or(args, "block_size", 0);
  if (hd <= 0 || nkv <= 0 || bs <= 0) return R4D_NO(RAD_E_SHAPE);
  if (kv->rank != 4 || kv->shape[1] != nkv || kv->shape[2] != bs || kv->shape[3] != 2 * hd)
    return R4D_NO(RAD_E_SHAPE);
  const long long M = qg ? qg->shape[0] : k->shape[0];
  long long Mv = 0;
  const long long v_ld = r4d_rows_pitch(v, nkv * hd, &Mv);
  if (v_ld < 0) return R4D_NO(RAD_E_STRIDE);
  if (Mv < M || r4d_numel(sm) != M) return R4D_NO(RAD_E_SHAPE);

  const double ks = r4d_getf(args, "k_scale", 1.0);
  const double vs = r4d_getf(args, "v_scale", 1.0);
  if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);

  QkKvStore kvs{};
  kvs.v          = (const __hip_bfloat16*)r4d_p(v);
  kvs.slot       = (const int32_t*)r4d_p(sm);
  kvs.cache      = r4d_wp(kv);
  kvs.v_pitch    = (int)v_ld;
  kvs.block_size = (int)bs;
  kvs.n_blocks   = (int)kv->shape[0];
  kvs.k_inv      = 1.0f / (float)ks;
  kvs.v_inv      = 1.0f / (float)vs;
  return kv->dtype == RAD_BF16 ? qk_body<QK_KV_BF16>(args, sk, kvs)
                               : qk_body<QK_KV_FP8>(args, sk, kvs);
}
R4D_ENTRY(r4d_rad_qk_norm_rope_gate_kv_store, rad_qk_norm_rope_gate_kv_store_body)

// The cos/sin plane the fused prologue above reads. One launch a step against `rope`'s one a
// layer, and the reason the fusion is reachable at all: `qk_norm_rope_gate` takes a table and not
// a theta, and until this row existed nothing in the tree could build one.
//
// ONLY THE ROWS `positions` NAMES ARE WRITTEN, at each position's own row of a [rows, rot] plane
// whose height is the operand's. A decode step names a few dozen, so this moves kilobytes where a
// whole-table fill would move six megabytes; a row nothing addresses keeps whatever it held, which
// is correct because nothing reads it.
static int rad_rope_table_body(const RadArgs* args, const R4dSink& sk) {
  const RadTensor* po = r4d_opt(args, 0);
  const RadTensor* cs = r4d_opt(args, 1);
  if (!po || !cs) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(po) || !r4d_contig(cs)) return R4D_NO(RAD_E_STRIDE);
  if (cs->dtype != RAD_F32 && cs->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
  if (po->dtype != RAD_I32 && po->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);

  // The row count is the OPERAND'S: `M` is declared as a band ceiling and this is called with a
  // step's rows, not the band's.
  const long long M   = r4d_numel(po);
  const long long rot = r4d_geti_or(args, "rot", 0);
  if (M < 1 || rot < 2 || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(cs) % rot) return R4D_NO(RAD_E_SHAPE);

  const float theta  = (float)r4d_getf(args, "theta", 10000.0);
  const float rscale = (float)r4d_getf(args, "scale", 1.0);
  return R4D_EMIT(sk, r4d_qk_rope_table_kernel, dim3((int)M), dim3(64), 0,
                  r4d_p(po), r4d_wp(cs), (long)(r4d_numel(cs) / rot), (int)rot, theta, rscale,
                  cs->dtype == RAD_F32 ? 1 : 0, po->dtype == RAD_I64 ? 1 : 0);
}
R4D_ENTRY(r4d_rad_rope_table, rad_rope_table_body)
