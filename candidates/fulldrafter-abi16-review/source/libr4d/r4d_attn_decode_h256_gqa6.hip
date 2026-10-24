// R4D decode. The prefill core (lazy reference-max softmax, f16 operands, folded query scale,
// contiguous-k fragments, prefetch + scheduling barrier, dot2 row sum) on the split-KV MTP-verify
// shape, plus two decode-only changes:
//
//   KLDS0  reads K straight from the paged cache, including when that cache is fp8. The
//          widening moves from once-per-workgroup to once-per-warp; it costs VALU on a leg that is
//          bandwidth-bound and buys back the whole sK allocation, which at TILE 16 is 8.3 KB of
//          the 14.5 KB that sets resident-CTA count. This is the argmin at every depth.
//   PF16   split-KV partials as f16 instead of f32. The partial buffer is written once and read
//          back once, and at 16K context that round trip is ~14 MB against ~17 MB of KV -- i.e.
//          nearly half the leg's DRAM traffic. Storing O/l rather than O bounds the value by
//          max|v|, so f16 cannot overflow and its 11-bit mantissa costs ~5e-4 relative.
//
// This file compiles neither an fp8-P leg nor an alternative to the CONTIG fragment layout; see
// r4d_attn_prefill_h256_gqa6.hip for the full option set and the reasoning behind each bit.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"
#include "r4d_q8.h"     // the gate quantiser's 8-bit codes: E4M3, or int8 (GQ == 2)

#define O_F16    1
#define O_MSKIP  2
#define O_FOLDQ  8
#define O_KLDS0 16
#define O_PF16  32
#define O_CONTIG 512
#define O_PF2   1024
#define O_PF4   2048
#define O_PF8   3072
#define O_SGB   4096
#define O_DOT2 16384
// The tile-staging set, shared with r4d_attn_prefill_h256_gqa6.hip -- see the comments there.
//   BTS   block table entries for the tile in SGPRs, loaded a tile ahead. On this leg it also
//         removes a dependent global load from INSIDE the m-tile loop: with KLDS0 the K address
//         comes from bt[kk/BS] on every m-tile.
//   GPREV tile ti+1's V fetched into registers before the m-tile loop.
//   LDSB  an LDS-scoped barrier, so __syncthreads() stops invalidating the vector cache.
#define O_BTS   262144
#define O_GPREV 1048576
#define O_LDSB  2097152
// O_KNT: a non-temporal hint on the K global load.  The KV a decode step reads is 134 MB a layer
// streamed once and never re-read, and it walks the whole last level on the way past -- so what the
// hint would protect is not this kernel but the GEMMs either side of it, whose activation is small
// enough to live in cache if something stops evicting it.  V cannot take the hint at all: it is
// loaded through `global_load_tr_b128` and that intrinsic has no non-temporal form.
//
// OFF BY DEFAULT.  The hint changes no bytes and costs nothing, and it also gains nothing: on this
// part the KV stream is not what evicts the neighbouring GEMMs' activation, so neither this kernel
// nor they move when it is set.  The bit costs one template parameter and is kept so the option
// stays available for a deployment where the cache pressure is different.
#define O_KNT   4194304
// O_V8: an fp8 cache's V tile staged in LDS as its codes, converted when a fragment is read.
//
// The f16 tile is 12 KiB a workgroup and a one-wave workgroup has nothing else in LDS, so it is
// what caps residency: ten workgroups a WGP, two and a half waves a SIMD, for a leg whose tile
// loop is a chain of global latencies. The codes are 6 KiB. At one wave a workgroup the number of
// conversions is the same either way -- each V element is read once by the one wave -- and the
// conversion is exact, so the fragments, and every byte after them, match the f16 tile's. With more
// waves a workgroup each would convert the whole tile, so the launcher sets it at one.
#define O_V8    8388608
// `__builtin_nontemporal_load` will not take `uint4`/`uint2` -- they are HIP_vector_type class
// templates, not native vectors -- so the hinted loads go through these and bit-cast the WHOLE
// value back (never a subscript of it, which miscompiles).
typedef unsigned r4d_ntu4 __attribute__((ext_vector_type(4)));
typedef unsigned r4d_ntu2 __attribute__((ext_vector_type(2)));

template <int NT, typename T, typename V>
__device__ __forceinline__ T r4d_kload(const T* p) {
  if (!NT) return *p;
  return __builtin_bit_cast(T, __builtin_nontemporal_load((const V*)p));
}

__host__ __device__ __forceinline__ int r4d_attn_tiles(int ctx, int tile) { return (ctx + tile - 1) / tile; }
__host__ __device__ __forceinline__ int r4d_attn_tps(int ctx, int tile, int splits) {
    int t = r4d_attn_tiles(ctx, tile); return (t + splits - 1) / splits;
}
// First KV position query row `qrow` may attend to under a sliding window: the row's
// causal limit, less the window, floored at zero. window <= 0 means no window.
__host__ __device__ __forceinline__ int r4d_attn_wlo(int ctx, int q_len, int qrow, int window) {
    if (window <= 0) return 0;
    const int lo = ctx - q_len + qrow - window + 1;
    return lo > 0 ? lo : 0;
}
__host__ __device__ __forceinline__ int r4d_attn_used(int ctx, int tile, int splits) {
    int t = r4d_attn_tiles(ctx, tile), tps = r4d_attn_tps(ctx, tile, splits); return (t + tps - 1) / tps;
}

__host__ __device__ __forceinline__ bool r4d_attn_is_visible(
    int k, int prefix_len, int ctx, int klimit, uint16_t mask, int wlo = 0, bool win = false, bool cz = true)
{
    if (win && (k < wlo || (!cz && k > klimit))) return false;
    if (k < prefix_len) return true;
    if (k >= ctx) return false;
    const int tree_k = k - prefix_len;
    return (tree_k < 16) && ((mask & (1u << tree_k)) != 0);
}

// WIN turns on a sliding window (a.window positions, ending at each row's causal
// limit). It is a TEMPLATE parameter and defaults to 0, so a full-attention geometry
// compiles with no bound, no clamp and no extra mask term.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_decode_kernel(const R4DArgs a, int splits)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;
    constexpr int PF16  = (OPT & O_PF16)  ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)  ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB) ? 1 : 0;
    constexpr int KNT   = (OPT & O_KNT)  ? 1 : 0;
    constexpr int V8    = ((OPT & O_V8) && !KVP) ? 1 : 0;
    static_assert(!V8 || (OPT & O_CONTIG), "the code tile is read in the contiguous layout");
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int NKS   = HEAD_DIM / 16;
    constexpr int MT    = TILE / 16;
    constexpr int KSTR  = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR  = TILE + (CTG ? 8 : 4);
    constexpr int NTHR  = NWARPS * 32;
    constexpr int NB    = TILE / BS;
    constexpr int NTR   = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS = (NTR + NWARPS - 1) / NWARPS;
    // A V prefetch costs VREGS x 4 VGPRs and this leg already carries a 128-VGPR accumulator and a
    // 64-VGPR query. At two uint4 (fp8 KV, TILE 16) it is worth 6-8% at depth; at four (fp8
    // TILE 32, or bf16 TILE 16, whose V is twice the bytes) it spills and LOSES 10%. So the
    // geometry decides, not the flag.
    // THREE is reachable only because the K prefetch needs just two: the three-uint4 form is what
    // the 48-row shape (q_len 8, gqa 6, so NWARPS 3 over NTR 8) needs, it stays inside 255 VGPRs
    // with no spill, and it is worth a further 2-15% on top -- most at the depths where the K
    // stream has the least other work to hide behind. At NWARPS 1 or 2, and for a bf16 cache whose
    // V is twice the bytes, VREGS is 4 or more and the prefetch stays off.
#ifndef R4D_GPREV_VREGS
#define R4D_GPREV_VREGS 3
#endif
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= R4D_GPREV_VREGS) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;

    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];
    __shared__ uint16_t sV16[V8 ? 1 : HEAD_DIM * VSTR];
    /* A row of the code tile is 16 keys and 8 of padding: 24 bytes is six dwords, so the sixteen
     * lanes of a fragment read land on distinct banks, and a lane's eight codes are one aligned
     * 8-byte load. */
    constexpr int VSTR8 = TILE + 8;
    __shared__ uint8_t sV8[V8 ? HEAD_DIM * VSTR8 : 1];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;

    const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches (r4d.h, R4DArgs::cu_q): the rows of q and out this sequence owns, and its
    // own query length. `a.q_len` is the batch maximum when cu_q is set, and it is what the
    // launcher sized the wave count from, so rows past `qlen` are simply not live.
    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    if (qlen <= 0) return;                 // a slot carrying no query rows this step

    const int ntl  = r4d_attn_tiles(ctx, TILE);
    // A sliding-window layer reads only the last `window` positions. The blocks below
    // that are not merely useless: the engine has freed them and may already have handed
    // them to another sequence, so reading one is wrong rather than wasteful. The walk
    // starts at the tile holding the window's first token -- taken at query row 0,
    // whose window reaches furthest back -- and the splits divide THAT range.
    const int wlo0 = WIN ? r4d_attn_wlo(ctx, qlen, 0, a.window) : 0;
    const int tb   = WIN ? wlo0 / TILE : 0;
    const int neff = ntl - tb;
    const int tps  = (neff + splits - 1) / splits;
    const int t_lo = tb + sp * tps;
    if (t_lo >= ntl) return;
    const int t_hi = min(t_lo + tps, ntl);

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const bool live = (qi < qlen);
    const int qrow = live ? qi : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
        }
    }

    // The row's absolute position. Causal masking bounds the key by it; non-causal
    // masking bounds the key by the sequence and lets the window bound it on BOTH
    // sides, which is what a block-diffusion drafter's own block needs.
    const int pos = ctx - qlen + qrow;
    const int cz  = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[seq] : a.causal;
    const int klimit = cz ? pos
                          : ((WIN && a.window > 0) ? min(ctx - 1, pos + a.window - 1)
                                                   : ctx - 1);
    const int wlo    = (WIN && a.window > 0) ? (pos - a.window + 1 > 0
                                               ? pos - a.window + 1 : 0) : 0;
    const int prefix_len = cz ? (ctx - qlen) : ctx;
    const uint16_t* p_tree = (const uint16_t*)a.q_descale;
    const uint16_t* seq_tree = (cz && p_tree) ? (p_tree + (size_t)seq * 16) : nullptr;
    const uint16_t my_ancestor_mask = (qrow < 16 && seq_tree) ? seq_tree[qrow] : (uint16_t)(qrow < 16 ? ((1u << (qrow + 1)) - 1u) : 0xFFFFu);
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int vg = lane >> 3, vj = lane & 7;
    uint4 vreg[VREGS];

    auto fetchV = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                int kk = min(k0 + kg * 8 + vj, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * (2 * HEAD_DIM) + HEAD_DIM + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const uint4 v = vreg[r];
                if (KVP) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else if (V8) {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    *(uint2*)&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8] = make_uint2(lo0, lo1);
                    *(uint2*)&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8] = make_uint2(hi0, hi1);
                } else {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    uint32_t wl[4], wh[4];
                    fp8x8_to_16x4w<D>(make_uint2(lo0, lo1), wl);
                    fp8x8_to_16x4w<D>(make_uint2(hi0, hi1), wh);
                    uint16_t* a0 = &sV16[(d0 + 2 * vj)     * VSTR + kg * 8];
                    uint16_t* a1 = &sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8];
                    *(uint2*)(a0)     = make_uint2(wl[0], wl[1]);
                    *(uint2*)(a0 + 4) = make_uint2(wl[2], wl[3]);
                    *(uint2*)(a1)     = make_uint2(wh[0], wh[1]);
                    *(uint2*)(a1 + 4) = make_uint2(wh[2], wh[3]);
                }
            }
        }
    };

    // A tile's NB entries are read whole, and when a tile spans several blocks the last one can run
    // past the sequence's last block -- past the end of its row, and on the last row past the end
    // of the table. Every key is clamped to ctx - 1 before it picks an entry, so no entry past
    // `lastb` is ever used and clamping the index there changes no value that is. Only a window
    // that reaches past `lastb` takes the clamped form, so every other tile keeps its one wide
    // scalar load; at one block a tile no window can reach past it and the test folds away.
    const int lastb = (ctx - 1) / BS;
    auto load_bt = [&](int t0, int* out) {
        if (NB > 1 && t0 + NB - 1 > lastb) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[min(t0 + i, lastb)];
        } else {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[t0 + i];
        }
    };
    int blk[NB];
    if (BTS) load_bt(t_lo * NB, blk);
    if (GPREV) fetchV(t_lo * TILE, blk);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < t_hi) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (ti > t_lo) {
            if (LDSB) lds_barrier(); else __syncthreads();
        }

        if (KLDS) {
            #pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / 8); i += NTHR) {
                const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
                int kk = min(k0 + key, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM);
                uint16_t* kd = &sK[key * KSTR];
                uint32_t w4[4];
                if (KVP) {
                    const uint4 r = r4d_kload<KNT, uint4, r4d_ntu4>(
                        (const uint4*)((const uint16_t*)a.kv + o + ch * 8));
                    w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                    w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                } else {
                    fp8x8_to_16x4w<D>(r4d_kload<KNT, uint2, r4d_ntu2>(
                        (const uint2*)((const uint8_t*)a.kv + o + ch * 8)), w4);
                }
                *(uint2*)(kd + ch * 8)     = make_uint2(w4[0], w4[1]);
                *(uint2*)(kd + ch * 8 + 4) = make_uint2(w4[2], w4[3]);
            }
        }
        if (!GPREV) fetchV(k0, blk);
        storeV();
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS) {
                int kk = min(k0 + mt * 16 + c, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                     + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM))
                     * (KVP ? 2 : 1);
            }
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int g = 0; g < NKS; g += PF) {
                frag16 kb[PF];
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    if (KLDS) {
                        const uint16_t* kp = &sK[(mt * 16 + c) * KSTR + 16 * t];
                        if (CTG) {
                            const uint4 u = r4d_kload<KNT, uint4, r4d_ntu4>(
                                (const uint4*)(kp + 8 * h));
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h), *(const uint2*)(kp + 8 + 4 * h));
                        }
                    } else if (KVP) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        const uint2 l0 = *(const uint2*)(kp + (CTG ? 8 * h : 4 * h));
                        const uint2 h0 = *(const uint2*)(kp + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        kb[j] = D::mk(make_uint2(D::from_bf16w(l0.x), D::from_bf16w(l0.y)),
                                      make_uint2(D::from_bf16w(h0.x), D::from_bf16w(h0.y)));
                    } else {
                        const uint8_t* kp = grow + 16 * t;
                        uint32_t w4[4];
                        // CONTIG makes this ONE 8-byte load where the split map needs two 4-byte.
                        const uint2 raw = CTG ? *(const uint2*)(kp + 8 * h)
                                              : make_uint2(*(const uint32_t*)(kp + 4 * h),
                                                           *(const uint32_t*)(kp + 8 + 4 * h));
                        fp8x8_to_16x4w<D>(raw, w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    }
                }
                if (SGB) sched_barrier();
                #pragma unroll
                for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
            }
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            if (!MSKIP || wave_any(kbase + 7 >= prefix_len) || (WIN && wave_any(kbase < wlo || (!cz && kbase + 7 > klimit)))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int k = kbase + e;
                    sc[e] = r4d_attn_is_visible(k, prefix_len, ctx, klimit, my_ancestor_mask, wlo, WIN, (cz != 0))
                          ? sc[e] : -INFINITY;
                }
            }
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);

            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                if (ti > t_lo || mt > 0) {
                    const float alpha = ex2(m_ref - mnew);
                    #pragma unroll
                    for (int dt = 0; dt < NKS; ++dt)
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                    l_i *= alpha;
                }
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off); pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            {
                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
                }
                const uint2 lo  = make_uint2(pw[0], pw[1]);
                const uint2 hi2 = make_uint2(pw[2], pw[3]);
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);          // QK already left the keys lane-local
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR;
                        if (V8) {
                            uint32_t w4[4];
                            fp8x8_to_16x4w<D>(*(const uint2*)&sV8[(c + (g + j) * 16) * VSTR8 +
                                                                 mt * 16 + 8 * h], w4);
                            vb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                        } else if (CTG) {
                            const uint4 u = *(const uint4*)vp;
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            vb[j] = D::mk(*(const uint2*)(vp + 4 * h),
                                          *(const uint2*)(vp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                }
            }
        }
        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }

    if (!live) return;
    const float lrow = l_i + swap16(l_i);
    const int tok = qoff + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead) * splits + sp;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    if (PF16) {
        // O/l in f16: bounded by max|v|, so no overflow, and the merge weight becomes w*l.
        uint16_t* so = (uint16_t*)a.scratch;
        float* sm = (float*)(so + nrow * HEAD_DIM);
        float* sl = sm + nrow;
        uint16_t* op = so + idx * HEAD_DIM;
        const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) tw[e >> 1] = DT16<1>::pk(acc[dt][e] * inv,
                                                                    acc[dt][e + 1] * inv);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    } else {
        float* so = (float*)a.scratch;
        float* sm = so + nrow * HEAD_DIM;
        float* sl = sm + nrow;
        float* op = so + idx * HEAD_DIM;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    }
}

// Vectorised merge, and the one the h256 launcher takes whenever the split count fits a wave.
// The arithmetic is identical to r4d_attn_splitkv_combine_kernel below, bit for bit; what changes
// is the thread map. One WAVE owns one output row and each lane owns EIGHT of that row's 256
// dims, so a segment is one 16-byte load per lane instead of eight 2-byte loads -- the same 512
// bytes per instruction across the wave, at an eighth of the instructions -- and one block
// carries JPB rows. There is no barrier and no LDS anywhere in it, which is also what lets the
// rows sharing a block have different contexts, and so different segment counts.
// ---- THE OUTPUT GATE AND THE fp8 QUANTISER AS THE MERGE'S EPILOGUE (attn_paged_gate_quant) ----
//
// A gated attention's output goes from here straight into `gate_quant_fp8`: bf16(sigmoid(gate)),
// times the merged row, rounded to bf16, and E4M3 codes with one scale per 128 columns taken over
// that product. A merge row is one (token, head) pair and 128 columns are one scale group, so the
// row that is merged here is exactly the rows the quantiser would read back -- and it is still in
// registers. The `_gq` forms below finish it in place, with no dispatch between the two.
//
// BYTE-FOR-BYTE `gate_quant_fp8`'s arithmetic, which r4d_fused_quant_fp8.hip states: the gate
// rounded to bf16 before the multiply, the product rounded, the group's absmax (a max, so the
// order it is taken in cannot matter), scale = amax / 448 or 1 at zero, codes over product / scale
// clamped to +-448 -- and the bf16 product is what lands in `out`, as the in-place pair leaves it.
// A padded slot merges to zeros and then takes the same epilogue, as the pair would give it.
//
// GQ == 2 IS THE SAME EPILOGUE AT INT8 CODES, for an int8 out projection: scale amax / 127 and the
// value over it rounded to nearest-even (r4d_q8.h) -- gate_quant_fp8's int8 form, byte for byte.
struct R4DGateQuant {
    const uint16_t* g;          // gate, row `job` at g + job * g_ld
    uint8_t*        q;          // E4M3 (or int8, `i8`) codes, row `job` at q + job * q_ld
    float*          s;          // one scale per 128 columns, row `job` at s + job * s_ld
    unsigned        g_ld, q_ld, s_ld;
    int             i8;         // read by the launcher only: which combine to instantiate
};
constexpr int kGqGroup = 128;

__device__ __forceinline__ uint16_t r4d_gq_gate(uint16_t gb, uint16_t xb) {
    const float    gv = bf16_to_f32(gb);
    const uint16_t ab = f32_to_bf16(1.0f / (1.0f + expf(-gv)));
    return f32_to_bf16(bf16_to_f32(ab) * bf16_to_f32(xb));
}
__device__ __forceinline__ float r4d_gq_clamp(float v) { return fminf(fmaxf(v, -448.0f), 448.0f); }

template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN, int GQ>
__device__ __forceinline__
void splitkv_combine_vec_body(const R4DArgs& a, int splits, int tile, const R4DGateQuant& gq)
{
    constexpr int DPL = HEAD_DIM / 32;                 // dims a lane owns
    const int lane = threadIdx.x & 31;
    const int job  = blockIdx.x * JPB + (int)(threadIdx.x >> 5);
    if (job >= r4d_q_total(a) * a.q_heads) return;
    const int tok = job / a.q_heads;
    const int seq = r4d_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];
    uint16_t* outp = (uint16_t*)a.out + (size_t)job * HEAD_DIM + lane * DPL;

    /* The row's bf16 values, then either the plain store or the gated, quantised one. A lane's
     * DPL columns sit inside one 128-column group, and the group is the 128 / DPL lanes around it,
     * so its absmax is a butterfly over those lanes and nothing wider. */
    auto finish = [&](const uint16_t (&ob)[DPL]) {
        if constexpr (!GQ) {
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } o_;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) o_.w[j] = ob[j];
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
        } else {
            constexpr int LPG = kGqGroup / DPL;        // lanes a scale group
            static_assert(DPL == 8 || DPL == 4, "a lane's columns are one or two dwords of codes");
            const uint16_t* gp = gq.g + (size_t)job * gq.g_ld + lane * DPL;
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } gi, o_;
            if (DPL == 8) gi.u4 = *(const uint4*)gp; else gi.u2 = *(const uint2*)gp;
            float v[DPL];
            float m = 0.0f;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) {
                o_.w[j] = r4d_gq_gate(gi.w[j], ob[j]);
                v[j] = bf16_to_f32(o_.w[j]);
                m = fmaxf(m, fabsf(v[j]));
            }
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
            if constexpr (LPG > 16) m = fmaxf(m, r4d_xor_lane<16>(m));
            if constexpr (LPG > 8)  m = fmaxf(m, r4d_xor_lane<8>(m));
            m = fmaxf(m, r4d_xor_lane<4>(m));
            m = fmaxf(m, r4d_xor_lane<2>(m));
            m = fmaxf(m, r4d_xor_lane<1>(m));
            const float sc  = m > 0.0f ? m * (1.0f / q8_top<GQ == 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((lane % LPG) == 0) gq.s[(size_t)job * gq.s_ld + (lane * DPL) / kGqGroup] = sc;
            uint32_t pk[DPL / 4];
            #pragma unroll
            for (int w = 0; w < DPL / 4; ++w)
                pk[w] = q8_pack4<GQ == 2>(v[4 * w + 0], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3],
                                          inv);
            uint8_t* qp = gq.q + (size_t)job * gq.q_ld + lane * DPL;
            if (DPL == 8) *(uint2*)qp = make_uint2(pk[0], pk[DPL / 4 - 1]);
            else          *(uint32_t*)qp = pk[0];
        }
    };

    if (ctx <= 0) {                                    // padded slot: zeros, read nothing
        uint16_t ob[DPL];
        #pragma unroll
        for (int i = 0; i < DPL; ++i) ob[i] = 0;
        finish(ob);
        return;
    }
    const int nef = r4d_attn_tiles(ctx, tile)
                  - (WIN ? r4d_attn_wlo(ctx, r4d_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = (nef + tp_ - 1) / tp_;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    // max is exact in any order, and taken sequentially it is wave-uniform, so this is a scalar
    // walk. The sum keeps the LDS tree association: the stride-32 pre-accumulation is that tree
    // off=32 round, and the xor butterfly 16..1 is the rest of it, leaving the value in every lane.
    float gmax = -INFINITY;
    for (int s = 0; s < u; ++s) gmax = fmaxf(gmax, sm[base + s]);
    float tot = 0.0f;
    for (int s = lane; s < u; s += 32) {
        const float mm = sm[base + s];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        tot += w * sl[base + s];
    }
    tot = r4d_wave_sum(tot);

    float ch[CHAINS][DPL];
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i)
        #pragma unroll
        for (int j = 0; j < DPL; ++j) ch[i][j] = 0.0f;

    auto wgt = [&](int sx) {
        const float mm = sm[base + sx];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        return PF16 ? w * sl[base + sx] : w;
    };

    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + lane * DPL;
        // One aligned access a lane a segment: 16 bytes at head_dim 256, 8 at 128.
        union { uint4 u4; uint2 u2; __fp16 h[DPL]; } r;
        auto rd = [&](int sx) {
            if (DPL == 8) r.u4 = *(const uint4*)(op + (size_t)sx * HEAD_DIM);
            else          r.u2 = *(const uint2*)(op + (size_t)sx * HEAD_DIM);
        };
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                rd(s + i);
                const float w = wgt(s + i);
                #pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, (float)r.h[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            rd(s);
            const float w = wgt(s);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, (float)r.h[j], ch[0][j]);
        }
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + lane * DPL;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                const float* p = op + (size_t)(s + i) * HEAD_DIM;
                const float w = wgt(s + i);
                #pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, p[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            const float* p = op + (size_t)s * HEAD_DIM;
            const float w = wgt(s);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, p[j], ch[0][j]);
        }
    }

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);
    const float vd = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    uint16_t ob[DPL];
    #pragma unroll
    for (int j = 0; j < DPL; ++j) {
        float av = 0.0f;
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) av += ch[i][j];
        // (av * vd) / tot, in that order: folding vd/tot into one reciprocal first is a different
        // rounding and shows up as a whole bf16 ULP on the merged output.
        ob[j] = f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f);
    }
    finish(ob);
}

template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0>
__global__ __launch_bounds__(32 * JPB)
void r4d_attn_splitkv_combine_vec_kernel(const R4DArgs a, int splits, int tile)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, 0>(a, splits, tile,
                                                                   R4DGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(32 * JPB)
void r4d_attn_splitkv_combine_vec_gq_kernel(const R4DArgs a, int splits, int tile,
                                            const R4DGateQuant gq)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, GQ>(a, splits, tile, gq);
}

// Merge. PF16=1 reads normalised f16 partials and folds l into the weight; PF16=0 is the plain
// path (unnormalised f32 partials, l summed separately). Both walk the segments in FIXED INDEX
// ORDER so a request's rounding never depends on CTA scheduling.
template<int HEAD_DIM, int CHAINS, int PF16, int WIN, int GQ>
__device__ __forceinline__
void splitkv_combine_body(const R4DArgs& a, int splits, int tile, const R4DGateQuant& gq)
{
    extern __shared__ float sw[];
    __shared__ float red[HEAD_DIM];
    const int job = blockIdx.x, d = threadIdx.x;
    const int tok = job / a.q_heads;
    const int seq = r4d_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];

    /* One thread a column, so a 128-column scale group is four waves: each takes its own absmax
     * and the group's is the max over its four, through `red` -- whose last read, the total's,
     * is behind a barrier before the merge loop, and which a padded slot never used. */
    auto finish = [&](uint16_t ob) {
        uint16_t* op = (uint16_t*)a.out + (size_t)job * HEAD_DIM + d;
        if constexpr (!GQ) {
            *op = ob;
        } else {
            constexpr int WPG = kGqGroup / 32;         // waves a scale group
            const uint16_t h = r4d_gq_gate(gq.g[(size_t)job * gq.g_ld + d], ob);
            *op = h;
            const float v = bf16_to_f32(h);
            const float m = r4d_wave_max(fabsf(v));
            if ((d & 31) == 0) red[d / 32] = m;
            __syncthreads();
            const int g0 = (d / kGqGroup) * WPG;
            float gm = red[g0];
            #pragma unroll
            for (int w = 1; w < WPG; ++w) gm = fmaxf(gm, red[g0 + w]);
            const float sc  = gm > 0.0f ? gm * (1.0f / q8_top<GQ == 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((d % kGqGroup) == 0) gq.s[(size_t)job * gq.s_ld + d / kGqGroup] = sc;
            if constexpr (GQ == 2) {
                gq.q[(size_t)job * gq.q_ld + d] =
                    (uint8_t)(int)rintf(fminf(fmaxf(v * inv, -127.0f), 127.0f));
            } else {
                const int p = __builtin_amdgcn_cvt_pk_fp8_f32(r4d_gq_clamp(v * inv), 0.0f, 0, false);
                gq.q[(size_t)job * gq.q_ld + d] = (uint8_t)(p & 0xff);
            }
        }
    };

    // A padded batch slot has no context at all. Its output is discarded by the caller, but the
    // segment count below would divide by zero, so leave zeros and stop. `ctx` is block-uniform,
    // so every thread takes this branch together and the barriers further down stay matched.
    if (ctx <= 0) { finish(0); return; }
    // Segment count, derived the same way the decode kernel derived it -- including the
    // window's tile floor, or the merge would read partials no segment ever wrote.
    const int nef = r4d_attn_tiles(ctx, tile)
                  - (WIN ? r4d_attn_wlo(ctx, r4d_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = (nef + tp_ - 1) / tp_;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    float gmax = -INFINITY;
    for (int s = d; s < u; s += HEAD_DIM) gmax = fmaxf(gmax, sm[base + s]);
    red[d] = gmax; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] = fmaxf(red[d], red[d + off]);
        __syncthreads();
    }
    gmax = red[0]; __syncthreads();

    float tot = 0.0f;
    for (int s = d; s < u; s += HEAD_DIM) {
        const float mm = sm[base + s];
        float w = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        if (PF16) { w *= sl[base + s]; sw[s] = w; tot += w; }
        else      { sw[s] = w;         tot += w * sl[base + s]; }
    }
    red[d] = tot; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] += red[d + off];
        __syncthreads();
    }
    tot = red[0]; __syncthreads();

    float ch[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) ch[i] = 0.0f;
    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], (float)__builtin_bit_cast(
                    __fp16, op[(size_t)(s + i) * HEAD_DIM]), ch[i]);
        }
        for (; s < u; ++s)
            ch[0] = fmaf(sw[s], (float)__builtin_bit_cast(__fp16, op[(size_t)s * HEAD_DIM]), ch[0]);
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], op[(size_t)(s + i) * HEAD_DIM], ch[i]);
        }
        for (; s < u; ++s) ch[0] = fmaf(sw[s], op[(size_t)s * HEAD_DIM], ch[0]);
    }
    float av = 0.0f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) av += ch[i];

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);
    const float vd = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    finish(f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f));
}

template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0>
__global__ __launch_bounds__(HEAD_DIM)
void r4d_attn_splitkv_combine_kernel(const R4DArgs a, int splits, int tile)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, 0>(a, splits, tile, R4DGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(HEAD_DIM)
void r4d_attn_splitkv_combine_gq_kernel(const R4DArgs a, int splits, int tile,
                                        const R4DGateQuant gq)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, GQ>(a, splits, tile, gq);
}
