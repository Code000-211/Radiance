// R4D DFlash2 draft-step preamble -- everything the DFlash speculator does between the fused
// context K/V projection GEMM and the draft forward, in ONE launch. Replaces the ~10-launch eager
// chain the serving path runs per step (dflash/speculator.py::propose):
//
//   prepare_dflash_inputs            1 Triton launch   (query ids/positions/slots, sample tables,
//                                                       context positions/slots, padding)
//   _normalize_context_k             rms_norm          (per-layer K norm, [L, T, nkv, 128])
//   ops.rotary_embedding             1 launch          (NeoX rope on K, positions repeated x L)
//   permute(...).contiguous()        1 copy            (layer-major K / V split)
//   do_kv_cache_update x L           L launches        (reshape_and_cache_flash, fp8 or bf16)
//
// The GEMM itself (F.linear over the L stacked KV projections) stays where it is: this kernel is
// "everything between the projection and the draft forward", the same cut qsa_index_prep makes on
// the target side. The hidden_norm that precedes the GEMM stays too.
//
// GRID. The Triton prepare kernel's own layout: (blocks per request, requests) plus one padding
// column. Program (b, r) owns request r's span rows j in [b*TPP, (b+1)*TPP) of the
// [0, num_ctx_r + Q) span the prepare kernel walks (Q = 1 + num_speculative): j < num_ctx is a
// context token t = start_r + j, the rest are the request's Q draft query rows. A program knows
// its request from blockIdx.y -- no search, no LDS, no barrier -- and a program past its
// request's span exits at once, as the prepare kernel's masked blocks do. Measured
// (verifiers/dflash2prep): the fastest of the layouts tried on the shapes serving captures,
// single-stream decode, 16 x 8 decode and the 4096-token prefill chunk; its one cost is empty
// programs on a mixed batch with one long request beside short ones, which runs eager.
//
// WHAT IT COMPUTES, per request r:
//   context row t = start_r + j: valid = j < num_ctx - num_rejected[r], pos = valid ?
//     positions[t] : 0. For every kv-cache group g: block = block_table_g[r, min(pos / bs_g,
//     cols-1)]; slot = (valid && block != 0) ? block*bs_g + pos % bs_g : PAD (-1).
//     context_positions[t] = pos, context_slot_g[t] = slot   (the rejected suffix gets 0 / PAD,
//     written on purpose: a replayed graph must never see a stale row).
//     Then, valid rows only, per (token, layer l, kv head h) item -- 16 lanes per item, 8
//     elements (16 B) per lane, two items per wave:
//       x    = fp32(kv[t, l, K, h, :])                          from the fused projection
//       y    = bf16( x * rsqrt(mean(x^2) + eps) * w_k[l] )     vllm rms_norm_kernel lineage:
//                                                              fp32 sum of squares, one bf16 round
//       y    = rope(y, pos)                                     NeoX pairs (i, i+rot/2), bf16 cos/sin,
//                                                              fp32 math, one bf16 round per element
//       cache_l[slot_{group(l)}][h][:] = K: convert(y), V: convert(kv[t, l, V, h, :])
//       convert = e4m3(bf16 / scale) with the fp8 cache (reshape_and_cache_flash's exact ISA
//       sequence, transcribed from mischip/reshape_and_cache_flash), or the bf16 bits as they are.
//   query row off = j - num_ctx in [0, Q): query_idx = r*Q + off,
//     last_valid_pos = positions[start + num_ctx - num_rejected - 1], query_pos = last_valid_pos
//     + 1 + off. input_ids[query_idx] = off == 0 ? bonus : mask_token (bonus = last_sampled[state]
//     when num_sampled[r] > 0, else next_prefill_tokens[state]: the chunked-prefill splice);
//     positions[query_idx] = min(query_pos, max_model_len-1); query_slot_g[query_idx] as above
//     (UNclamped pos, block 0 is the null block -> PAD). off >= 1: sample_indices / sample_pos /
//     sample_idx_mapping at r*S + off - 1. off == 0: query_start_loc[r] = r*Q, seq_lens[r] =
//     min(last_valid_pos + 1 + Q, max_model_len), temperature / seeds copied at the state index.
//   PADDING program (column max_blocks, row 0): query_start_loc[num_reqs .. max_num_reqs] =
//     num_reqs*Q, seq_lens[num_reqs .. max_num_reqs) = 0, sample tables past num_reqs*S ->
//     (0, 0, -1), query slots past num_reqs*Q -> PAD in every group.
//
// LOADS. Branch-free, the library's rule: every lane computes the address of everything it
// will need -- its item's K / V vectors and norm weights, its token's position, then from the
// position its block-table entry and cos/sin row -- with an absent item pointed at the
// request's first row and masked afterwards, so nothing sits under a divergent branch and all
// of a level's loads go out together. The request scalars come through the scalar cache. Two
// dependent levels (position -> table, cos/sin) is the depth of the problem.
//
// PACKING. 256 / 512 / 1024 threads = 16 / 32 / 64 item slots, a slot carrying one or two
// (token, layer, head) items with both items' loads issued together. Measured on the 9070 XT
// (results-graphtime): below 96 tokens one token per 256-thread program wins (the widest front
// at the lowest latency); at and above it, 8 tokens per 1024-thread program (two items per
// slot at 10 rows) wins by dividing the workgroup count, which is what limited prefill.
//
// PARITY. Integer logic is prepare_dflash_inputs' bit for bit (context parallel size 1 only; a
// CP > 1 serve is refused at the entry point). The norm follows vllm's rms_norm_kernel
// (fp32 variance, rsqrtf(var/128 + eps), out = bf16(x * s * w)) with the sum-of-squares ORDER
// unspecified -- the reference reduces through cub::BlockReduce, this kernel through a 16-lane
// butterfly -- so a bf16 tie may land either side, the same caveat qsa_index_prep carries. The
// rope is vllm's rotary_embedding_kernel arithmetic in fp32 with one bf16 round;
// R4D_DFLASH2_ROPE_FMA (default 1) selects hipcc's contracted form; the verifier
// (verifiers/dflash2prep) measured both arms bit-identical under it. The fp8 convert is
// bit-exact to Triton's gfx1201 sequence: IEEE division by the scale, RTNE at 3 kept mantissa
// bits, saturation to 448 at |x| >= 464, the 8-rung subnormal ladder, NaN -> 0x7f | sign.
//
// GEOMETRY. head_dim 128 (in the name); num_layers <= 8 and num_kv_heads <= 8 with
// num_layers * num_kv_heads <= 64 rows per token; kv-cache groups <= 4; rotary a multiple of 16
// up to 128, NeoX only; tokens per program <= 8; max_query_len is the host-known largest
// per-request scheduled token count (the Triton wrapper's num_scheduled_tokens.max()). Strides
// are in ELEMENTS of the cache dtype (1 byte fp8, 2 bytes bf16) and must be multiples of 8
// (16-byte lane vectors). Nothing allocates or synchronises; the grid is a function of
// host-known counts, so the launch records into a HIP graph.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_common.h"
#include <cstdint>

#ifndef R4D_DFLASH2_ROPE_FMA
#define R4D_DFLASH2_ROPE_FMA 1
#endif

#define DP_HEAD_DIM  128
#define DP_MAX_ROWS  64      // num_layers * num_kv_heads
#define DP_MAX_TPP   8
#define DP_MAX_ITEMS 2       // items one slot may carry
#define DP_PAD_SLOT  (-1L)

namespace {

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}

// f32 -> fp8 e4m3fn, transcribed from Triton's gfx1201 ISA (mischip/reshape_and_cache_flash,
// tools/isa/variant_b.amdgcn lines 143-284). Change only against a fresh ISA dump.
__device__ __forceinline__ uint32_t fp8_e4m3_code(float x) {
    const uint32_t b = __float_as_uint(x);
    const uint32_t a = b & 0x7fffffffu;
    uint32_t t = a + ((a >> 20) & 1u) + 0x7ffffu;
    t &= 0xfff00000u;
    t = t < 0x3c800000u ? 0x3c800000u : t;
    t += 0x04000000u;
    uint32_t c = t >> 20;
    c = (a < 0x43e80000u) ? c : 0x7eu;
    c = (a > 0x3c6fffffu) ? c : 7u;
    c = (a > 0x3c500000u) ? c : 6u;
    c = (a > 0x3c2fffffu) ? c : 5u;
    c = (a > 0x3c100000u) ? c : 4u;
    c = (a > 0x3bdfffffu) ? c : 3u;
    c = (a > 0x3ba00000u) ? c : 2u;
    c = (a > 0x3b3fffffu) ? c : 1u;
    c = (a > 0x3a800000u) ? c : 0u;
    c = (x == x) ? c : 0x7fu;
    return (c & 0x7fu) | ((b >> 24) & 0x80u);
}

// Sum across the 16 lanes that share a row (the two halves of a wave hold two rows).
__device__ __forceinline__ float row_sum16(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

__device__ __forceinline__ float rope_x(float x, float c, float y, float s) {
#if R4D_DFLASH2_ROPE_FMA
    return fmaf(x, c, -(y * s));
#else
    return x * c - y * s;
#endif
}
__device__ __forceinline__ float rope_y(float y, float c, float x, float s) {
#if R4D_DFLASH2_ROPE_FMA
    return fmaf(y, c, x * s);
#else
    return y * c + x * s;
#endif
}

// prepare_dflash_inputs' slot law at CP 1 from an already-loaded table entry: block 0 is the
// null block, an invalid row is PAD.
__device__ __forceinline__ long slot_from(int bid, int bs, long pos, bool valid) {
    return (valid && bid != 0) ? (long)bid * bs + (pos % bs) : DP_PAD_SLOT;
}
// The table column the law reads for a position (clamped into the table).
__device__ __forceinline__ long bt_col(const R4DDflash2PrepArgs& a, int g, long pos) {
    long bn = pos / a.block_size[g];
    const long cols = a.bt_stride[g];
    return bn < cols - 1 ? bn : cols - 1;
}

// Per-program wall-clock stamps, when a.prof is set: DP_PROF_STAMPS u64 per program from the
// constant-rate device counter (r4d_dflash2_prep_wall_clock_khz() converts), thread 0 only:
//   0 entry   1 request scalars in   2 item vectors and positions in (norm scale computed)
//   3 table entry and cos/sin in (rope done)   4 stores issued   5 exit
// Programs that exit early write 0 and 5 only. Null in serving: one uniform branch per stamp.
#define DP_PROF_STAMPS 6
struct Prof {
    unsigned long long* p;
    __device__ __forceinline__ Prof(unsigned long long* prof, int prog, int tid)
        : p(prof && tid == 0 ? prof + DP_PROF_STAMPS * (long)prog : nullptr) {
        if (p) p[0] = wall_clock64();
    }
    __device__ __forceinline__ void at(int i) { if (p) p[i] = wall_clock64(); }
    __device__ __forceinline__ ~Prof() { if (p) p[DP_PROF_STAMPS - 1] = wall_clock64(); }
};

__device__ __forceinline__ void unpack8(const uint4 v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = bf16_to_f32((uint16_t)(w[i >> 1] >> (16 * (i & 1))));
}

template <bool FP8, int THREADS>
__global__ __launch_bounds__(THREADS) void r4d_dflash2_prep_kernel(const R4DDflash2PrepArgs a) {
    constexpr int SLOTS = THREADS / 16;
    const int b    = blockIdx.x;                     // block within the request's span
    const int r    = blockIdx.y;                     // request (batch row)
    const int tid  = threadIdx.x;
    const int Q    = a.num_query_per_req;
    const int S    = a.num_speculative_steps;
    const int TPP  = a.tokens_per_program;           // resolved by the launcher, 1..8
    const int max_blocks = (a.max_query_len + Q + TPP - 1) / TPP;
    Prof prof(a.prof, r * (max_blocks + 1) + b, tid);

    // ---------------------------------------------------------------- padding program
    if (b == max_blocks) {
        if (r != 0) return;
        const int nq = a.num_reqs * Q;
        for (int i = a.num_reqs + tid; i <= a.max_num_reqs; i += THREADS)
            a.query_start_loc[i] = nq;
        for (int i = a.num_reqs + tid; i < a.max_num_reqs; i += THREADS)
            a.seq_lens[i] = 0;
        for (int i = a.num_reqs * S + tid; i < a.max_num_reqs * S; i += THREADS) {
            a.sample_indices[i]    = 0;
            a.sample_pos[i]        = 0;
            a.sample_idx_mapping[i] = -1;
        }
        for (int g = 0; g < a.num_groups; ++g)
            for (int i = nq + tid; i < a.max_num_tokens; i += THREADS)
                a.query_slot_mapping[g][i] = DP_PAD_SLOT;
        return;
    }

    // ---------------------------------------------------------------- request scalars
    const int ctx_start = a.target_query_start_loc[r];
    const int ctx_end   = a.target_query_start_loc[r + 1];
    const int num_ctx   = ctx_end - ctx_start;
    const int j0        = b * TPP;
    if (j0 >= num_ctx + Q) return;                   // past this request's span
    const int num_rejected  = a.num_rejected[r];
    const int num_valid_ctx = num_ctx - num_rejected;
    const int state         = a.idx_mapping[r];
    const long last_valid_pos = a.target_positions[ctx_start + num_valid_ctx - 1];
    prof.at(1);

    // ---------------------------------------------------------------- items: level 0 loads
    // Every lane addresses its item unconditionally; an item past the span reads row
    // ctx_start (always a scheduled token) and is masked at the store.
    const int lane = tid & 31;
    const int sub  = lane & 15;
    const int slotid = tid >> 4;
    const int e0   = sub * 8;
    const int nkv  = a.num_kv_heads;
    const int rows = a.num_layers * nkv;
    const int rot  = a.rotary;
    const int half = rot >> 1;
    const bool in_x = e0 < half;
    const bool in_y = !in_x && e0 < rot;
    const int  src  = in_x ? lane + (half >> 3) : (in_y ? lane - (half >> 3) : lane);
    const int  ci   = in_x ? e0 : e0 - half;

    uint4 kraw[DP_MAX_ITEMS], vraw[DP_MAX_ITEMS], wraw[DP_MAX_ITEMS];
    long  ipos[DP_MAX_ITEMS];
    int   il[DP_MAX_ITEMS], ih[DP_MAX_ITEMS];
    bool  ivalid[DP_MAX_ITEMS];
#pragma unroll
    for (int k = 0; k < DP_MAX_ITEMS; ++k) {
        const int item = k * SLOTS + slotid;
        const int jj = item / rows;
        const int rr = item - jj * rows;
        const int j  = j0 + jj;
        il[k] = rr / nkv;
        ih[k] = rr - il[k] * nkv;
        const bool ctx = jj < TPP && j < num_ctx;
        ivalid[k] = ctx && j < num_valid_ctx;
        const int t = ctx_start + (ctx ? j : 0);
        const uint16_t* __restrict__ kvrow = (const uint16_t*)a.kv + (long)t * a.kv_row_stride;
        kraw[k] = *(const uint4*)(kvrow + ((long)(il[k] * 2) * nkv + ih[k]) * DP_HEAD_DIM + e0);
        vraw[k] = *(const uint4*)(kvrow + ((long)(il[k] * 2 + 1) * nkv + ih[k]) * DP_HEAD_DIM + e0);
        wraw[k] = *(const uint4*)((const uint16_t*)a.k_norm_w + il[k] * DP_HEAD_DIM + e0);
        ipos[k] = a.target_positions[t];
    }
    // The per-token rows this program writes (context positions / slots, query rows): thread
    // jj < TPP owns span row j0 + jj. Its position load goes out with the item loads.
    const int  my_j     = j0 + tid;
    const bool my_ctx   = tid < TPP && my_j < num_ctx;
    const bool my_valid = my_ctx && my_j < num_valid_ctx;
    const bool my_query = tid < TPP && my_j >= num_ctx && my_j < num_ctx + Q;
    const long my_tpos  = a.target_positions[ctx_start + (my_ctx ? my_j : 0)];

    // ---------------------------------------------------------------- level 1: table, cos/sin
    // norm scale first: it needs only the item vectors, so it overlaps the dependent loads
    float x[DP_MAX_ITEMS][8], wf[DP_MAX_ITEMS][8], sc[DP_MAX_ITEMS];
#pragma unroll
    for (int k = 0; k < DP_MAX_ITEMS; ++k) {
        unpack8(kraw[k], x[k]);
        unpack8(wraw[k], wf[k]);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += x[k][i] * x[k][i];
        ss = row_sum16(ss);
        sc[k] = rsqrtf(ss / (float)DP_HEAD_DIM + a.eps);
    }
    prof.at(2);
    uint4 craw[DP_MAX_ITEMS], sraw[DP_MAX_ITEMS];
    int   ibid[DP_MAX_ITEMS];
#pragma unroll
    for (int k = 0; k < DP_MAX_ITEMS; ++k) {
        const long p = ivalid[k] ? ipos[k] : 0L;
        ipos[k] = p;
        const int g = a.layer_group[il[k]];
        ibid[k] = a.block_table[g][(long)r * a.bt_stride[g] + bt_col(a, g, p)];
        const uint16_t* __restrict__ cs = (const uint16_t*)a.cos_sin + p * rot + ci;
        craw[k] = *(const uint4*)cs;
        sraw[k] = *(const uint4*)(cs + half);
    }
    // this thread's token row: table entries for every group
    const long my_pos = my_valid ? my_tpos : 0L;
    const long my_qpos = last_valid_pos + 1 + (my_j - num_ctx);   // meaningful when my_query
    const long my_lpos = my_query ? my_qpos : my_pos;             // the position the row keys on
    int my_bid[R4D_DFLASH2_PREP_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < R4D_DFLASH2_PREP_MAX_GROUPS; ++g)
        my_bid[g] = g < a.num_groups
                  ? a.block_table[g][(long)r * a.bt_stride[g] + bt_col(a, g, my_lpos)] : 0;

    // ---------------------------------------------------------------- per-token rows
    if (my_ctx) {
        const int t = ctx_start + my_j;
        a.context_positions[t] = my_pos;
        for (int g = 0; g < a.num_groups; ++g)
            a.context_slot_mapping[g][t] = slot_from(my_bid[g], a.block_size[g], my_pos, my_valid);
    } else if (my_query) {
        const int off = my_j - num_ctx;
        const int query_idx = r * Q + off;
        int input_id = a.mask_token_id;
        if (off == 0)
            input_id = a.num_sampled[r] > 0 ? (int)a.last_sampled[state]
                                            : a.next_prefill_tokens[state];
        a.input_ids[query_idx] = input_id;
        a.positions[query_idx] = my_qpos < (long)a.max_model_len - 1 ? my_qpos
                                                                     : (long)a.max_model_len - 1;
        for (int g = 0; g < a.num_groups; ++g)
            a.query_slot_mapping[g][query_idx] = slot_from(my_bid[g], a.block_size[g], my_qpos, true);
        if (off >= 1) {
            const int si = r * S + off - 1;
            a.sample_indices[si]     = query_idx;
            a.sample_pos[si]         = my_qpos;
            a.sample_idx_mapping[si] = state;
        } else {
            a.query_start_loc[r] = r * Q;
            const long sl = last_valid_pos + 1 + Q;
            a.seq_lens[r] = (int)(sl < (long)a.max_model_len ? sl : (long)a.max_model_len);
            a.temperature_out[state] = a.temperature_in[state];
            a.seeds_out[state]       = a.seeds_in[state];
        }
    }

    // ---------------------------------------------------------------- items: rope, convert, store
#pragma unroll
    for (int k = 0; k < DP_MAX_ITEMS; ++k) {
        const int l = il[k], h = ih[k];
        const int g = a.layer_group[l];
        const long s = slot_from(ibid[k], a.block_size[g], ipos[k], ivalid[k]);
        float y[8], p[8], cosv[8], sinv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            y[i] = bf16_to_f32(f32_to_bf16(x[k][i] * sc[k] * wf[k][i]));  // the norm's bf16 round
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = __shfl(y[i], src, 32);
        unpack8(craw[k], cosv);
        unpack8(sraw[k], sinv);
        uint16_t kb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float o = y[i];
            if (in_x)      o = rope_x(y[i], cosv[i], p[i], sinv[i]);
            else if (in_y) o = rope_y(y[i], cosv[i], p[i], sinv[i]);
            kb[i] = f32_to_bf16(o);                                       // the rope's bf16 round
        }
        if (k == 0) prof.at(3);
        if (s == DP_PAD_SLOT) continue;                                   // masked item
        const long blk = s / a.block_size[g];
        const long off = s - blk * a.block_size[g];
        const long cbase = blk * a.cache_block_stride + (long)h * a.cache_head_stride
                         + off * a.cache_tok_stride + e0;
        if constexpr (FP8) {
            const float ks = a.kv_scales[l];
            const float vs = a.kv_scales[a.num_layers + l];
            uint8_t* __restrict__ cache = (uint8_t*)a.cache[l];
            float vf[8];
            unpack8(vraw[k], vf);
            uint2 kq = make_uint2(0u, 0u), vq = make_uint2(0u, 0u);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t kc = fp8_e4m3_code(bf16_to_f32(kb[i]) / ks);
                const uint32_t vc = fp8_e4m3_code(vf[i] / vs);
                if (i < 4) { kq.x |= kc << (8 * i); vq.x |= vc << (8 * i); }
                else       { kq.y |= kc << (8 * (i - 4)); vq.y |= vc << (8 * (i - 4)); }
            }
            *(uint2*)(cache + cbase) = kq;
            *(uint2*)(cache + cbase + a.cache_v_offset) = vq;
        } else {
            uint16_t* __restrict__ cache = (uint16_t*)a.cache[l];
            uint4 kq;
            kq.x = (uint32_t)kb[0] | ((uint32_t)kb[1] << 16);
            kq.y = (uint32_t)kb[2] | ((uint32_t)kb[3] << 16);
            kq.z = (uint32_t)kb[4] | ((uint32_t)kb[5] << 16);
            kq.w = (uint32_t)kb[6] | ((uint32_t)kb[7] << 16);
            *(uint4*)(cache + cbase) = kq;
            *(uint4*)(cache + cbase + a.cache_v_offset) = vraw[k];
        }
    }
    prof.at(4);
}

// Packing law, measured on the 9070 XT (results-graphtime): below 96 tokens one token per
// 256-thread program; at and above, 8 tokens per 1024-thread program (two items per slot at
// 10 rows, both issued together), capped so the items fit two per slot. A pinned tpp takes the
// smallest workgroup that holds its items at one per slot, or 1024 threads at two per slot.
struct Packing { int tpp; int threads; };
static Packing packing_for(int T, int rows, int pinned) {
    Packing p;
    if (pinned > 0) {
        p.tpp = pinned;
        const int items = pinned * rows;
        p.threads = items <= 16 ? 256 : (items <= 32 ? 512 : 1024);
        return p;
    }
    if (T < 96) { p.tpp = 1; p.threads = 256; return p; }
    int tpp = 128 / rows;
    if (tpp < 1) tpp = 1;
    if (tpp > DP_MAX_TPP) tpp = DP_MAX_TPP;
    p.tpp = tpp;
    p.threads = 1024;
    return p;
}

template <bool FP8, int THREADS>
int launch_wg(R4DDflash2PrepArgs a, hipStream_t stream) {
    const int max_blocks = (a.max_query_len + a.num_query_per_req + a.tokens_per_program - 1)
                         / a.tokens_per_program;
    hipLaunchKernelGGL((r4d_dflash2_prep_kernel<FP8, THREADS>), dim3(max_blocks + 1, a.num_reqs),
                       dim3(THREADS), 0, stream, a);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}

template <bool FP8>
int launch(const R4DDflash2PrepArgs* ap, hipStream_t stream) {
    if (!ap) return -4;
    const R4DDflash2PrepArgs& a = *ap;
    if (a.num_layers < 1 || a.num_layers > R4D_DFLASH2_PREP_MAX_LAYERS) return -1;
    if (a.num_kv_heads < 1 || a.num_layers * a.num_kv_heads > DP_MAX_ROWS) return -1;
    if (a.rotary < 16 || a.rotary > DP_HEAD_DIM || (a.rotary & 15)) return -2;
    if (a.num_groups < 1 || a.num_groups > R4D_DFLASH2_PREP_MAX_GROUPS) return -3;
    for (int l = 0; l < a.num_layers; ++l) {
        if (!a.cache[l]) return -3;
        if (a.layer_group[l] < 0 || a.layer_group[l] >= a.num_groups) return -3;
    }
    for (int g = 0; g < a.num_groups; ++g) {
        if (!a.block_table[g] || !a.query_slot_mapping[g] || !a.context_slot_mapping[g])
            return -3;
        if (a.block_size[g] < 1 || a.bt_stride[g] < 1) return -3;
    }
    if (a.T < 1 || a.num_reqs < 1 || a.num_reqs > 65535 || a.max_query_len < 1 ||
        a.max_query_len > a.T || a.num_query_per_req < 2 ||
        a.num_speculative_steps != a.num_query_per_req - 1 ||
        a.max_num_reqs < a.num_reqs || a.max_num_tokens < a.num_reqs * a.num_query_per_req ||
        a.max_model_len < 1 || a.cp_size != 1)
        return -4;
    if (!a.kv || !a.k_norm_w || !a.cos_sin || (FP8 && !a.kv_scales)) return -4;
    if (a.tokens_per_program < 0 || a.tokens_per_program > DP_MAX_TPP) return -5;
    if (a.kv_row_stride & 7) return -6;
    if ((a.cache_block_stride & 7) || (a.cache_head_stride & 7) || (a.cache_tok_stride & 7) ||
        (a.cache_v_offset & 7))
        return -6;
    const int rows = a.num_layers * a.num_kv_heads;
    const Packing p = packing_for(a.T, rows, a.tokens_per_program);
    if (p.tpp * rows > (p.threads / 16) * DP_MAX_ITEMS) return -5;
    R4DDflash2PrepArgs b = a;
    b.tokens_per_program = p.tpp;
    switch (p.threads) {
        case 256:  return launch_wg<FP8, 256>(b, stream);
        case 512:  return launch_wg<FP8, 512>(b, stream);
        default:   return launch_wg<FP8, 1024>(b, stream);
    }
}

}  // namespace

extern "C" int r4d_dflash2_prep_h128_fp8kv(const R4DDflash2PrepArgs* a, hipStream_t stream) {
    return launch<true>(a, stream);
}

extern "C" int r4d_dflash2_prep_h128_bf16kv(const R4DDflash2PrepArgs* a, hipStream_t stream) {
    return launch<false>(a, stream);
}

extern "C" void r4d_dflash2_prep_dims(int* head_dim, int* max_layers, int* max_groups,
                                      int* max_rows) {
    if (head_dim)   *head_dim   = DP_HEAD_DIM;
    if (max_layers) *max_layers = R4D_DFLASH2_PREP_MAX_LAYERS;
    if (max_groups) *max_groups = R4D_DFLASH2_PREP_MAX_GROUPS;
    if (max_rows)   *max_rows   = DP_MAX_ROWS;
}

// Rate of the counter the prof stamps are taken from, in kHz, for the given device. 0 when the
// runtime cannot say (then the stamps are still ordered, just unitless).
extern "C" long r4d_dflash2_prep_wall_clock_khz(int device) {
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess)
        return 0;
    return (long)khz;
}

// The packing a launch runs at for T, rows = num_layers * kv_heads and the caller's pin (0 =
// the launcher's choice): tokens per program and workgroup threads.
extern "C" void r4d_dflash2_prep_packing(int T, int rows, int tokens_per_program, int* tpp,
                                         int* threads) {
    const Packing p = packing_for(T, rows, tokens_per_program);
    if (tpp) *tpp = p.tpp;
    if (threads) *threads = p.threads;
}
// Programs one launch runs: (blocks per request + 1) * num_reqs, program index
// r * (blocks + 1) + b; the padding program is (blocks, 0).
extern "C" long r4d_dflash2_prep_programs(int T, int max_query_len, int num_reqs,
                                          int num_query_per_req, int rows,
                                          int tokens_per_program) {
    const int tpp = packing_for(T, rows, tokens_per_program).tpp;
    const long blocks = (max_query_len + num_query_per_req + tpp - 1) / tpp;
    return (blocks + 1) * (long)num_reqs;
}
extern "C" long r4d_dflash2_prep_prof_stamps(int T, int max_query_len, int num_reqs,
                                             int num_query_per_req, int rows,
                                             int tokens_per_program) {
    return (long)DP_PROF_STAMPS *
           r4d_dflash2_prep_programs(T, max_query_len, num_reqs, num_query_per_req, rows,
                                     tokens_per_program);
}
extern "C" int r4d_dflash2_prep_prof_stamps_per_program(void) { return DP_PROF_STAMPS; }
