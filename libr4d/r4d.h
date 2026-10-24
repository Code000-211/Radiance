// R4D -- the RDNA4 (gfx1201) kernel library: attention, gated delta net, all-reduce, GEMMs and
// the small model ops, compiled into one shared object. Plain C ABI: every entry point takes raw
// device pointers and a stream, so a consumer needs no C++ ABI coupling. Nothing on the launch
// paths allocates or synchronises, so any of them can be recorded into a HIP graph.
//
// Naming. An entry point is
//
//     r4d_<family>_<op>_<geometry it is compiled for>
//
// Every dimension in the geometry suffix is a compile-time constant, and the entry point rejects a
// mismatch instead of running. `attn_decode_h256_gqa6_fp8kv` serves head_dim 256, 6 queries per KV
// head and an fp8-e4m3 paged cache, nothing else; another head size gets its own instantiation
// beside it. Dimensions fixed for the whole library stay out of the names and are reported by
// r4d_*_dims() and the kernel registry: the paged block size (16), the query dtype (bf16
// everywhere, see r4d_attn_paged_h256_gqa6.hip) and the target architecture.
//
// Source files carry the entry point's name, geometry included:
// r4d_gdn_kkt_solve_k128_c64_bf16.hip holds r4d_gdn_kkt_solve_k128_c64_bf16. A unit providing a
// family that differs only in a suffix stops at the shared part (r4d_gdn_conv_w4_h128_bf16.hip,
// r4d_attn_paged_h256_gqa6.hip). Shared machinery that provides no kernel -- r4d_common.h,
// r4d_dt16.h, r4d_gdn_wmma.h -- is named for what it is.
//
// On an unsupported shape the attention and GDN entry points return a negative code; the
// all-reduce and GEMM entry points throw std::runtime_error, which the plugin shim's R4D_TRY turns
// into an ABI status code. Nothing throws across the .so boundary.
#pragma once
#include <hip/hip_runtime.h>

// Library version; expected to match the git tag it was built from, so a consumer can assert it
// linked the sources it pinned.
#define R4D_VERSION "0.5.0"

struct R4DArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3 or bf16
                                //   strides below are in elements of that dtype; K then V per slot
    const int*   block_table;   // (num_seqs, max_blocks)            int32
    const int*   seqused_k;     // (num_seqs,)                       int32
    void*        out;           // (total_q, q_heads, head_dim)      bf16
    const float* k_descale;     // (num_seqs, kv_heads)
    const float* v_descale;     // (num_seqs, kv_heads)
    const float* q_descale;     // unused: the query is bf16
    void*        scratch;       // split-KV partials (decode only), or null
    int num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks;
    long kv_block_stride;       // elements between consecutive blocks
    long kv_head_stride;        // elements between kv heads inside a block
    float scale;
    int  splits;                // decode only; 0 = let the split law choose
    int  max_ctx;               // host-visible context bound (seqused_k is device-side)
    int  window;                // sliding window, 0 = none (a full-attention layer)
    int  causal;                // 1 = key <= query; 0 = key < seq_len, window two-sided
    const void* causal_seq;     // (num_seqs,) uint8 per-sequence flag, or null for `causal`

    // ---- Ragged batches --------------------------------------------------------------------
    // A serving step mixes one sequence's prefill chunk with other sequences' single decode rows,
    // so query lengths differ per sequence.
    //
    // `cu_q` is cu_seqlens: [num_seqs + 1] int32 on the device, cu_q[0] == 0, cu_q[num_seqs] ==
    // total_q; sequence s owns query rows [cu_q[s], cu_q[s+1]) of q and out. Null means uniform:
    // row r of sequence `seq` is row `seq * q_len + r`.
    //
    // With cu_q set, `q_len` is the maximum query length in the batch (it sizes the grid and picks
    // the decode kernel's wave count) and `total_q` is the row count of q and out, host-side so the
    // split-KV scratch can be sized without reading device memory. Both are ignored when cu_q is
    // null.
    const int* cu_q;            // (num_seqs+1,) int32 device, or null for a uniform q_len
    int  total_q;               // rows of q/out; ignored unless cu_q is set

    // ---- Views the split-KV decode kernel reads (r4d_attn_decode_h256_gqa6.hip); every other
    // kernel leaves them at zero and reads q dense with one cache head per query group.
    //
    // `q_ld` is the element distance between q rows, 0 for dense (q_heads * head_dim): the q part
    // of a fused [q|k|v] projection read where it lies (attn_paged_qrow). A multiple of 4.
    //
    // `kv_rep` maps query groups onto cache heads: group g (q heads g*GQA .. g*GQA+GQA-1) reads
    // cache head g / kv_rep, and the descales are per cache head. kv_heads then counts the groups
    // (q_heads / GQA) and the cache holds kv_heads / kv_rep heads. 0 or 1 is one head a group. It
    // runs a gqa of GQA * kv_rep on the kernel compiled for GQA, whose workgroup holds at most 64
    // rows of q_len * GQA: the 31B's DFlash drafter, gqa 8 at a block of 16, on the gqa-4 kernel.
    long q_ld;
    int  kv_rep;
};

// Rows of q and out. The split-KV scratch is laid out by dense query row, so the host sizing hook
// and the device kernels must agree on this count; hence total_q is carried rather than read off
// the end of cu_q.
__host__ __device__ __forceinline__ int r4d_q_total(const R4DArgs& a) {
    return a.cu_q ? a.total_q : a.num_seqs * a.q_len;
}

extern "C" {
// ---- attention: paged, causal, varlen ------------------------------------------------------
// Compiled for head_dim 256, 6 queries per KV head, paged block size 16, bf16 query. The prefill
// kernel tiles the query; the decode kernel splits the KV and takes at most 64 query rows
// (q_len * gqa), the band a speculative-decode verify step falls in.
// Return 0 on success, negative on a shape this instantiation does not serve.
int  r4d_attn_prefill_h256_gqa6_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_bf16kv (const R4DArgs* a, hipStream_t stream);
// The DFlash2 drafter's geometry: head_dim 128, 4 queries per KV head, sliding window. Decode
// only (q_len * gqa <= 64).
int  r4d_attn_decode_h128_gqa4_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h128_gqa4_bf16kv (const R4DArgs* a, hipStream_t stream);
long r4d_attn_decode_h128_gqa4_scratch_bytes(const R4DArgs* a);
// ...with R4DArgs::kv_rep and q_ld read: a gqa of 4 * kv_rep over a cache of kv_heads / kv_rep.
int  r4d_attn_decode_h128_gqa4_rep_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h128_gqa4_rep_bf16kv (const R4DArgs* a, hipStream_t stream);
// Bytes of split-KV partial buffer one decode launch of this shape needs. Independent of the cache
// dtype: the partials are f16 either way.
long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a);
// The Gemma 4 dense sliding-window geometry: head_dim 256, 2 queries per KV head. Both bands:
// decode (q_len*gqa <= 64, i.e. q_len <= 32) and windowed prefill, which is the h256/gqa6
// r4d_attn_prefill_kernel with the window bit on.
int  r4d_attn_prefill_h256_gqa2_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa2_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa2_bf16kv (const R4DArgs* a, hipStream_t stream);
long r4d_attn_decode_h256_gqa2_scratch_bytes(const R4DArgs* a);
/* Gemma 3 sliding window (r4d_attn_paged_h128_gqa2w.hip): head_dim 128, gqa 2, window >= 1. */
int  r4d_attn_prefill_h128_gqa2w_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h128_gqa2w_bf16kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h128_gqa2w_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h128_gqa2w_bf16kv (const R4DArgs* a, hipStream_t stream);
long r4d_attn_decode_h128_gqa2w_scratch_bytes(const R4DArgs* a);
// ...with q at R4DArgs::q_ld (attn_paged_qrow).
int  r4d_attn_decode_h256_gqa2_qld_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa2_qld_bf16kv (const R4DArgs* a, hipStream_t stream);
// The geometry the attention kernels above are compiled for, so a caller can check a model
// before the first launch.
void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows);

// ---- attention: vision encoder -------------------------------------------------------------
// Dense, non-causal, multi-head attention at the vision tower's head_dim of 72, bf16 throughout.
// q, k, v and o are contiguous [total_tokens, heads, 72]; cu_seqlens [num_seqs + 1] bounds one
// image (or attention window) per entry, so a batch is one launch. `max_seqlen` is the host-side
// bound on those lengths and selects the query-block height. head_dim 72 is carried natively
// (5 k-tiles of 16, the last half zero), not padded to 128. Returns -1 for any other head size.
int  r4d_attn_vit_h72_bf16(const void* q, const void* k, const void* v, void* o,
                           const void* cu_seqlens, int num_seqs, int max_seqlen,
                           int heads, int head_dim, float scale, void* stream);
// head_dim, the two query-block heights, and the segment length at which the launcher switches.
void r4d_attn_vit_dims(int* head_dim, int* rows_large, int* rows_small, int* split);
int  r4d_attn_vit_h72_strided_bf16(const void* q, const void* k, const void* v, void* o,
                                   const void* cu_seqlens, int num_seqs, int max_seqlen,
                                   int heads, int head_dim, float scale, int q_ld, int k_ld,
                                   int v_ld, int o_ld, void* stream);

// ---- gated delta net -----------------------------------------------------------------------
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan
// and the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h +
// chunk_fwd_o). Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt must be the compile-time 128/128/64 in the name; a mismatch returns -1. Prefill and
// chunked prefill run this; decode runs the recurrent update.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
// The same scan against a slot-indexed state cache: `ssm_state_indices` [N] names each batch
// row's slot, and the two strides are the cache's, in elements. The call above is this one with a
// null index and packed [N, H, V, K] strides, which is correct only while slot == batch row.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, const void* ssm_state_indices, long indices_stride,
        long state_slot_stride, long state_head_stride,
        int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, i.e.
// FLA's chunk_scaled_dot_kkt_fwd then solve_tril, with the fp32 gram kept out of HBM.
// k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk); A [T,H,64] bf16 out;
// cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// DFlash2's grouped dynamic depthwise convolution in one pass (the reference uses ~6 elementwise
// passes and a materialised coefficient tensor). delta is a slice of the [T, 2, taps, NG] kernel
// projection, so dpitch is 2*taps*NG; out must not alias x. block_size must be a power of two.
int r4d_dflash_conv_t2_g16_bf16(
        const void* x, const void* delta, const void* base, void* out,
        int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream);

// DFlash2's candidate selector: the low-rank edge scores and the greedy path walk, one workgroup
// a sequence. `cand` and `unary` are row_topk's two output planes at row s*steps+l, `hp` is
// hidden_projection(x) for the same row, and `anchor` is the token the target just committed,
// read at anchor[s * anchor_stride]. R must be a multiple of 32 and at most 512; K at most 32.
//
// A null `hp` means ones: the score is the plain bigram <pred[prev], succ[c]>, DSpark's rank-256
// Markov head over the base head's top K; `hp_ld` is then ignored.
//
// `V` is the codebooks' row count. An id outside [0, V) (a merge pad is -1) reads no codebook row:
// the predecessor contributes nothing and the candidate scores its unary term alone, as in
// ref_dflash_select.
int r4d_dflash_select_bf16(
        const void* cand, const void* unary, const void* hp, const void* anchor,
        const void* pred, const void* succ, void* out,
        int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
        int anchor_stride, void* stream);

// Everything between the qkv projection and the chunked scan, in one kernel: the depthwise causal
// convolution (width 4, silu) with its state cache, the q/k/v split, the l2 norm on q and k, the
// gate g = -exp(A_log).softplus(a + dt_bias) with its per-chunk cumsum, and beta = sigmoid(b).
// Replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum. Strides are in elements;
// x may be a padded view (the qkvz split). `hi_i32` is has_init's width: 1 for int32, 0 for
// unsigned char (reading int32 flags as bytes is right only when the value is 0).
int r4d_gdn_conv_prep_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
        const void* has_init, int hi_i32,
        const void* a, const void* b, long ab_stride, int ab_is_bf16,
        const void* A_log, const void* dt_bias, void* q, void* k, void* v, void* g, void* beta,
        const void* cu, int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr,
        void* stream);

// The same convolution for a decode step over a speculative window. The state cache is a rolling
// buffer of width-1 + num_spec entries read at the slot the last accepted token left; q / k / v
// are written straight into their own layouts. Replaces causal_conv1d_update and its cat.
int r4d_gdn_conv_update_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
        long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
        int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream);

// The gated RMS norm on the layer's output: out = rms(x) . w . act(z), one row per (token, head).
// Replaces FLA's rmsnorm_fn. act: 0 = silu/swish, 1 = sigmoid. Prefill only; the decode kernel
// folds the same arithmetic into its epilogue.
int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o,
                                    long rows, long xrow, long zrow, long orow,
                                    long zheads, long ztok, int width,
                                    float eps, int act, void* stream);

// The recurrent delta-rule update decode runs: gating, the qk l2 norm, the state update and the
// output, against the paged state cache. The state is fp32 (per the model config) and one is
// written per candidate token, which dominates the cost. Replaces
// fused_sigmoid_gating_delta_rule_update.
int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
        const void* q, const void* k, const void* v, const void* a, const void* b,
        long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
        long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
        long oq_pitch, long os_ld, const void* cu,
        const void* ssm_state_indices, long indices_stride, const void* num_accepted,
        const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
        int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
        void* stream);

// ---- all-reduce: one-shot, push, 2 ranks over P2P ------------------------------------------
// Each rank pushes its whole input into the peer's IPC scratch, then reduces locally; no ring, so
// the scratch is sized by the full message. Exactly 2 ranks: the handshake is a single peer flag.
// Every 2-rank kernel on one buffer set shares its counters: `seq_ctrs` is 2 * r4d_ar_max_blocks()
// zeroed words, a handshake and a slot-parity counter per block of capacity (r4d_ar_seq.h).
enum { R4D_AR_HANDLE_BYTES = 64 };
// IPC scratch helpers, startup-only: alloc returns the device pointer and writes the IPC handle
// (up to HANDLE_BYTES) into out_handle. A refused export (beside another engine on the same cards)
// leaves it zeroed with *out_len 0: libr4d's ranks are threads of one process and share the pointer.
long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len);
// Strided peer (or local) device-to-device copy through the copy engines; pitches and width in
// bytes, height in rows. Returns 0 or -100 - hipError_t.
int r4d_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream);
int r4d_p2p_fence(long stream);
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream);
void r4d_ar_ipc_free(long p);
void r4d_ar_ipc_enable_peer(long peer);
// Exchange buffers in pinned, fine-grained host memory, for cards that cannot reach each other's
// VRAM. Zeroed; the address is valid on every card and the CPU. Startup-only.
long r4d_ar_host_alloc(long size);
void r4d_ar_host_free(long p);
// Exact sum, fp32 accumulate, bf16 / fp16 / fp32 payload (dtype 0 / 1 / 2).
void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                                long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                                long dtype, long stream, long nblocks, long nthreads, long drain,
                                long acq);
// Same topology, with the wire payload Walsh-Hadamard rotated and quantised to 6 bits per element
// in groups of 64 (plus a bf16 scale per group). Lossy; bf16 / fp16 payload only. loc_pack is this
// rank's own packed copy, so the reduce folds exactly the bytes it sent.
void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq);
// The exact all-reduce of `y` [M][n] bf16 with hc_write as its epilogue: h[r][c*n + i] +=
// inj[r][c] * y[r][i] over the reduced y, byte-identical to all_reduce then hc_write. `irow` and
// `hrow` are the pitches of inj [M][hc] and h [M][hc*n] in elements; y is tight. A non-zero `gate`
// is the engine's link word, ended once the peer's push has landed (r4d_ar_link_landed);
// `gate_ctr` is its block counter. The gather form below takes the same pair.
void r4d_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream, long nblocks, long nthreads,
                     long drain, long acq, long gate, long gate_ctr);
// The same with the MoE gather as its prologue: y is produced, not read, from the routed experts'
// rows `ye` [M*top_k][ye_ld] summed to token order by weights `ew` [M][ew_ld] and `sorted`
// [M*top_k], with the shared arm `sh` [M][sh_ld] and its gate `sg` folded when both are non-zero.
// Byte-identical to moe_gather, all_reduce, hc_write. A non-zero `inv` is M*top_k int32 device
// scratch for the inverse slot table a prefill-sized call reads instead of a per-block LDS one.
// Returns 0, or a negative code for a shape it does not serve.
int  r4d_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                            long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                            long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                            long top_k, long sig, long stream, long nblocks, long nthreads,
                            long drain, long acq, long inv, long gate, long gate_ctr);
// r4d_ar_hc_write and r4d_ar_gather_hc_write on the rotated 6-bit wire (r4d_ar_oneshot_2rank_wht6's
// format, loc_pack and scratch split): byte-identical to all_reduce on that wire then hc_write,
// with `ye` non-zero preceded by moe_gather. Gather operands as r4d_ar_gather_hc_write; its slot
// table is built in LDS, and the grid is widened until a block's table fits. M*n must be a
// multiple of 64. `gate` / `gate_ctr` as r4d_ar_hc_write. Returns 0, or a negative code for a
// shape it does not serve.
int  r4d_ar_hc_write_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                          long seq_ctrs, long loc_pack, long slot_stride_bytes,
                          long scale_off_bytes, long y, long inj, long h, long M, long n, long hc,
                          long irow, long hrow, long ye, long ew, long sorted, long sh, long sg,
                          long ye_ld, long ew_ld, long sh_ld, long sg_ld, long top_k, long sig,
                          long stream, long nblocks, long nthreads, long drain, long acq,
                          long gate, long gate_ctr);
int  r4d_ar_max_blocks(void);                                   // the two 2-rank kernels
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems); // rotated-6-bit payload only

// ---- all-gather: one-shot, push, exactly 2 ranks over P2P ----------------------------------
// The push and per-block flag handshake of r4d_ar_oneshot_2rank_exact, with placement instead of
// reduction: `out` is 2 * n_elem elements, this rank's input at rank * n_elem and the peer's at
// (1 - rank) * n_elem. `dtype` only gives the element size; the payload is never summed, because
// the vocab-parallel sampler gathers (u32 token id, f32 logit) pairs.
void r4d_ar_gather_2rank(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                         long dtype, long rank, long row_elems, long stream, long nblocks,
                         long nthreads, long drain, long acq, long dpitch,
                         long spitch);

// ---- all-reduce: one-shot, push, 4 or 8 ranks over P2P -------------------------------------
// The same algorithm at width (r4d_ar_oneshot_Nrank_exact.hip). Peer arguments are in ascending
// global rank with the current rank removed. Scratch is 2*world-size slots of slot_stride16 16B
// words (parity x source rank, this rank's own slot unused); flags are max_blocks*ws words indexed
// [block][source rank]; seq_ctrs is 2 * max_blocks words, a handshake and a slot-parity counter
// per block of capacity (r4d_ar_seq.h).
// Three ranks is the same kernel at WS=3; the wide family does not need a power of two. The
// two-shot serves 3 too, with shards one word uneven where the message does not divide.
void r4d_ar_oneshot_3rank_exact(long peer_scratch0, long peer_scratch1,
                                long peer_flags0, long peer_flags1,
                                long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                long inp, long out, long n_elem, long dtype, long rank,
                                long stream, long nblocks, long nthreads, long drain, long acq,
                                long pub);
void r4d_ar_oneshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                long inp, long out, long n_elem, long dtype, long rank,
                                long stream, long nblocks, long nthreads, long drain, long acq,
                                long pub, long gate, long gate_ctr);
// The two wide releases' extra pair at four ranks: `gate` is the engine's link word (0 when this
// launch holds nothing -- the host mailbox, a recorded pass, or a width the row does not release),
// `gate_ctr` the block counter beside the sequence counters (r4d_ar_seq.h). The 3- and 8-rank
// entries take neither: their rows answer 1 and the engine releases by barrier.
// ---- all-gather at width (r4d_ar_gather_Nrank.hip) ------------------------------------------
// The 2-rank gather's placement on the wide family's transport: `out` is world * n_elem elements,
// rank r's slice at r * n_elem, or row-interleaved [rows][world][row] when row_elems > 0. Shares
// the wide all-reduce's instance (scratch, [block][source] flags, seq counters).
void r4d_ar_gather_3rank(long peer_scratch0, long peer_scratch1, long peer_flags0,
                         long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                         long slot_stride16, long inp, long out, long n_elem, long dtype,
                         long rank, long row_elems, long stream, long nblocks, long nthreads,
                         long drain, long acq, long pub, long dpitch,
                         long spitch);
void r4d_ar_oneshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub);
// r4d_ar_hc_write at four ranks: the 4-rank exact one-shot of `y` [M][n] bf16, in place, with
// hc_write as its epilogue, byte-identical to that all-reduce then hc_write. Buffers, peer order,
// flag values and slots match r4d_ar_oneshot_4rank_exact, so the two (and the two-shots) may share
// one instance. `irow`/`hrow` as r4d_ar_hc_write; y is tight.
void r4d_ar_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                           long peer_flags0, long peer_flags1, long peer_flags2,
                           long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                           long y, long inj, long h, long M, long n, long hc, long irow,
                           long hrow, long rank, long stream, long nblocks, long nthreads,
                           long drain, long acq, long pub, long gate, long gate_ctr);
// r4d_ar_hc_write_4rank with the MoE gather as its prologue, byte-identical to moe_gather, the
// 4-rank exact all-reduce, then hc_write. Gather operands, `inv` and the grid rule are
// r4d_ar_gather_hc_write's; buffers, peer order, flag values and slots are
// r4d_ar_oneshot_4rank_exact's. Exact wire only.
void r4d_ar_gather_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                  long peer_flags0, long peer_flags1, long peer_flags2,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride16, long y, long inj, long h, long M, long n,
                                  long hc, long irow, long hrow, long ye, long ew, long sorted,
                                  long sh, long sg, long ye_ld, long ew_ld, long sh_ld,
                                  long sg_ld, long top_k, long sig, long rank, long stream,
                                  long nblocks, long nthreads, long drain, long acq, long pub,
                                  long inv, long gate, long gate_ctr);
// The inverse slot table both widths' gather forms read at a prefill-sized call: inv[f] = the
// sorted row holding slot f, -1 where none does, over T = M*top_k int32. One block, on `stream`.
// Returns 0 or a negative / HIP error code.
int  r4d_ar_gather_inv(long inv, long sorted, long T, long stream);
void r4d_ar_wide_dims(int* max_blocks, int* nb_design, int* max_peers);

// ---- all-reduce: two-shot, 3, 4 or 8 ranks over P2P ----------------------------------------
// Reduce-scatter + all-gather on the wide one-shot's buffer trio and peer order
// (r4d_ar_twoshot_Nrank_exact.hip): shard s, a contiguous run of 16-byte words, is reduced by
// rank s (fp32, ascending rank order), then gathered. Sends 2*(ws-1)/ws*N bytes per rank in two
// hops, for sizes where the one-shot's (ws-1)*N send is too heavy. Shards are as even as the word
// count allows (the first n16 % ws take one more word), so the message need not divide by the
// rank count. At 8 ranks `shard_weights` (8 ints, or null for the even split) sizes the shards per
// rank instead (r4d_ar_shards.h); every rank must pass the same weights. Regions A and B sit at 0
// and the longest shard, so the slot stride must hold twice that. `pieces` (8 ranks, 1 to
// R4D_AR_TS_MAX_PIECES, the same on every rank) pipelines each block's words in that many runs so
// phase 1 of one run and phase 2 of the previous share the link; 1 is the unpipelined kernel.
// Above the bound below the caller falls back to the one-shot; it only guards the kernel's int
// word offsets (2^28 elements is 2^26 words at fp32). A lower bound would send large prefill
// chunks (e.g. 8192 tokens at Gemma 4's 2816 width, 23M-46M elements) to the one-shot.
enum { R4D_AR_TWOSHOT_WIDE_MAX_ELEMS = 268435456, R4D_AR_TS_MAX_PIECES = 16 };
void r4d_ar_twoshot_3rank_exact(long peer_scratch0, long peer_scratch1, long peer_flags0,
                                long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub);
void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub, long gate, long gate_ctr);
void r4d_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub,
                                const int* shard_weights, long pieces);
// The tiered int8 wire on both two-shot hops (r4d_ar_twoshot_Nrank_ti8.hip; codec in
// r4d_ar_ti8.h), 3 or 4 ranks. Int8 groups of 32 with an fp16 scale and a per-group record
// (T1 plain, T2 top-1-exact, T3 wht32), 38 bytes/group; tiering is deterministic and
// bit-identical across ranks. Peer order and flag layout as the exact pair; slot stride in bytes
// (the wire is not 16B-aligned), holding two wire images; the encoder writes its own slot too (no
// loc_pack). numel must be whole 32-element groups, split over the ranks as evenly as they divide
// (the first ng % ws shards one group longer), with every slot laid out for the longest shard.
// The 4-rank entry still requires numel % 128 == 0, the even split.
void r4d_ar_twoshot_3rank_ti8(long peer_scratch0, long peer_scratch1, long peer_flags0,
                              long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub);
void r4d_ar_twoshot_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                              long peer_flags0, long peer_flags1, long peer_flags2,
                              long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub, long gate,
                              long gate_ctr);
void r4d_ar_ti8_dims(int* group, int* group_bytes);  // tiered-int8 wire only

// ---- GEMM ----------------------------------------------------------------------------------
// Skinny bf16 GEMM for M up to 64: C[M,N] = A[M,K] @ W[N,K]^T with 16x16x16 WMMA, so a 16-wide K
// step costs ceil(M/16) activation fragments and one weight fragment.
void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream);
int  r4d_gemm_bf16_nt_m64_max_m(void);
// The same GEMM at prefill row counts: LDS-staged tiles, eight waves a block, output bytes
// identical to r4d_gemm_bf16_nt_m64 at the same split count SK. `tile` picks the block: 1 = 128 x
// 128, 2 = 64 x 64; r4d_gemm_bf16_nt_tiled_arm is the automatic choice. K a multiple of 64.
// Returns 0, negative on a shape it does not serve, or the launch's HIP error.
int  r4d_gemm_bf16_nt_tiled(long a, long w, long c, int M, int K, int N, int SK, int tile,
                            long stream);
int  r4d_gemm_bf16_nt_tiled_arm(int M, int N);

// Skinny GEMM with a 4-bit weight: C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C. The weight
// is asymmetric per output channel per group of `group()` contiguous K -- w ~= scale * (q - zero),
// q in 0..15, zero an integer -- and pre-permuted offline into WMMA fragment order, so the weight
// path is one global_load_b128 per lane per four k steps. Wq is N*K/2 bytes; Wsz is one dword per
// (row, group): the f16 scale low, the f16 of -(1024 + zero) high. N must be a multiple of 16 and
// K divisible by SK * group().
void r4d_gemm_w4a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w4a16_nt_m64_max_m(void);
int  r4d_gemm_w4a16_nt_m64_group(void);

// Skinny GEMM with an 8-bit weight: same fragment order, packed-block size and (scale, zero) dword
// as r4d_gemm_w4a16_nt_m64, with one packed block covering 32 K instead of 64. q in 0..255. Wq is
// N*K bytes.
void r4d_gemm_w8a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w8a16_nt_m64_max_m(void);
int  r4d_gemm_w8a16_nt_m64_group(void);


// Exact per-row top-R of a wide bf16 matrix (the row_topk op, r4d_rowtopk_bf16.hip): the number
// of chunks a launch over N columns uses for a requested chunk count NCH. The chunk width is
// rounded up to a multiple of eight columns, and the stage-1 partial plane is M * chunks * RC
// int64, so a scratch sizer asks this rather than recomputing it.
int  r4d_rowtopk_bf16_chunks(int N, int NCH);

// 2-bit weight, 8-bit activation, for the DFlash2 draft head. Unsigned 2-bit codes in the byte's
// high two bits, asymmetric per (row, group of group()) K, corrected with r4d_quant_act_i8_asum's
// [K/group][Mpad] group-major sums: the activation, quantiser and layout of
// r4d_gemm_w4a8_asym_nt_m64. Wq is N*K/4 bytes; Ws is the shared (f16 scale, f16 of
// -(1024+zero)) dword per (row, group).
void r4d_gemm_w2a8_nt(long a, long ascale, long wq, long ws, long asum, long c,
                      int M, int K, int N, int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w2a8_nt_group(void);

// 4-bit weight, 8-bit activation. Signed 4-bit codes, per-row activation scale, int8 WMMA.
void r4d_gemm_w4a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream);
// Asymmetric variant: the WMMA reads the weight nibble unsigned and the stored zero point is
// corrected with `asum` from r4d_quant_act_i8_asum. Same packed weight and scale dword; the zero
// is the f16 of -(1024 + zero) in its high half, the f16 kernel's encoding.
void r4d_gemm_w4a8_asym_nt_m64(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               long stream);
// The same GEMM tiled for prefill batches, where the skinny kernel's 64-row block re-reads the
// weight M/64 times. Same packed weight and pre-shuffled activation; N must be a multiple of 256
// and K of the scale group. `variant` is one of the kernel's compiled tile arms (r4d_rows.cpp,
// kTileArms); 0 is the shipped default.
void r4d_gemm_w4a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant = 0);
// Same kernel on the asymmetric grid: unsigned nibbles plus r4d_quant_act_i8_asum's [G][Mpad] row
// sums.
void r4d_gemm_w4a8_asym_tiled(long a, long ascale, long wq, long ws, long asum, long c, int M,
                              int K, int N, long stream, int variant = 0);
// The same GEMM for prefill row counts where the matrix pipe, not the weight stream, is the limit.
// Same packed weight and rotated int8 activation as the tiled kernel; N a multiple of BN (256 at
// the default shape), K of the scale group. The activation is stored half-major in LDS so one
// `ds_load_b128` fetches two k steps, and the group fold is plain `v_fmac_f32` so VOPD can pair it.
void r4d_gemm_w4a8_prefill(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                           long stream, int variant = 0);
int  r4d_gemm_w4a8_nt_m64_max_m(void);
int  r4d_gemm_w4a8_nt_m64_group(void);

// 8-bit weight, 8-bit activation: the promoted linears of the INT4 target (every down_proj and the
// attention o_proj) and lm_head. Signed int8 codes in the iu8 16x16x16 fragment order, the order
// the A side uses, with one f16 scale per (output row, group() of K) in an f16-only plane; the
// per-row activation scale is applied at the store. `a` is the same permuted int8 buffer
// r4d_gemm_w4a8_nt_m64 reads. Wq is N*K bytes; Ws is N*K/group() f16.
void r4d_gemm_w8a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w8a8_nt_m64_max_m(void);
int  r4d_gemm_w8a8_nt_m64_group(void);
int  r4d_gemm_w8a8_nt_m64_kpb(void);
// The same 8-bit GEMM tiled for prefill, where the skinny kernel's MB*16-row block re-reads the
// weight M/(MB*16) times. Same packed weight and permuted activation; N a multiple of the variant's
// BN (256 at the default shape), K of the scale group. Built on r4d_gemm_w4a8_tiled's "unpack at
// staging" path, whose LDS buffer already holds bytes.
void r4d_gemm_w8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant = 0);

// OCP-MXFP4 weight, fp8 activation. e2m1 elements with one E8M0 exponent per 32 K, folded
// against a per-row reference exponent so the inner loop has no rescale; fp8 WMMA.
// as_group 0 = one f32 activation scale per row, 128 = one per (row, 128 of K) at row stride as_rs.
void r4d_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long wref, long c,
                             int M, int K, int N, int WV, int SK, int MB, int NPW,
                             int as_group, int as_rs, long stream,
                             int lda = 0, int ldws = 0, int ldc = 0);
int  r4d_gemm_mxfp4a8_nt_m64_max_m(void);
int  r4d_gemm_mxfp4a8_nt_m64_group(void);
// The same weight tiled for prefill: a 256-row block reads A fragments from a fragment-ordered copy
// of the activation (`at`, written by r4d_mxfp4a8_tile_a, at least r4d_mxfp4a8_tile_a_bytes long)
// and unpacks the weight once per K slab. TN (2 or 4, 0 = by M) is the per-wave column width in
// 16-column fragments. as_group as r4d_gemm_mxfp4a8_nt_m64.
void r4d_gemm_mxfp4a8_tiled(long at, long ascale, long wq, long ws, long wref, long c,
                            int M, int K, int N, int TN, int as_group, int as_rs, long stream,
                            int ldws = 0, int ldc = 0);
// `lda` is the activation's row pitch in bytes (0 = K), for a window of a wider plane.
void r4d_mxfp4a8_tile_a(long a, long at, int M, int K, long stream, int lda = 0);
long r4d_mxfp4a8_tile_a_bytes(int M, int K);
int  r4d_gemm_mxfp4a8_tiled_group(void);
// The decode band (M <= 128) is r4d_rad_gemm_mxfp4a8_decode only: its split-K partial is the op's
// scratch, which only the plugin ABI provides.
int  r4d_gemm_mxfp4a8_decode_max_m(void);

// ---- Block-scaled FP8, as checkpoints ship it -------------------------------------------------
//
// Reads the weight as safetensors stores it: `<linear>.weight` F8_E4M3 row-major [N][K] and
// `<linear>.weight_scale_inv` BF16 on a [ceil(N/128)][ceil(K/128)] grid (DeepSeek-style,
// `weight_block_size = [128, 128]`). No permutation and no scale conversion.
//
// The block is 128 on both axes and is a property of the file, not a build flag (unlike
// R4D_GEMM_W4A8_GROUP). The `_block()` queries report it, and every entry point rejects a K that
// would leave a partial block.
//
// Three kernels, one per band. The two W8A8 ones put the weight on the WMMA A operand, so a lane's
// accumulator slots share one activation scale; see r4d_gemm_fp8a8.hip.
//
//   M == 1    r4d_gemm_fp8a16_nt_m1   bf16 activation read directly, no quantise launch
//   M <= 64   r4d_gemm_fp8a8_nt_m16   E4M3 activation, up to four 16-wide token tiles, split-K
//   any M     r4d_gemm_fp8a8_tiled    E4M3 activation, 128x128 tile, eight waves; the weight read
//                                     into registers, the activation staged through LDS
//
// `w_ld` is bytes per weight row and `sw_ld` elements per scale row, 0 meaning tight. All return 0
// or a negative code instead of throwing, since they are reachable from C consumers.
int  r4d_gemm_fp8a16_nt_m1(long a, long wq, long ws, long c,
                           int M, int K, int N, int w_ld, int sw_ld, int rb, long stream);
// The same GEMM for M <= r4d_gemm_fp8a16_nt_m1_max_m(): M token rows against each weight row, so
// the weight is read once. `a` is [M][K] with `a_ld` elements a row and `c` is [M][N] with `c_ld`
// (0 = N). This is the speculative-verify width; the 16-wide fp8a8 tile needs a quantise launch
// first and reaches a lower fraction of the memory ceiling on the same weight traffic.
int  r4d_gemm_fp8a16_nt_mt(long a, long wq, long ws, long c, int M, int K, int N,
                           int a_ld, int c_ld, int w_ld, int sw_ld, long stream);
/* The lm_head at fp8, scales interleaved in each row's tail; see the .hip for why that layout. */
int  r4d_logits_gemm_fp8(long a, long wq, long c, int M, int K, int N,
                         int a_ld, int c_ld, int w_ld, long stream);
int  r4d_gemm_fp8a16_nt_m1_block(void);
int  r4d_gemm_fp8a16_nt_m1_max_m(void);

// `a` is E4M3 [M][K] with `ascale` f32 [M][ceil(K/128)], r4d_quant_act_fp8's output. `scratch` is
// the split-K partial plane; 0 runs the launch unsplit (correct, slower). Size it with the query,
// which returns 0 when this shape will not split.
//
// Split-K is not bit-comparable across settings: ksplit repartitions K, so the accumulation order
// differs. Two runs of the same binary still agree.
int  r4d_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                           int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                           long scratch, long scratch_bytes, long stream);
long r4d_gemm_fp8a8_nt_m16_scratch(int M, int K, int N);
int  r4d_gemm_fp8a8_nt_m16_max_m(void);
// Test hook: pins the narrow tile's wave count, normally a launch heuristic over (N, M), for every
// subsequent call on every thread; 0 restores the rule. Not for serving; r4d_selftest holds it for
// one comparison. Unlike the split, the wave count is bit-comparable: it moves which wave owns an
// output element, not the order of the K tiles it accumulates, and the split derives from
// `gx * mw`, which the rule keeps invariant. r4d_selftest compares bytes at every legal count.
void r4d_gemm_fp8a8_pin_mw(int mw);
// Test hook: pins the QSA score's form. 0 is the launcher's rule, 1 the scalar decode form, 2 the
// matrix form (which refuses a shape it is not built for). The two sum in different orders, so
// r4d_selftest --perfqsa compares them to a tolerance.
void r4d_qsa_score_pin_form(int form);
// Test hook: pins the QSA selection's form. 0 is the launcher's rule, 1 the single-workgroup
// kernel, 2 the wide-launch (three-pass) kernel; the split path keeps its own rule. The pinned
// forms are bit-identical, and r4d_selftest compares them.
// Test hook: pins the routed-expert GEMM's form. 0 is the launcher's rule, 1 the run-aligned
// prefill form, 2 the (tile, run) decode form, 3 the uniform prefill grid. r4d_selftest --perfmoe
// times each and compares it with form 1.
void r4d_moe_gemm_pin_prefill(int form);
// Test hook: pins the decode form's staged K. 0 is the launcher's row limit (kMoeGuFk256Rows),
// 1 one 128-wide block a pass, 2 two wherever the shape allows (the token-operand projection,
// every class's K a multiple of 256, a format that folds scale groups). Bit-identical.
void r4d_moe_gemm_pin_fk(int pin);
// Test hook: pins the run-aligned form's column tile. 0 is the launcher's rule (the down
// projection's 256-column tile wherever the sorted operand, one part, a w4 weight and every
// class's width allow it), 1 never that tile, 2 always (refused where the shape cannot take it or
// the launch is not the aligned form). Bit-identical.
void r4d_moe_gemm_pin_down_tile(int pin);
// Test hook: pins the gather's form. 0 is the launcher's rule, 1 a column a thread
// (gather_kernel), 2 a wave a token and sixteen bytes a lane (gather_wide_kernel), refused on a
// shape it cannot take. Bit-identical.
void r4d_moe_gather_pin_form(int form);
// Test hook: pins the sorted operand's decode kernel. 0 is the launcher's rule
// (moe_down_dec_kernel on the sorted operand, one part, at most 512 experts, column tiles set by
// the routed rows -- kMoeDownDecG2Rows), -1 the (tile, run) form, n > 0 moe_down_dec_kernel at n
// column tiles a block (capped at 16, refused on a shape it does not serve). Applies inside the
// decode form only, which r4d_moe_gemm_pin_prefill(2) pins. Bit-identical.
void r4d_moe_gemm_pin_down_dec(int pin);
// Test hook: pins the gated residual read's prefill tile. 0 is the launcher's rule, 1 the decode
// tiles' kernels at 128 rows (hc_down_tiled, then hc_up_tiled into `ubuf` and hc_mix), 2 the wide
// kernels (refusing shapes they are not built for), 3 the PF2 kernels the rule takes from 512 rows
// (refused likewise, and below 512 rows, whose scratch is not sized for their weight planes).
// All bit-identical; r4d_selftest compares their bytes.
void r4d_hc_read_pin_prefill(int form);
// Test hook: pins the MLA matrix row's key load. 0 is the launcher's rule (sixteen-byte vectors
// wherever the cache rows are aligned to them), 1 the element-wise load the rule falls back to.
// Bit-identical; r4d_selftest compares them and --perfmla times both.
void r4d_mla_matrix_pin_form(int form);
// Test hook: records the function each launch dispatches into `fns[0..cap)`, in order, until the
// next call, and returns how many launches the previous recording saw (which may exceed its cap).
// Null `fns` stops recording. r4d_selftest uses it to show that a pinned form dispatches a kernel
// of its own, so a pin that is ignored cannot pass by comparing a form with itself.
int r4d_record_dispatches(const void** fns, int cap);
// Test hook for the PLE row cache's seqlock (r4d_model_bf16.hip): `ctl` is a device buffer of
// r4d_ple_probe_words uint32 words, or null to take the probe away. Null is every launch outside
// a test: the gathers then allocate nothing, add no word of state and write the same bytes.
// With word 0 set, a reader that found its row parks in its window, before the payload load that
// follows, and a block that finds it parked reclaims that way, writes a row that is not the row's,
// and publishes, so the load reads that rewrite and the recheck after it is forced to fall on a
// generation two on. Word 1 counts those rechecks; words 2.. carry the parking handshake. See the
// words below.
enum {
    r4d_ple_probe_mode      = 0,  // 0 off, 1 force the reclaim/rewrite window
    r4d_ple_probe_conflicts = 1,  // rechecks that found the way rewritten for the same id
    r4d_ple_probe_parked    = 2,  // a reader is parked, and the way's slot, generation and id
    r4d_ple_probe_slot      = 3,
    r4d_ple_probe_gen       = 4,
    r4d_ple_probe_id        = 5,
    r4d_ple_probe_rewrites  = 6,  // ways rewritten under a parked reader
    r4d_ple_probe_claim     = 7,  // one parked reader a run
    r4d_ple_probe_serve     = 8,  // one block a run serves that park; a second would publish again
    r4d_ple_probe_words     = 9
};
int r4d_ple_probe(void* instance, unsigned* ctl);
// The narrow GEMM with r4d_gated_quant_fp8 as its epilogue, for a gate_up projection whose only
// consumer is that quantiser. `wq` is the [2*nout][K] gate_up plane, gate first; outputs are the
// quantiser's: `q` E4M3 [M][nout], `s` f32 [M][ceil(nout/128)], and `y` the bf16 silu product or 0.
//
// Byte-identical to the unfused pair: the accumulator is rounded to bf16 first and the gate,
// product and amax are taken over the rounded values. `act` is r4d_gated_quant_fp8's; only 0
// (silu) is compiled.
//
// Refuses, without fallback, any shape the fold does not fit: M past the band, K not a multiple of
// 256, `nout` not a multiple of 128 (a block emits exactly one 128-column scale group), or a grid
// the split rule would widen.
int  r4d_gemm_fp8a8_gated_nt_m16(long a, long ascale, long wq, long ws,
                                 long q, long s, long y, int act,
                                 int M, int K, int nout,
                                 int a_ld, int sa_ld, int w_ld, int sw_ld,
                                 int q_ld, int s_ld, int y_ld, long stream);
int  r4d_gemm_fp8a8_gated_nt_m16_max_m(void);
// The narrow GEMM with a bias: `bias` bf16 [N], added to each output column's f32 accumulator before
// its one rounding, and C written at row pitch `c_ld` (0 = N), so C may be a column slice of a wider
// buffer. Never split. M up to r4d_gemm_fp8a8_nt_m16_max_m(); K a multiple of 128, N of 16.
int  r4d_gemm_fp8a8_bias_nt_m16(long a, long ascale, long wq, long ws, long bias, long c,
                                int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                int c_ld, long stream);
int  r4d_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long stream);
// The same two GEMMs at int8 codes for both operands (r4d_gemm_fp8a8.hip's I8 instantiations):
// the activation i8 with an f32 scale per (row, 128 K), the weight i8 in the same fragment order
// with a bf16 scale per row per 128 K, tile-major: ws is [N/16][ceil(K/128)][16] (written by
// r4d_fp8_layout.cpp's i8 hooks); sw_ld is not read.
int  r4d_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long scratch, long scratch_bytes, long stream);
// The lm_head at int8 codes: r4d_logits_gemm_fp8's row layout (codes, the row's bf16 scales in its
// tail, the pad) with i8 codes and a scale per 128 columns.
int  r4d_logits_gemm_i8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                        int w_ld, long stream);
int  r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                         int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                         long stream);
int  r4d_gemm_fp8a8_block(void);

// Per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation, read by the fp8a8 GEMMs.
// Plain row-major output, unlike r4d_quant_act_i8's fragment order: the fp8a8 GEMMs stage both
// operands through the same byte copy and swizzle. The scale is amax/448, a dequant multiplier
// like the checkpoint's weight_scale_inv, so the GEMM multiplies both scales and never divides.
int  r4d_quant_act_fp8(long a, long q, long s, int M, int K,
                       int x_ld, int q_ld, int s_ld, long stream);
int  r4d_quant_act_fp8_group(void);

// The same quantiser fused into the pass that produced the row. Both write the bf16 output the
// unfused producer wrote and the E4M3 pair, quantising the bf16-rounded value so the codes match
// the unfused pair (see r4d_fused_quant_fp8.hip). `y` may be null where nothing reads the bf16.
// `w_f32` gives the norm gain's width; `mode` is 0 for silu and 1 for sigmoid.
// A non-null `res` is the fused_add_rms_norm form: `res += x`, rounded to bf16 and stored, then
// normalised, bit-identical to the `add` + `rmsnorm` pair. It may alias `y`.
int  r4d_rmsnorm_quant_fp8(long x, long res, long w, long y, long q, long s, int M, int n,
                           float eps, float wadd, int w_f32, int x_ld, int r_ld, int y_ld,
                           int q_ld, int s_ld, long stream);
int  r4d_gated_quant_fp8(long gu, long y, long q, long s, int M, int n, int mode,
                         int gu_ld, int y_ld, int q_ld, int s_ld, long stream);
// The two above with had_quant_act_i8g's quantiser on the pass instead (the `_had_quant_i8g`
// forms): the width-128 Hadamard, then int8 codes at amax / 127 per (row, 128), for a weight at
// i8a8h. `q` is int8, `n` a multiple of 128, and `y` keeps the unrotated bf16 value. The codes
// are the producer + had_quant_act_i8g pair's bytes.
int  r4d_rmsnorm_had_quant_i8g(long x, long res, long w, long y, long q, long s, int M, int n,
                               float eps, float wadd, int w_f32, int x_ld, int r_ld, int y_ld,
                               int q_ld, int s_ld, long stream);
int  r4d_gated_had_quant_i8g(long gu, long y, long q, long s, int M, int n, int mode,
                             int gu_ld, int y_ld, int q_ld, int s_ld, long stream);
// The activation is rounded to bf16 before the product, matching the unfused `sigmoid` then `mul`
// pair with a bf16 buffer between them.
// The same epilogue with the gate in a different buffer from what it scales, as in gated
// attention: `x` is the attention output and the gate a column slice of the [q|gate] projection.
// A row is then a (token, head) pair; both sources are a fixed stride apart and the scale plane
// lands where the out projection reads it. `g` is passed already offset to the gate half.
int  r4d_gate_quant_fp8(long g, long x, long y, long q, long s, int M, int n, int mode,
                        int g_ld, int x_ld, int y_ld, int q_ld, int s_ld, long stream);

// The same norm with the preceding tensor-parallel all-reduce folded in. Every row-parallel
// linear's output is all-reduced then normalised (128 adjacent dispatches per decode step); fused,
// the all-reduce's bf16 output is never read back and the peer's contribution is a second operand
// on the norm's load. IPC arguments are r4d_ar_oneshot_2rank_exact's; `nblocks` is clamped to M
// because a block owns whole rows, and `x` takes no pitch because it is also the message. Like the
// int8 twin r4d_ar_ln_had_quant_i8 it has two wires: `wire` 0 is the exact bf16 payload, 6 the
// rotated 6-bit one. Wire 6 needs `n` to be a multiple of r4d_ar_rmsnorm_quant_fp8_chunk_elems()
// so a chunk never straddles a row.
int  r4d_ar_rmsnorm_quant_fp8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                              long seq_ctrs, long slot_stride16,
                              long x, long res, long w, long y, long q, long s,
                              int M, int n, float eps, float wadd, int w_f32,
                              int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                              int nblocks, int nthreads, int drain, int acq, int wire,
                              long loc_pack, long loc_scale_off, long wp = 0);
int  r4d_ar_rmsnorm_quant_fp8_chunk_elems(void);
// Its rotated int8 form, as r4d_rmsnorm_had_quant_i8g: the reduction and the norm unchanged, the
// norm's pass quantised by had_quant_act_i8g's rule. No post-norm.
int  r4d_ar_rmsnorm_had_quant_i8g(long peer_scratch, long my_scratch, long peer_flags,
                                  long my_flags, long seq_ctrs, long slot_stride16,
                                  long x, long res, long w, long y, long q, long s,
                                  int M, int n, float eps, float wadd, int w_f32,
                                  int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                  int nblocks, int nthreads, int drain, int acq, int wire,
                                  long loc_pack, long loc_scale_off);

// The same IPC with the sandwich-norm junction after it (r4d_rmsnorm_quant_i8.hip): post-norm of
// the reduced `x` (`wp`), the residual add, the [1] scalar (`sc`), the norm (`w`) and the w4a8
// GEMM's per-row int8 quantiser. Exact wire only; `x` is the message and takes no pitch.
int  r4d_ar_rmsnorm_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                             long seq_ctrs, long slot_stride16, long x, long res, long w, long wp,
                             long sc, int sc_f32, long y, long q, long s, int M, int n, float eps,
                             float wadd, int w_f32, int r_ld, int y_ld, long stream, int nblocks,
                             int drain, int acq);
// ...and at four ranks, on the wide family's transport: r4d_ar_oneshot_4rank_exact's IPC
// arguments (peers in ascending rank, this rank's own, the counters, the slot stride), `pub`, the
// engine's link gate with the instance's landed counter (0 for none; r4d_ar_seq.h), and, with
// `tail_nbt` blocks (positive), the standalone two-shot on that many with the junction as its tail,
// counting on the two words at `tail_ctr` (ar_rmsnorm_quant_i8_tail_kernel).
int  r4d_ar_rmsnorm_quant_i8_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                   long peer_flags0, long peer_flags1, long peer_flags2,
                                   long my_scratch, long my_flags, long seq_ctrs,
                                   long slot_stride16, long x, long res, long w, long wp, long sc,
                                   int sc_f32, long y, long q, long s, int M, int n, float eps,
                                   float wadd, int w_f32, int r_ld, int y_ld, int rank,
                                   long stream, int nblocks, int drain, int acq, int pub,
                                   long gate, long gate_ctr, long tail_ctr, int tail_nbt);

// ---- sampling ----------------------------------------------------------------------------------
//
// One workgroup a position: temperature, softmax, top_k, top_p, min_p, renormalise and the draw,
// in vLLM's order (reordering changes which tokens survive). Three outputs for a greedy-draft
// speculative verify: a draw, a draw with the proposed token removed and the rest renormalised,
// and the proposal's filtered probability, which is the rejection-sampling accept probability.
struct R4DSpecDraw {
    unsigned int  tok;        // a draw from the filtered distribution
    unsigned int  tok_excl;   // a draw with `query` removed and the remainder renormalised
    float         p_query;    // filtered probability of `query` -- the accept probability
};
// `logits[pos*col_stride + v*row_stride]`, so either head layout is read without a transpose.
// `query` may be 0; `u` is two uniforms a position. `n_vocab` is the full vocabulary: a
// vocab-parallel rank gathers the other bands first, since a nucleus needs the whole distribution.
int  r4d_sample_chain_f32(long logits, long query, long u, long out,
                          int M, int n_vocab, int row_stride, int col_stride,
                          float temperature, float top_p, int top_k, float min_p, long stream);

// ---------------------------------------------------------------------------- the quantisers
//
// These take the plugin ABI's arguments: `int r4d_rad_<name>(const RadArgs*, RadStream)`, declared
// in libr4d/r4d_plugin.h because the row table is the only caller and that header is host-only.
// The entry point named under each note is the symbol. See libr4d/r4d_args.h for why this is
// libr4d's only calling convention.

// Per-row symmetric int8 quantisation of a bf16 activation, in the A-fragment byte order. The
// optional fourth operand asks for the per-row per-group sum of the int8 codes an asymmetric
// weight grid needs -- [K/group][Mpad] float, group-major; see r4d_gemm_w4a8_asym_nt_m64.
//   r4d_rad_quant_act_i8
// Block Hadamard of width `had` along K, then the same per-row int8 quantisation. `had` must be a
// power of two dividing K; a rotation block cannot cross a tensor-parallel shard, so it must
// divide the per-rank K (5120 / 3072 / 8704 here, which rules 1024 out).
//   r4d_rad_had_quant_act_i8
// The residual add, the RMS norm, the same block Hadamard and int8 quantiser in one pass.
// `residual` and `out_bf16` may be null: pass a residual for vLLM's fused_add_rms_norm form, and
// out_bf16 only on layers with a bf16 consumer of the post-norm tensor (a GDN block's in_proj_a /
// in_proj_b, which stay bf16 and read it unrotated).
// `wadd` is added to every gain element in fp32 after the bf16 load: GemmaRMSNorm (Qwen3.5,
// Qwen3-Next, Gemma) stores a zero-centred gamma and uses `1 + w`, and forming that in bf16 costs
// 0.3% per channel. Pass 0.0f for a plain RMS norm.
//   r4d_rad_rmsnorm_had_quant_i8
// The same norm with the preceding all-reduce folded in, as in r4d_ar_rmsnorm_quant_fp8: IPC
// arguments, double-buffered scratch, per-block flags and device-resident sequence counters
// are r4d_ar_oneshot_2rank_exact's; `nblocks` is clamped to M because a block owns whole rows.
void r4d_ar_ln_wire_dims(int* stage_bits, int* chunk_elems, int* chunk_bytes,
                         int* scale_bytes);
// The fewest blocks at which [M, K] fits the kernel's LDS at this width and wire (`nthreads` 0
// means the norm's own width), or -1 when no block count does. A block count below it is one the
// launcher refuses.
int  r4d_ar_ln_blocks_min(int M, int K, int had, int nthreads, int wire);
void r4d_ar_ln_had_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16,
                            long x, long residual, long wgain, long out_bf16, long q, long s,
                            int M, int K, int had, float eps, float wadd,
                            long stream, int nblocks, int nthreads, int drain, int acq,
                            int wire = 0);
// An elementwise gated product, then the same Hadamard and int8 quantiser. Mode 0 is silu(a)*b
// (down_proj's input, vLLM's silu_and_mul); mode 1 is sigmoid(a)*b (o_proj's input, since
// Qwen3_5Attention computes attn_output * sigmoid(gate)). Strides are in elements and may be 0 for
// a tight K-pitch; pass 2*K for vLLM's fused gate_up layout where `b` is `a + K`.
//   r4d_rad_gated_had_quant_i8
// The gated-attention QKV preamble: split [q|gate] per head, QK RMS norm, partial NeoX RoPE and
// the gate copy, in one pass. The norm output is rounded to bf16 before the rotation, because the
// unfused reference stores it and reads it back. `wadd` is the GemmaRMSNorm `1 +` (see
// r4d_rmsnorm_had_quant_i8). Pitches are in elements.
//   r4d_rad_qk_norm_rope_gate
// Fills the cos/sin table the fused kernel above reads, at each position's own row of an
// [max_ctx, rot] plane: `rot` values a row, cos for the first rot/2 and sin for the second. One
// launch a step serves every layer. The arithmetic matches r4d_model_bf16.hip's rope_kernel line
// for line, so the fusion is bit-identical to the unfused norm + rope pair.
//   r4d_rad_rope_table
// The GDN block's gated RMS norm (x * rsqrt(mean_head(x^2)+eps) * w * silu(z) over 128-wide heads),
// then the same Hadamard and int8 quantiser over the flattened row: out_proj's input on the
// linear-attention layers. `wgain` is 128 floats; strides are in elements and may be 0.
// `zgrp`/`zblk` let Z be the interleaved qkvz projection instead of a compacted gate: element i of
// the flattened row is at (i / zblk) * zgrp + (i % zblk). Pass 0 for a contiguous gate.
//   r4d_rad_gdn_gated_norm_hq_i8
// The packed 4-bit weight back to plain row-major bf16 [N, K], undoing the fragment order. For
// prefill, where W4A8 gains nothing: dequantise a layer into scratch and use the bf16 GEMM. Serves
// both grids; the zero comes from the scale dword's high half (the constant 8 for symz).
// `twos` selects the encoding: 1 for the symmetric grid's two's-complement code (value =
// (n^8)-8, what radiance_w4.pack writes), 0 for the asymmetric grid's unsigned nibble minus a
// per-group zero. They differ by bit 3; the wrong one agrees for n<8 and is off by 16 above.
void r4d_dequant_w4_bf16(long wq, long ws, long out, int N, int K, int group, int twos,
                         long stream);

// ---- the small bf16 model ops (r4d_model_bf16.hip) ---------------------------------------------
//
// The embedding gather, norm, adds and rotary between the larger kernels, on the device because
// the buffers are in VRAM and the engine refuses an op with no device launch. The arithmetic is
// libref's, the oracle they are compared against: every reduction in f32, only the store narrows,
// round-to-nearest-even.
//
// Every pitch is in elements and 0 means tight. Index arrays are i32. A negative or out-of-range
// index writes a zero row or skips the store: a kernel cannot return a per-row error, and a
// defined answer beats a fault, which is why the bounds are arguments.
// `vocab_off` is the first global row this rank holds: the table is a vocab-parallel slice and an
// id outside it writes zeros, so summing the ranks reconstructs the embedding. 0 means the whole
// table.
int r4d_embed_lookup_bf16(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                          int vocab_off, int x_ld, long stream);
// The same lookup with a runtime scale on found rows (x = table[id] * scale); 1.0 is the plain
// copy, and a missed row still writes zero. Gemma 4 scales its embedding by sqrt(n_embd) at the
// lookup rather than in the table.
int r4d_embed_lookup_bf16_scaled(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                 int vocab_off, int x_ld, float scale, long stream);
int r4d_gather_rows_bf16(long x, long idx, long y, int M, int n, int src_rows,
                         int x_ld, int y_ld, long stream);
// y = x * rsqrt(mean(x^2) + eps) * (w + wadd). `wadd` is GemmaRMSNorm's `1 +`, added in f32 after
// the bf16 load (see r4d_rmsnorm_had_quant_i8). `w_f32` says whether the gain is f32 (as every
// architecture plugin declares it) or bf16; the activation is bf16 either way.
int r4d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd, int w_f32,
                     int x_ld, int y_ld, long stream);
// y = x * rsqrt(mean(x^2) + eps): rmsnorm without the gain multiply, Gemma's value normalisation.
int r4d_rmsnorm_p_bf16(long x, long y, int M, int n, float eps, int x_ld, int y_ld, long stream);
// `op` 0 sums, 1 multiplies. A `b_ld` of 0 broadcasts one row of n elements to every row (a bias
// add).
int r4d_binary_bf16(long a, long b, long y, int M, int n, int op,
                    int a_ld, int b_ld, int y_ld, long stream);
int r4d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r4d_fill_bf16(long y, int M, int n, int y_ld, float value, long stream);
int r4d_fill_f32(long y, int M, int n, int y_ld, float value, long stream);
// y[.., i] = silu(gu[.., i]) * gu[.., n + i]. The gate is the first half, the order rad-convert
// fuses gate_proj and up_proj in.
int r4d_silu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream);
// y[.., i] = gelu_tanh(gu[.., i]) * gu[.., n + i]. The gate is the first half; the tanh
// approximation (Gemma's gelu_pytorch_tanh), not erf.
int r4d_gelu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream);
// In place over `n_rope_heads` consecutive heads of a row; later heads (the V heads of a fused
// projection) are untouched. `gptj` 0 pairs i with i + rot/2 (NeoX, and mrope when every position
// component is equal); 1 pairs 2i with 2i+1. `rot` may be less than head_dim: the frequency
// exponent divides by `rot`, so the rotated prefix is a complete rotary of its own width.
int r4d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int gptj,
                  float theta, float rscale, int row_pitch, long stream);
// The multi-component rotary (docs/OPS.md, `rope`): `mode` 2 mrope, 3 imrope, 4 axial over
// `sec[0..n_sec)`, component c of token m at pos[c * plane + m]. plane 0 is the call above.
// `den` is the width the base ladder's exponent divides by: `rot` (the `rotary` ladder, the
// default) or `head_dim` (the `head` ladder, the proportional variant). The `rope` row resolves it
// from its `ladder` param; the plain form above passes `rot`.
int r4d_rope_mc_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int den,
                      int gptj, float theta, float rscale, int row_pitch, int mode, const int* sec,
                      int n_sec, int plane, long stream);
// C[M,N] f32 = A[M,K] bf16 @ W[N,K]^T bf16, f32 accumulate. The output is f32 because the
// engine's logits buffer is: a bf16 logit has eight mantissa bits, too coarse for a nucleus cut
// that compares neighbouring probabilities. K must be a multiple of 8.
int r4d_logits_gemm_bf16(long a, long w, long c, int M, int K, int N, int a_ld, int c_ld,
                         long stream);
// Scatter into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim]: K at [.., d], V at
// [.., head_dim + d], slot decomposing as block = slot / block_size, offset = slot % block_size.
int r4d_kv_store_bf16(long k, long v, long slot, long kv, int M, int head_dim, int kv_heads,
                      int block_size, int n_blocks, int k_ld, int v_ld, long stream);
// The same scatter into an E4M3 cache, read by the fp8kv attention family. `k_scale` and `v_scale`
// are the descales the reader multiplies by, so this divides by them; the read side has one per
// (sequence, head), hence a caller constant here. 1.0 is vLLM's default and what the attention
// kernels assume when their descale pointers are null. head_dim must be a multiple of 4.
int r4d_kv_store_fp8(long k, long v, long slot, long kv, int M, int head_dim, int kv_heads,
                     int block_size, int n_blocks, int k_ld, int v_ld,
                     float k_scale, float v_scale, long stream);

// ---- the decomposed sampler chain (r4d_sample_stages_f32.hip) -----------------------------------
//
// docs/OPS.md's eleven ops, one entry point each, beside r4d_sample_chain_f32's fused form. The
// fused one suits a fixed order and a speculative verify; these let an engine compose the chain
// per request and add the stages the fused one lacks: penalties, DRY, a grammar bitmask,
// locally-typical, XTC.
//
// `params` is an array of abi/rad_sample.h's RadSampleParams, one row per sampled position; the
// header is included, not restated. Logits are f32. A dropped candidate is index -1, value
// -infinity, and the buffers never shrink.
//
// The entry points are r4d_rad_sample_{temp,mask,penalties,dry,argmax,topk,topp,minp,typical,xtc,
// pick,merge_topk}, declared in libr4d/r4d_plugin.h with the plugin ABI's signature. The flattened
// `(long lg, long params, int M, ...)` launch forms are private to r4d_sample_stages_f32.hip.
//
// Exact top-k: a radix select over the monotone logit key, a second over the token id to break a
// tie at the threshold by the lower id, then a descending bitonic sort of the survivors, which
// every later stage assumes. The optional `pairs` plane is the vocab-parallel path: (u32 global
// id, f32 logit) pairs with vocab_off added, what the all-gather moves and the merge reads. The
// merge re-applies the row's top_k: each rank kept k, so the merged set holds up to
// world_size * k and nothing downstream narrows again (at k = 1 it would no longer be an argmax).

}
