// R4D -- the RDNA4 (gfx1201) kernel library: attention, gated delta net, all-reduce and a skinny
// bf16 GEMM, compiled into one shared object. Plain C ABI: every entry point takes raw device
// pointers and a stream, so a consumer needs no C++ ABI coupling to call one. Nothing on the
// launch paths allocates or synchronises, so any of them can be recorded into a HIP graph.
//
// NAMING. An entry point is
//
//     r4d_<family>_<op>_<geometry it is compiled for>
//
// and the geometry suffix is not decoration: these are specialised kernels, and every dimension in
// the name is a compile-time constant that the entry point REJECTS a mismatch on rather than
// running. `attn_decode_h256_gqa6_fp8kv` runs for head_dim 256 with 6 queries per KV head and an
// fp8-e4m3 paged cache, and for nothing else. A model with a different head size needs a new
// instantiation, which will sit beside this one under its own name; nothing has to be renamed to
// make room for it. Dimensions that are fixed for the whole library, and so discriminate nothing
// between entry points, stay out of the names and are reported by r4d_*_dims() and the kernel
// registry instead: the paged block size (16), the query dtype (bf16 everywhere -- see
// r4d_attn_paged_h256_gqa6.hip) and the target architecture.
//
// SOURCE FILES CARRY THE SAME NAME. A translation unit is named for the entry point it provides,
// geometry included -- r4d_gdn_kkt_solve_k128_c64_bf16.hip holds r4d_gdn_kkt_solve_k128_c64_bf16
// and nothing else. Where one unit provides a family that differs only in a suffix, the file name
// stops at the shared part: r4d_gdn_conv_w4_h128_bf16.hip provides the prep and update pair, and
// r4d_attn_paged_h256_gqa6.hip provides all four paged variants by including the two kernel
// templates that sit beside it under their own names. Only the shared machinery -- r4d_common.h,
// r4d_dt16.h, r4d_gdn_wmma.h -- is named for what it is rather than for a kernel, because it
// provides none.
//
// Where a shape is unsupported the attention and GDN entry points return a negative code; the
// all-reduce and GEMM entry points throw std::runtime_error instead, and the plugin shim in front
// of them wraps the call in R4D_TRY, which turns the throw into an ABI status code. Nothing throws
// across the .so boundary.
#pragma once
#include <hip/hip_runtime.h>

// Library version. The git tag it was built from is expected to match -- which is what lets a
// consumer assert it linked the sources it pinned rather than whatever a stale clone happened to
// hold.
#define R4D_VERSION "0.5.0"

struct R4DArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3 or bf16
                                //   strides below are in ELEMENTS of that dtype; K then V per slot
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

    // ---- RAGGED BATCHES --------------------------------------------------------------------
    // `q_len` above is a SCALAR, so without these every sequence in a batch has to present the
    // same query length. A serving scheduler does not: it mixes one sequence's prefill chunk with
    // others' single decode rows in the same step deliberately, because a decode row costs nothing
    // once the weights are being read anyway. The uniform form is the special case, not the rule.
    //
    // `cu_q` is cu_seqlens: [num_seqs + 1] int32 on the DEVICE, cu_q[0] == 0, cu_q[num_seqs] ==
    // total_q, so sequence s owns query rows [cu_q[s], cu_q[s+1]) of the dense q and out blocks.
    // Null means uniform: row r of sequence `seq` is row `seq * q_len + r`, and the launchers, the
    // kernels and the scratch layout all reduce to exactly that arithmetic.
    //
    // `q_len` stays meaningful when cu_q is set: it is then the MAXIMUM query length in the batch,
    // which is what sizes the grid and picks the decode kernel's wave count. `total_q` is the row
    // count of q and out; it is host-side because the split-KV scratch has to be sized without
    // reading device memory. Both are ignored when cu_q is null.
    const int* cu_q;            // (num_seqs+1,) int32 device, or null for a uniform q_len
    int  total_q;               // rows of q/out; ignored unless cu_q is set
};

// Rows of q and out in this batch. The split-KV scratch is laid out by dense query row rather than
// by (seq, q_len), so this is the row count both the host sizing hook and the device kernels index
// against, and it must give the same answer in both -- which is why total_q is carried alongside
// cu_q rather than read off the end of it.
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
// The DFlash2 drafter's geometry: head_dim 128, 4 queries per KV head, sliding window.
// Decode only -- a query long enough to want the prefill kernel goes to the caller's
// fallback, which is what `r4d_attn_h128_gqa4_have_prefill()` reporting 0 tells it.
int  r4d_attn_decode_h128_gqa4_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h128_gqa4_bf16kv (const R4DArgs* a, hipStream_t stream);
long r4d_attn_decode_h128_gqa4_scratch_bytes(const R4DArgs* a);
// 0 means this build has no h128/gqa4 PREFILL kernel, so a caller that sees a query long
// enough to need one must route the whole batch to its fallback rather than this library.
int  r4d_attn_h128_gqa4_have_prefill(void);
// Bytes of split-KV partial buffer one decode launch of this shape needs. Independent of the cache
// dtype: the partials are f16 either way.
long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a);
// The geometry the attention kernels above are compiled for, so a caller can test a model against
// it instead of discovering the mismatch at the first launch.
void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows);

// ---- attention: vision encoder -------------------------------------------------------------
// Dense, non-causal, multi-head attention at the vision tower's head_dim of 72, bf16 throughout.
// Nothing is paged and nothing is quantised: q, k, v and o are [total_tokens, heads, 72] and
// contiguous, and cu_seqlens [num_seqs + 1] bounds one image (or one attention window) per entry,
// so a whole batch is ONE launch rather than a per-segment loop and a concatenate. `max_seqlen` is
// the host-side bound on those lengths and selects the query-block height; the kernel reads the
// device-side cu_seqlens itself. head_dim 72 is carried natively (5 k-tiles of 16, the last half
// structurally zero) rather than padded to 128. Returns -1 for any other head size.
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
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan and
// the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o).
// Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt are the compile-time 128/128/64 in the name; a mismatch returns -1 rather than
// running. Named for the algorithm, not the serving phase: this is the chunked scan, which is what
// prefill and chunked prefill both run. Decode uses the recurrent update, a different kernel.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
// The same scan against a SLOT-INDEXED state cache: `ssm_state_indices` is [N] and names the slot
// each batch row owns, and the two strides are the cache's own, in elements. The call above is
// this one with a null index and the packed [N, H, V, K] strides -- correct only while a slot
// equals its batch row, which stops being true the moment a slot is reused out of order.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, const void* ssm_state_indices, long indices_stride,
        long state_slot_stride, long state_head_stride,
        int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, which is
// FLA's chunk_scaled_dot_kkt_fwd followed by solve_tril with the fp32 gram in between never
// reaching HBM. k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk);
// A [T,H,64] bf16 out; cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// DFlash2's grouped dynamic depthwise convolution, fused into one pass. The reference builds it as
// ~6 full [T, H] elementwise passes plus a materialised [T, taps, num_groups, group] coefficient
// tensor, all consumed exactly once. delta is a SLICE of the [T, 2, taps, NG] kernel projection, so
// dpitch is 2*taps*NG; out must not alias x. block_size must be a power of two.
int r4d_dflash_conv_t2_g16_bf16(
        const void* x, const void* delta, const void* base, void* out,
        int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream);

// DFlash2's candidate selector: the low-rank edge scores and the greedy path walk, one workgroup a
// sequence and no launch between steps. `cand` and `unary` are row_topk's two output planes at row
// s*steps+l, `hp` is hidden_projection(x) for the same row, `anchor` is the token the target just
// committed, read at anchor[s * anchor_stride] so a packed [M] array is the stride-1 case. R must
// be a multiple of 32 and at most 512; K at most 32.
//
// `hp` MAY BE NULL, and then it is ones: the score becomes the plain bigram <pred[prev], succ[c]>
// with no dependence on the hidden state, which is DSpark's rank-256 Markov head over the base
// head's top K. `hp_ld` is ignored in that case.
//
// `V` is the codebooks' row count. A predecessor or candidate id outside [0, V) -- a merge pad is
// -1 -- reads no codebook row: the predecessor contributes nothing and the candidate scores its
// unary term alone, as ref_dflash_select has it.
int r4d_dflash_select_bf16(
        const void* cand, const void* unary, const void* hp, const void* anchor,
        const void* pred, const void* succ, void* out,
        int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
        int anchor_stride, void* stream);

// Everything between the qkv projection and the chunked scan, in one kernel: the depthwise causal
// convolution (width 4, silu) with its state cache, the q/k/v split, the l2 norm on q and k, the
// gate g = -exp(A_log).softplus(a + dt_bias) with its per-chunk cumsum, and beta = sigmoid(b).
// Replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum; the conv output never
// reaches HBM. Strides are in ELEMENTS; x may be a padded view (the qkvz split the layer hands it).
int r4d_gdn_conv_prep_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
        /* `hi_i32` gives has_init's WIDTH: 1 for int32, 0 for unsigned char. The caller's derived
         * buffers are int32 and this read them as bytes, which is right only when the value is 0. */
        const void* has_init, int hi_i32,
        const void* a, const void* b, long ab_stride, int ab_is_bf16,
        const void* A_log, const void* dt_bias, void* q, void* k, void* v, void* g, void* beta,
        const void* cu, int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr,
        void* stream);

// The same convolution for a decode step: the tokens are a speculative window, the state cache is
// a rolling buffer of width-1 + num_spec entries read at the slot the last ACCEPTED token left,
// and q / k / v are written straight into their own layouts. Replaces causal_conv1d_update and
// the cat that made its output contiguous.
int r4d_gdn_conv_update_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
        long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
        int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream);

// The gated RMS norm the layer applies to its own output: out = rms(x) . w . act(z), one row per
// (token, head). Replaces FLA's rmsnorm_fn. act: 0 = silu/swish, 1 = sigmoid. Only the prefill
// path needs it -- the decode kernel below folds the same arithmetic into its epilogue, because
// its workgroup owns the whole row.
int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o,
                                    long rows, long xrow, long zrow, long orow,
                                    long zheads, long ztok, int width,
                                    float eps, int act, void* stream);

// The recurrent delta-rule update decode runs where prefill runs the chunked scan: gating, the qk
// l2 norm, the state update and the output, against the paged state cache. The state is fp32
// (this model's config asks for it) and one is written per candidate token, which is the whole
// cost of the kernel. Replaces fused_sigmoid_gating_delta_rule_update.
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
// One-shot means each rank pushes its whole input into the peer's IPC scratch and then reduces
// locally -- no ring, no two-shot reduce-scatter, so the scratch is sized by the full message.
// Exactly 2 ranks: the handshake is a single peer flag, not a tree. Every 2-rank kernel on one
// buffer set shares its counters: `seq_ctrs` is 2 * r4d_ar_max_blocks() zeroed words, a handshake
// counter and a slot-parity counter per block of capacity (r4d_ar_seq.h).
enum { R4D_AR_HANDLE_BYTES = 64 };
// IPC scratch helpers. Startup-only, and not kernels: alloc returns the device pointer and writes
// the IPC handle (up to HANDLE_BYTES) into out_handle.
long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len);
long r4d_ar_ipc_open(const char* handle, int len);
// Strided peer (or local) device-to-device copy through the copy engines; pitches and width in
// bytes, height in rows. Returns 0 or -100 - hipError_t.
int r4d_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream);
int r4d_p2p_fence(long stream);
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream);
void r4d_ar_ipc_free(long p);
void r4d_ar_ipc_memzero(long p, long size);
void r4d_ar_ipc_enable_peer(long peer);
// Exact sum, fp32 accumulate, bf16 / fp16 / fp32 payload (dtype 0 / 1 / 2).
void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                                long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                                long dtype, long stream, long nblocks, long nthreads, long drain,
                                long acq);
// Same topology, but the wire payload is Walsh-Hadamard rotated and quantised to 6 bits per element
// over groups of 64 (plus a bf16 scale per group). Lossy, and bf16 / fp16 payload only. Takes this
// rank's own packed copy (loc_pack) as well, so the reduce folds exactly the bytes it sent.
void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq);
// The exact all-reduce of `y` [M][n] bf16 with hc_write as its epilogue: h[r][c*n + i] +=
// inj[r][c] * y[r][i] over the reduced y, byte-identical to all_reduce then hc_write. `irow` and
// `hrow` are the pitches of inj [M][hc] and h [M][hc*n] in elements; y is tight.
void r4d_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream, long nblocks, long nthreads,
                     long drain, long acq);
// The same with the MoE gather as its prologue: y is not read but produced -- the routed experts'
// rows `ye` [M*top_k][ye_ld] summed back to token order by the weights `ew` [M][ew_ld] and the
// sorted list `sorted` [M*top_k], with the shared arm `sh` [M][sh_ld] and its gate `sg` folded
// when both are non-zero -- byte-identical to moe_gather, then all_reduce, then hc_write. `inv`,
// when non-zero, is M*top_k int32 of device scratch for the inverse slot table a prefill-sized
// call reads instead of its per-block LDS one. Returns 0, or a negative code for a shape it does
// not serve.
int  r4d_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                            long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                            long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                            long top_k, long sig, long stream, long nblocks, long nthreads,
                            long drain, long acq, long inv);
// r4d_ar_hc_write and r4d_ar_gather_hc_write on the rotated 6-bit wire (r4d_ar_oneshot_2rank_wht6's
// format and arithmetic, its loc_pack and scratch split): byte-identical to all_reduce on that wire
// then hc_write, and with `ye` non-zero to moe_gather, then those two. The gather's operands are
// r4d_ar_gather_hc_write's; its slot table is built in LDS, and the grid is widened until a
// block's table fits. M*n must be a multiple of 64. Returns 0, or a negative code for a shape it
// does not serve.
int  r4d_ar_hc_write_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                          long seq_ctrs, long loc_pack, long slot_stride_bytes,
                          long scale_off_bytes, long y, long inj, long h, long M, long n, long hc,
                          long irow, long hrow, long ye, long ew, long sorted, long sh, long sg,
                          long ye_ld, long ew_ld, long sh_ld, long sg_ld, long top_k, long sig,
                          long stream, long nblocks, long nthreads, long drain, long acq);
int  r4d_ar_max_blocks(void);                                   // the two 2-rank kernels
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems); // rotated-6-bit payload only

// ---- all-GATHER: one-shot, push, exactly 2 ranks over P2P ----------------------------------
// The same push and the same per-block flag handshake as r4d_ar_oneshot_2rank_exact, with the
// reduction replaced by placement: `out` is 2 * n_elem elements, this rank's input at
// rank * n_elem and the peer's at (1 - rank) * n_elem. `dtype` is used only for the element size
// -- the payload is moved, never summed, because the vocab-parallel sampler gathers (u32 token id,
// f32 logit) pairs and adding the id lanes would destroy them.
void r4d_ar_gather_2rank(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                         long dtype, long rank, long row_elems, long stream, long nblocks,
                         long nthreads, long drain, long acq);

// ---- all-reduce: one-shot, push, 4 or 8 ranks over P2P -------------------------------------
// The same algorithm at width (r4d_ar_oneshot_Nrank_exact.hip). The peer arguments are ascending
// global rank with the current rank removed. Scratch is 2*world size slots of slot_stride16 16B 
// words parity x source rank, this rank's own slot unused); flags are max_blocks*ws words indexed 
// [block][source rank]; seq_ctrs is 2 * max_blocks words, a handshake and a slot-parity counter
// per block of capacity (r4d_ar_seq.h).
// Three ranks: the same kernel instantiated at WS=3 (two peers). Nothing in the wide family is
// power-of-two; the two-shot cannot serve 3 because it shards the message `world` ways.
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
                                long pub);
// ---- all-gather at width (r4d_ar_gather_Nrank.hip) ------------------------------------------
// The 2-rank gather's placement on the wide family's transport: `out` is world * n_elem elements,
// rank r's slice at r * n_elem, or row-interleaved [rows][world][row] when row_elems > 0. Shares
// the wide all-reduce's instance (scratch, [block][source] flags, seq counters).
void r4d_ar_gather_3rank(long peer_scratch0, long peer_scratch1, long peer_flags0,
                         long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                         long slot_stride16, long inp, long out, long n_elem, long dtype,
                         long rank, long row_elems, long stream, long nblocks, long nthreads,
                         long drain, long acq, long pub);
void r4d_ar_oneshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub);
// r4d_ar_hc_write at four ranks: the 4-rank exact one-shot of `y` [M][n] bf16, in place, with
// hc_write as its epilogue -- byte-identical to that all-reduce then hc_write. Same buffer trio,
// peer order, flag values and slots as r4d_ar_oneshot_4rank_exact, so the two (and the two-shots)
// may share one instance. `irow`/`hrow` as r4d_ar_hc_write's; y is tight.
void r4d_ar_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                           long peer_flags0, long peer_flags1, long peer_flags2,
                           long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                           long y, long inj, long h, long M, long n, long hc, long irow,
                           long hrow, long rank, long stream, long nblocks, long nthreads,
                           long drain, long acq, long pub);
// r4d_ar_gather_hc_write at four ranks: r4d_ar_hc_write_4rank with the MoE gather as its prologue
// -- byte-identical to moe_gather, then the 4-rank exact all-reduce, then hc_write. The gather's
// operands, `inv` and the grid rule are r4d_ar_gather_hc_write's; the buffers, peer order, flag
// values and slots are r4d_ar_oneshot_4rank_exact's. Exact wire only.
void r4d_ar_gather_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                  long peer_flags0, long peer_flags1, long peer_flags2,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride16, long y, long inj, long h, long M, long n,
                                  long hc, long irow, long hrow, long ye, long ew, long sorted,
                                  long sh, long sg, long ye_ld, long ew_ld, long sh_ld,
                                  long sg_ld, long top_k, long sig, long rank, long stream,
                                  long nblocks, long nthreads, long drain, long acq, long pub,
                                  long inv);
// The inverse slot table both widths' gather forms read at a prefill-sized call: inv[f] = the
// sorted row holding slot f, -1 where none does, over T = M*top_k int32. One block, on `stream`.
// Returns 0 or a negative / HIP error code.
int  r4d_ar_gather_inv(long inv, long sorted, long T, long stream);
void r4d_ar_wide_dims(int* max_blocks, int* nb_design, int* max_peers);

// ---- all-reduce: two-shot, 4 or 8 ranks over P2P -------------------------------------------
// Reduce-scatter + all-gather on the same buffer trio and peer-argument order as the wide
// one-shot (r4d_ar_twoshot_Nrank_exact.hip): shard s, a contiguous 1/ws of the message, is
// reduced by rank s (fp32, ascending rank order), then gathered. Sends 2*(ws-1)/ws*N bytes
// per rank in two hops. Covers the sizes where the one-shot's (ws-1)*N send is too heavy.
// Regions A and B sit at 0 and shard16, so the slot stride must hold 2*shard16. Above this bound
// the caller falls back:
enum { R4D_AR_TWOSHOT_WIDE_MAX_ELEMS = 20971520 };  // a 4096-token hidden-5120 prefill chunk
void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub);
void r4d_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub);
// The tiered int8 wire on both of the two-shot's hops (r4d_ar_twoshot_Nrank_ti8.hip; codec
// in r4d_ar_ti8.h), 4 ranks only. The wire: int8 groups of 32 with an fp16 scale and a
// per-group record (T1 plain, T2 top-1-exact, T3 wht32) 38 bytes/group, deterministic
// tiering, bit-identical across ranks. Same peer-argument order and flag layout as the exact
// pair; slot stride in bytes (the wire is not 16B-aligned), holding two wire images; the 
// encoder writes its own slot too (no loc_pack). numel must tile 32-elem groups over the ws
// shards (numel % 128).
void r4d_ar_twoshot_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                              long peer_flags0, long peer_flags1, long peer_flags2,
                              long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub);
void r4d_ar_ti8_dims(int* group, int* group_bytes);  // tiered-int8 wire only

// ---- GEMM ----------------------------------------------------------------------------------
// Skinny bf16 GEMM for M up to 64: the same C[M,N] = A[M,K] @ W[N,K]^T, computed with
// 16x16x16 WMMA so a 16-wide step of K costs ceil(M/16) activation fragments and one weight
// fragment, instead of one weight load and M activation loads.
void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream);
int  r4d_gemm_bf16_nt_m64_max_m(void);
// The same GEMM at prefill row counts: LDS-staged tiles, eight waves a block, and output bytes
// identical to r4d_gemm_bf16_nt_m64 at the same split count SK. `tile` picks the block: 1 = 128 x
// 128, 2 = 64 x 64; r4d_gemm_bf16_nt_tiled_arm is the automatic choice. K a multiple of 64.
// Returns 0, or negative on a shape it does not serve, or the HIP error of the launch.
int  r4d_gemm_bf16_nt_tiled(long a, long w, long c, int M, int K, int N, int SK, int tile,
                            long stream);
int  r4d_gemm_bf16_nt_tiled_arm(int M, int N);

// Skinny GEMM with a 4-BIT weight: C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C. The weight
// is asymmetric per output channel per group of `group()` contiguous K -- w ~= scale * (q - zero),
// q in 0..15, zero an integer -- and is pre-permuted offline into the WMMA fragment order, so the
// kernel's whole weight path is one global_load_b128 per lane per four k steps. Wq is N*K/2 bytes;
// Wsz is one dword per (row, group), the f16 scale in its low half and the f16 of -(1024 + zero)
// in its high half. N must be a multiple of 16 and K divisible by SK * group().
void r4d_gemm_w4a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w4a16_nt_m64_max_m(void);
int  r4d_gemm_w4a16_nt_m64_group(void);

// Skinny GEMM with an 8-BIT weight: C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C. Same
// fragment order, packed-block size and (scale, zero) dword as r4d_gemm_w4a16_nt_m64, with one
// packed block covering 32 K instead of 64. Asymmetric per output channel per group of `group()`
// contiguous K -- w ~= scale * (q - zero), q in 0..255, zero an integer. Wq is N*K bytes; Wsz is
// one dword per (row, group), the f16 scale low and the f16 of -(1024 + zero) high.
void r4d_gemm_w8a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w8a16_nt_m64_max_m(void);
int  r4d_gemm_w8a16_nt_m64_group(void);


// Exact per-row top-R of a wide bf16 matrix, in two launches.  `partial` is
// M * rowtopk_chunks(N, NCH) * RC int64, where RC >= R is how many a chunk keeps; `outv` is M*R float and `outi` M*R int32, both descending
// by value and, on a tie, by the LOWER column.  NCH is a REQUEST -- the chunk width is rounded up
// to a multiple of eight columns, so ask rowtopk_chunks() what the launch will actually use.
void r4d_rowtopk_bf16(long y, long partial, long outv, long outi,
                      int M, int N, int R, int RC, int NCH, int PT1, long stream);
// The same top-R over one rank's SLICE of a wider plane.  `off` is the first global column the
// slice stands for and is added to every column this writes, so `outi` is global ids; an empty
// slot still writes column 0, because offsetting a pad would make it this rank's first real token.
// `outp` -- 0 for none -- is an M*R plane of (int32 id, float value) pairs, eight bytes an entry,
// which is the element an all-gather moves so ranks can agree on a global top-R without gathering
// the plane itself; a pad writes id -1 there, the sentinel a merge drops.  At off = 0 with outp 0
// this is r4d_rowtopk_bf16 to the bit, which is what the plain entry point calls.
void r4d_rowtopk_bf16_sharded(long y, long partial, long outv, long outi, long outp,
                              int M, int N, int R, int RC, int NCH, int PT1, int off,
                              long stream);
// The other half of a sharded top-R: the R largest of `gathered`, an [M, W*R, 2] plane of (int32
// id, float value) pairs with rank r's at [r*R, (r+1)*R), written in the same (`outv`, `outi`)
// form the two stages above write.  Id -1 is a pad and is dropped; ties break on the LOWER id, so
// the answer equals a top-R over the unsharded plane rather than merely approximating it.
void r4d_rowtopk_merge(long gathered, long outv, long outi, int M, int R, int W, long stream, int ld = 0);
void r4d_logit_rerank(long x, long idx_in, long head, long outi, long outv, long outp,
                      int M, int R, long K, long voff, long nv, long stream, int R_pairs = 0);
int  r4d_rowtopk_bf16_chunks(int N, int NCH);

// 2-BIT weight, 8-bit activation, for the DFlash2 draft head.  Unsigned 2-bit codes in the byte's
// high two bits, asymmetric per (row, group of group()) K, corrected with r4d_quant_act_i8_asum's
// [K/group][Mpad] group-major sums -- the same activation, quantiser and layout as
// r4d_gemm_w4a8_asym_nt_m64.  Wq is N*K/4 bytes; Ws is the shared (f16 scale, f16 of -(1024+zero))
// dword per (row, group).
void r4d_gemm_w2a8_nt(long a, long ascale, long wq, long ws, long asum, long c,
                      int M, int K, int N, int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w2a8_nt_group(void);

// 4-bit weight, 8-bit activation. Signed 4-bit codes, per-row activation scale, int8 WMMA.
void r4d_gemm_w4a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream);
// Asymmetric variant: the weight nibble is read UNSIGNED by the WMMA and the stored zero point is
// corrected out with `asum` from r4d_quant_act_i8_asum.  Same packed weight, same scale dword --
// the zero rides in its high half as the f16 of -(1024 + zero), which is the encoding the f16
// kernel already requires.
void r4d_gemm_w4a8_asym_nt_m64(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               long stream);
// The same GEMM tiled for PREFILL batches, where the skinny kernel gives a block 64 rows and so
// re-reads the weight M/64 times.  Same packed weight and same pre-shuffled activation; N must be a
// multiple of 256 and K of the scale group.
// variant 1 is the transposed-accumulator epilogue (see the source); 0 is the M-major original.
void r4d_gemm_w4a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant = 0);
// Same kernel on the ASYMMETRIC grid: unsigned nibbles plus r4d_quant_act_i8_asum's [G][Mpad] row
// sums.
void r4d_gemm_w4a8_asym_tiled(long a, long ascale, long wq, long ws, long asum, long c, int M,
                              int K, int N, long stream, int variant = 0);
// The same GEMM again, for PREFILL row counts, where the matrix pipe rather than the weight stream
// is the constraint.  Same packed weight and same rotated int8 activation as the tiled kernel; N
// must be a multiple of BN (256 at the default shape) and K of the scale group.  The differences
// are in the inner loop: the activation is stored half-major in LDS so one `ds_load_b128` fetches
// two k steps, and the group fold is plain `v_fmac_f32` so the VOPD packer can pair it.
void r4d_gemm_w4a8_prefill(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                           long stream, int variant = 0);
int  r4d_gemm_w4a8_nt_m64_max_m(void);
int  r4d_gemm_w4a8_nt_m64_group(void);
int  r4d_gemm_w4a8_nt_m64_aperm(void);
int  r4d_gemm_w4a8_nt_m64_ws16(void);

// 8-BIT weight, 8-bit activation.  The promoted linears of the INT4 target (every down_proj and the
// attention o_proj) and lm_head.  Signed int8 codes in the iu8 16x16x16 fragment order -- the same
// order the A side already uses, so the packer is one routine at two radices -- with one f16 scale
// per (output row, group of group() along K) in an f16-ONLY plane, and the per-row activation scale
// applied once at the store.  `a` is the SAME permuted int8 buffer r4d_gemm_w4a8_nt_m64 reads, from
// the same fused producer.  Wq is N*K bytes; Ws is N*K/group() f16.
void r4d_gemm_w8a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream);
int  r4d_gemm_w8a8_nt_m64_max_m(void);
int  r4d_gemm_w8a8_nt_m64_group(void);
int  r4d_gemm_w8a8_nt_m64_kpb(void);
int  r4d_gemm_w8a8_nt_m64_aperm(void);
// The same 8-bit GEMM tiled for PREFILL, where the skinny kernel gives a block MB*16 rows and so
// re-reads the weight M/(MB*16) times.  Same packed weight and same permuted activation; N must be
// a multiple of the variant's BN (256 at the default shape) and K of the scale group.  Inherited
// from r4d_gemm_w4a8_tiled's "unpack at staging" path, whose LDS buffer already holds bytes.
void r4d_gemm_w8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant = 0);

// OCP-MXFP4 weight, fp8 activation. e2m1 elements with one E8M0 exponent per 32 K, folded
// against a per-row reference exponent so the inner loop has no rescale; fp8 WMMA.
void r4d_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long wref, long c,
                             int M, int K, int N, int WV, int SK, int MB, int NPW, long stream);
int  r4d_gemm_mxfp4a8_nt_m64_max_m(void);
int  r4d_gemm_mxfp4a8_nt_m64_group(void);

// ---- BLOCK-SCALED FP8: the format a checkpoint ships in ----------------------------------------
//
// Every other quantised GEMM above reads a weight this library's own quantiser produced.  This
// family reads the weight as safetensors stores it: `<linear>.weight` F8_E4M3 row-major [N][K] and
// `<linear>.weight_scale_inv` BF16 on a [ceil(N/128)][ceil(K/128)] grid -- DeepSeek-style block
// scaling, `quantization_config.weight_block_size = [128, 128]`.  No permutation and no scale
// conversion: the bytes in the file are the bytes these kernels read, so a container built for them
// costs exactly what the checkpoint does.
//
// THE BLOCK IS 128 ON BOTH AXES AND IS NOT A BUILD FLAG, unlike R4D_GEMM_W4A8_GROUP.  That group is
// a choice this library's quantiser makes; this one is a property of the file on disk.  The
// `_block()` queries report it and every entry point rejects a K that would make the fold read a
// partial one.
//
// Three entry points, one per band, and they are three kernels rather than one with a switch
// because the operand carrying the coarse (per-128-row) scale must be the one whose tile is wide --
// see the AW comment in r4d_gemm_fp8a8.hip.
//
//   M == 1    r4d_gemm_fp8a16_nt_m1   bf16 activation read directly, no quantise launch
//   M <= 16   r4d_gemm_fp8a8_nt_m16   E4M3 activation, one 16-wide token tile, split-K
//   any M     r4d_gemm_fp8a8_tiled    E4M3 activation, 128x128 tile, eight waves
//
// All of them take `w_ld` in BYTES a weight row and `sw_ld` in ELEMENTS a scale row, 0 meaning
// tight, and all of them return 0 or a NEGATIVE code rather than throwing the way the other GEMM
// entry points do.  These are reachable from a C consumer, where an exception crossing an .so
// boundary is precisely the failure mode that boundary exists to avoid.
int  r4d_gemm_fp8a16_nt_m1(long a, long wq, long ws, long c,
                           int M, int K, int N, int w_ld, int sw_ld, int rb, long stream);
// The same GEMM for M <= r4d_gemm_fp8a16_nt_m1_max_m(): M token rows against one weight row, so the
// weight is read ONCE for all of them. `a` is [M][K] with `a_ld` elements a row and `c` is [M][N]
// with `c_ld` (0 = N). This is the speculative-verify width -- the alternative at M = 1 + n_spec is
// the 16-wide fp8a8 tile, which needs the activation quantised first and reaches a lower fraction
// of the memory ceiling on identical weight traffic.
int  r4d_gemm_fp8a16_nt_mt(long a, long wq, long ws, long c, int M, int K, int N,
                           int a_ld, int c_ld, int w_ld, int sw_ld, long stream);
/* The lm_head at fp8, scales interleaved in each row's tail; see the .hip for why that layout. */
int  r4d_logits_gemm_fp8(long a, long wq, long c, int M, int K, int N,
                         int a_ld, int c_ld, int w_ld, long stream);
int  r4d_gemm_fp8a16_nt_m1_block(void);
int  r4d_gemm_fp8a16_nt_m1_max_m(void);

// `a` is E4M3 [M][K] with `ascale` f32 [M][ceil(K/128)] -- r4d_quant_act_fp8's output.  `scratch` is
// the split-K partial plane and may be 0, in which case the launch runs unsplit: correct and slow
// rather than wrong.  Size it with the query, whose answer is 0 when this shape will not split.
//
// SPLIT-K IS NOT BIT-COMPARABLE ACROSS SETTINGS and cannot be: ksplit repartitions K, so the
// accumulation chain differs and the emitted text can move.  The invariant that matters -- two runs
// of the SAME binary agreeing -- is unaffected.
int  r4d_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                           int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                           long scratch, long scratch_bytes, long stream);
long r4d_gemm_fp8a8_nt_m16_scratch(int M, int K, int N);
int  r4d_gemm_fp8a8_nt_m16_max_m(void);
// A TEST HOOK, and the only way to ask the bit question about the narrow tile's wave count. The
// wave count is a launch heuristic chosen from (N, M), so two settings of it cannot be compared
// from outside libr4d and the environment pin that sweeps it is read once per process. This
// overrides the rule for every subsequent call on every thread; 0 restores it. Not for a serve --
// r4d_selftest holds it for the length of one comparison and puts it back.
//
// Unlike the split above, the wave count IS bit-comparable BY CONSTRUCTION, which is what the
// check exists to hold to: it moves which wave owns an output element and not the order of the
// K tiles that element accumulates, and the split is derived from `gx * mw`, which the rule keeps
// invariant. `r4d_selftest` compares the bytes at every legal wave count on every shape it runs.
void r4d_gemm_fp8a8_pin_mw(int mw);
// THE QSA SCORE'S FORM, pinned for a test: 0 is the launcher's rule, 1 the scalar decode form, 2
// the matrix form (which then refuses a shape it is not built for). The two sum in different
// orders, so r4d_selftest's --perfqsa compares them to a tolerance rather than byte for byte.
void r4d_qsa_score_pin_form(int form);
// THE QSA SELECTION'S FORM, pinned for a test: 0 is the launcher's rule, 1 the single-workgroup
// kernel, 2 the wide-launch (three-pass) kernel. The split path keeps its own rule. The two pinned
// forms write the same table bit for bit, and r4d_selftest compares them.
// THE ROUTED-EXPERT GEMM'S FORM, pinned for a test: 0 is the launcher's rule, 1 the run-aligned
// prefill form, 2 the (tile, run) decode form, 3 the uniform prefill grid. r4d_selftest --perfmoe
// times each and compares it with form 1.
void r4d_moe_gemm_pin_prefill(int form);
// THE GATED RESIDUAL READ'S PREFILL TILE, pinned for a test: 0 is the launcher's rule, 1 the decode
// tiles' kernels at 128 rows (hc_down_tiled, then hc_up_tiled into `ubuf` and hc_mix), 2 the wide
// kernels (which then refuse a shape they are not built for). The two are bit-identical by
// construction, and r4d_selftest compares their bytes.
void r4d_hc_read_pin_prefill(int form);
// THE SAME NARROW GEMM WITH r4d_gated_quant_fp8's PASS IN ITS EPILOGUE, for a gate_up projection
// whose only consumer is that quantiser.  `wq` is the [2*nout][K] gate_up plane -- gate first, the
// layout the unfused GEMM already reads -- and the outputs are the quantiser's, not the GEMM's:
// `q` E4M3 [M][nout], `s` f32 [M][ceil(nout/128)], and `y` the bf16 silu product or 0.
//
// BYTE-IDENTICAL to the pair, and unlike the split above that is a property this is written to
// hold: the accumulator is rounded to bf16 first and the gate, the product and the amax are all
// taken over the round-tripped values, which is what a GEMM that stores a tensor and a quantiser
// that reads it back compute.  `act` is r4d_gated_quant_fp8's, and only 0 (silu) is compiled.
//
// Refuses -- rather than falling back -- any shape the fold does not fit: M past the band, K not a
// multiple of 256, `nout` not a multiple of 128 (a block emits exactly one 128-column scale group
// and there is nowhere to finish a straddling one), or a grid the split rule would widen.
int  r4d_gemm_fp8a8_gated_nt_m16(long a, long ascale, long wq, long ws,
                                 long q, long s, long y, int act,
                                 int M, int K, int nout,
                                 int a_ld, int sa_ld, int w_ld, int sw_ld,
                                 int q_ld, int s_ld, int y_ld, long stream);
int  r4d_gemm_fp8a8_gated_nt_m16_max_m(void);
int  r4d_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long stream);
// The same two GEMMs at INT8 codes for both operands (r4d_gemm_fp8a8.hip's I8 instantiations):
// the activation i8 with its f32 scale per (row, 128 K) as the fp8 one, the weight i8 in the same
// fragment order with a bf16 scale per ROW per 128 K, tile-major: ws is [N/16][ceil(K/128)][16]
// (r4d_fp8_layout.cpp's i8 hooks write it); sw_ld is not read.
int  r4d_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                          int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                          long scratch, long scratch_bytes, long stream);
// The lm_head at INT8 codes (logits_gemm_i8): r4d_logits_gemm_fp8's stored row -- codes, the row's
// bf16 scales in its tail, the pad -- with i8 codes and each row's own scale per 128 columns.
int  r4d_logits_gemm_i8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                        int w_ld, long stream);
int  r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                         int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                         long stream);
int  r4d_gemm_fp8a8_block(void);

// Per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation -- what the two fp8a8
// GEMMs read.  PLAIN ROW-MAJOR output, unlike r4d_quant_act_i8's fragment order: the fp8a8 GEMMs
// stage both operands through the same byte copy and the same swizzle, so a permutation would have
// to be undone.  The scale is amax/448, a DEQUANT multiplier in the same sense as the checkpoint's
// weight_scale_inv, so the GEMM multiplies both scales and never divides.
int  r4d_quant_act_fp8(long a, long q, long s, int M, int K,
                       int x_ld, int q_ld, int s_ld, long stream);
int  r4d_quant_act_fp8_group(void);

// The same quantiser riding on the pass that PRODUCED the row, which is where most of a decode
// step's quantise launches come from.  Both write the bf16 output the unfused producer wrote AND the
// E4M3 pair, and both quantise the bf16-ROUNDED value so the codes are the ones the unfused pair
// produced -- see r4d_fused_quant_fp8.hip on why that is a requirement and not a nicety.  `y` may
// be null where nothing reads the bf16.  `w_f32` states the norm gain's width; `mode` is 0 for
// silu and 1 for sigmoid.
// `res` non-null is the fused_add_rms_norm form -- `res += x`, rounded to bf16 and STORED, then
// normalised -- which is bit-for-bit the `add` + `rmsnorm` pair it replaces, and removes one
// launch per block per layer. It may alias `y`.
int  r4d_rmsnorm_quant_fp8(long x, long res, long w, long y, long q, long s, int M, int n,
                           float eps, float wadd, int w_f32, int x_ld, int r_ld, int y_ld,
                           int q_ld, int s_ld, long stream);
int  r4d_gated_quant_fp8(long gu, long y, long q, long s, int M, int n, int mode,
                         int gu_ld, int y_ld, int q_ld, int s_ld, long stream);
// The activation is rounded to bf16 before the product: the unfused pair is `sigmoid` then `mul`,
// two kernels with a bf16 buffer between them, and that round trip is part of what is replaced.
// The same epilogue with the gate in a DIFFERENT buffer from the thing it scales, which is the
// shape a gated attention has: `x` is the attention output and the gate is a column slice of the
// [q|gate] projection.  The row is then a (token, head) pair rather than a token -- at that shape
// both sources are a fixed stride apart and the scale plane still lands exactly where the out
// projection reads it.  `g` is passed ALREADY OFFSET to the gate half of its row.
int  r4d_gate_quant_fp8(long g, long x, long y, long q, long s, int M, int n, int mode,
                        int g_ld, int x_ld, int y_ld, int q_ld, int s_ld, long stream);

// The same norm again with the tensor-parallel all-reduce that always precedes it folded in.  Every
// row-parallel linear's output is all-reduced and then immediately normalised, so the pair is 128
// adjacent dispatches on a decode step; fused, the all-reduce's bf16 output is never read back and
// the peer's contribution is a second operand on the norm's load.  The IPC arguments are
// r4d_ar_oneshot_2rank_exact's; `nblocks` is CLAMPED TO M, because a block owns whole rows, and `x`
// takes no pitch, because it is the message as well as the input.  The int8 twin is
// r4d_ar_ln_had_quant_i8, and like it this one carries BOTH wires: `wire` is 0 for the exact bf16
// payload and 6 for the rotated 6-bit one, which the caller picks the same way the standalone
// all-reduce does.  Wire 6 needs `n` to be a multiple of r4d_ar_rmsnorm_quant_fp8_chunk_elems(),
// so that a chunk never straddles a row.
int  r4d_ar_rmsnorm_quant_fp8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                              long seq_ctrs, long slot_stride16,
                              long x, long res, long w, long y, long q, long s,
                              int M, int n, float eps, float wadd, int w_f32,
                              int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                              int nblocks, int nthreads, int drain, int acq, int wire,
                              long loc_pack, long loc_scale_off);
int  r4d_ar_rmsnorm_quant_fp8_chunk_elems(void);

// ---- sampling ----------------------------------------------------------------------------------
//
// One workgroup a position: temperature, softmax, top_k, top_p, min_p, renormalise and the draw, in
// vLLM's order -- reordering changes which tokens survive.  Three outputs, because a greedy-draft
// speculative verify needs all of them from one pass: a draw, a draw with the proposed token
// removed and the remainder renormalised, and the proposal's FILTERED probability, which is the
// accept probability under rejection sampling.
struct R4DSpecDraw {
    unsigned int  tok;        // a draw from the filtered distribution
    unsigned int  tok_excl;   // a draw with `query` removed and the remainder renormalised
    float         p_query;    // filtered probability of `query` -- the accept probability
};
// `logits[pos*col_stride + v*row_stride]`, so either head layout is read without a transpose.
// `query` may be 0; `u` is two uniforms a position.  `n_vocab` is the FULL vocabulary: on a
// vocab-parallel rank the caller gathers the other bands first, because a nucleus over half a
// distribution is not a nucleus.
int  r4d_sample_chain_f32(long logits, long query, long u, long out,
                          int M, int n_vocab, int row_stride, int col_stride,
                          float temperature, float top_p, int top_k, float min_p, long stream);

// ---------------------------------------------------------------------------- the quantisers
//
// THESE TAKE THE PLUGIN ABI'S ARGUMENTS, not a flattened C list: `int r4d_rad_<name>(const
// RadArgs*, RadStream)`, declared in libr4d/r4d_plugin.h because the ROW TABLE is the only
// caller and that header is host-only. The entry point named under each note is the symbol.
// See libr4d/r4d_args.h for why that is the only calling convention libr4d has.

// Per-row symmetric int8 quantisation of a bf16 activation, in the A-fragment byte order. The
// optional fourth operand asks for the per-row per-group sum of the int8 codes an asymmetric
// weight grid needs -- [K/group][Mpad] float, group-major; see r4d_gemm_w4a8_asym_nt_m64.
//   r4d_rad_quant_act_i8
// Block Hadamard of width `had` along K, then the same per-row int8 quantisation. `had` must be a
// power of two dividing K -- a rotation block cannot cross a tensor-parallel shard boundary, so the
// per-rank K is what it has to divide (5120 / 3072 / 8704 here, which rules 1024 out).
//   r4d_rad_had_quant_act_i8
// The residual add, the RMS norm, the same block Hadamard and the same int8 quantiser in ONE pass.
// `residual` and `out_bf16` may be null: pass a residual for vLLM's fused_add_rms_norm form, and an
// out_bf16 only on layers that still have a bf16 consumer of the post-norm tensor (a GDN block's
// in_proj_a / in_proj_b, which stay bf16 and read it unrotated).
// `wadd` is added to every gain element in fp32 after the bf16 load: GemmaRMSNorm (Qwen3.5,
// Qwen3-Next, Gemma) stores a zero-centred gamma and uses `1 + w`, and forming that in bf16
// first costs 0.3% per channel.  Pass 0.0f for a plain RMS norm.
//   r4d_rad_rmsnorm_had_quant_i8
// The same norm, with the tensor-parallel all-reduce that always precedes it folded in.  Every
// row-parallel linear's output is all-reduced and then immediately normalised, so the pair is 128
// adjacent dispatches on a decode step; fused, the all-reduce's bf16 output never exists and the
// peer's contribution is just a second operand on the norm's load.  The IPC arguments are
// r4d_ar_oneshot_2rank_exact's, with the same double-buffered scratch, per-block flags and
// device-resident sequence counters; `nblocks` is CLAMPED TO M, because a block owns whole rows.
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
// An elementwise gated product, then the same Hadamard and int8 quantiser.  Two of the four rotated
// sites have this shape and differ by one multiply: mode 0 is silu(a)*b (down_proj's input, vLLM's
// silu_and_mul) and mode 1 is sigmoid(a)*b (o_proj's input, since Qwen3_5Attention computes
// attn_output * sigmoid(gate)).  Strides are in ELEMENTS and may be 0 for a tight K-pitch; pass 2*K
// for vLLM's fused gate_up layout where `b` is `a + K`.
//   r4d_rad_gated_had_quant_i8
// The gated-attention QKV preamble: split [q|gate] per head, QK RMS norm, partial NeoX RoPE and
// the gate copy, in one pass.  Replaces four elementwise kernels per attention layer.  The norm
// output is rounded to bf16 BEFORE the rotation, because the unfused reference stores it and reads
// it back.  `wadd` is the GemmaRMSNorm `1 +` (see r4d_rmsnorm_had_quant_i8).  Pitches are in
// ELEMENTS.
//   r4d_rad_qk_norm_rope_gate
// Fills the cos/sin table the fused kernel above reads, at each position's OWN row of an
// [max_ctx, rot] plane: `rot` values a row, cos for the first rot/2 and sin for the second.  One
// launch a step serves every layer.  The arithmetic is r4d_model_bf16.hip's rope_kernel line for
// line, which is what makes the fusion bit-for-bit against the unfused norm + rope pair.
//   r4d_rad_rope_table
// The GDN block's gated RMS norm -- x * rsqrt(mean_head(x^2)+eps) * w * silu(z) over 128-wide heads
// -- then the same Hadamard and int8 quantiser over the flattened row.  out_proj's input on the 48
// linear-attention layers.  `wgain` is 128 floats; strides are in elements and may be 0.
// `zgrp`/`zblk` let Z be the INTERLEAVED qkvz projection instead of a compacted gate: element i of
// the flattened row is at (i / zblk) * zgrp + (i % zblk).  Pass 0 for a contiguous gate.
//   r4d_rad_gdn_gated_norm_hq_i8
// The packed 4-bit weight back to a plain row-major bf16 [N, K], undoing the fragment order.  For
// prefill, where W4A8 wins nothing and the skinny GEMM is structurally bad: dequantise a layer into
// a scratch buffer and hand it to the tuned bf16 GEMM.  Serves both grids -- the zero comes out of
// the scale dword's high half, which is the constant 8 for symz.
// `twos` says which encoding these bytes carry: 1 for the symmetric grid's two's-complement
// code (value = (n^8)-8, what radiance_w4.pack writes) and 0 for the asymmetric grid's
// unsigned nibble minus a per-group zero.  They differ by bit 3 and the wrong one is silently
// wrong -- it agrees for n<8 and is off by 16 above.
void r4d_dequant_w4_bf16(long wq, long ws, long out, int N, int K, int group, int twos,
                         long stream);

// ---- the small bf16 model ops (r4d_model_bf16.hip) ---------------------------------------------
//
// The embedding gather, the norm, the adds and the rotary that sit BETWEEN the interesting kernels.
// They are here because a step is a chain and a chain with one host link does not run: the buffers
// are in VRAM and the engine refuses an op with no device launch.  The arithmetic is libref's
// -- every reduction in f32, only the store narrows, round-to-nearest-even -- because ref is the
// oracle these are compared against.
//
// Every pitch is in ELEMENTS and 0 means tight.  Index arrays are i32 (the engine's derived buffers
// all are).  A NEGATIVE index writes a zero row or skips the store; an OUT-OF-RANGE one does the
// same, because a device kernel cannot return a per-row error and a fault is worse than a defined
// answer -- which is why the bounds are arguments.
// `vocab_off` is the first GLOBAL row this rank holds: the table is a vocab-parallel SLICE and an
// id outside it takes the zero-row arm, so summing the ranks reconstructs the embedding.  0 is the
// whole table, an unsliced lookup.
int r4d_embed_lookup_bf16(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                          int vocab_off, int x_ld, long stream);
int r4d_gather_rows_bf16(long x, long idx, long y, int M, int n, int src_rows,
                         int x_ld, int y_ld, long stream);
// y = x * rsqrt(mean(x^2) + eps) * (w + wadd).  `wadd` is GemmaRMSNorm's `1 +`, added in f32 after
// the bf16 load for the reason r4d_rmsnorm_had_quant_i8 gives.
// `w_f32` says whether the gain is f32 -- which every norm gain the architecture plugins declare
// is -- or bf16. The activation either side is bf16 in both cases.
int r4d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd, int w_f32,
                     int x_ld, int y_ld, long stream);
// `op` 0 sums, 1 multiplies.  A `b_ld` of 0 is the ONE broadcast form: every row reads the same n
// elements, which is what makes a bias add this op rather than a second one.
int r4d_binary_bf16(long a, long b, long y, int M, int n, int op,
                    int a_ld, int b_ld, int y_ld, long stream);
int r4d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
// y[.., i] = silu(gu[.., i]) * gu[.., n + i].  The GATE IS THE FIRST HALF -- the order rad-convert
// fuses gate_proj and up_proj in.
int r4d_silu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream);
// In place over `n_rope_heads` consecutive heads of a row; anything after them (the V heads of a
// fused projection) is untouched.  `gptj` 0 pairs i with i + rot/2 (NeoX, and mrope, which is the
// same arithmetic when every position component is equal); 1 pairs 2i with 2i+1.  `rot` may be less
// than head_dim -- the frequency exponent divides by `rot`, so the rotated prefix is a complete
// rotary of its own width.
int r4d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int gptj,
                  float theta, float rscale, int row_pitch, long stream);
// The multi-component rotary (docs/OPS.md, `rope`): `mode` 2 mrope, 3 imrope, 4 axial over
// `sec[0..n_sec)`, component c of token m at pos[c * plane + m]. plane 0 is the call above.
int r4d_rope_mc_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int gptj,
                     float theta, float rscale, int row_pitch, int mode, const int* sec, int n_sec,
                     int plane, long stream);
// C[M,N] f32 = A[M,K] bf16 @ W[N,K]^T bf16, f32 accumulate. THE OUTPUT IS f32 and that is why this
// is not r4d_gemm_bf16_nt_m16 under another name: the engine declares its logits buffer f32,
// because a bf16 logit has eight bits of mantissa and a nucleus cut compares neighbouring
// probabilities. K must be a multiple of 8.
int r4d_logits_gemm_bf16(long a, long w, long c, int M, int K, int N, int a_ld, int c_ld,
                         long stream);
// Scatter into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim]: K at [.., d], V at
// [.., head_dim + d], slot decomposing as block = slot / block_size, offset = slot % block_size.
int r4d_kv_store_bf16(long k, long v, long slot, long kv, int M, int head_dim, int kv_heads,
                      int block_size, int n_blocks, int k_ld, int v_ld, long stream);
// The same scatter into an E4M3 cache, which is what the fp8kv attention family reads. `k_scale`
// and `v_scale` are the DESCALES the reader multiplies back by, so this divides by them; they are
// one number per (sequence, head) on the read side, hence a caller constant here rather than
// anything derived from the tokens being written. 1.0 is vLLM's default and is what the attention
// kernels assume when their descale pointers are null. head_dim must be a multiple of 4.
int r4d_kv_store_fp8(long k, long v, long slot, long kv, int M, int head_dim, int kv_heads,
                     int block_size, int n_blocks, int k_ld, int v_ld,
                     float k_scale, float v_scale, long stream);
// The same scatter into a K4V4 (INT4) cache with in-register wave-local FWHT-256 on keys.
int r4d_kv_store_k4v4(long k, long v, long slot, long kv, int M, int head_dim, int kv_heads,
                      int block_size, int n_blocks, int k_ld, int v_ld, long stream);
// In-place wave-local FWHT-256 rotation for queries [M, heads, 256] with 1/16 scaling under KVarN.
int r4d_query_fwht256(long q, int64_t total_heads, long stream);

// ---- the decomposed sampler chain (r4d_sample_stages_f32.hip) -----------------------------------
//
// docs/OPS.md's eleven ops, one entry point each, as opposed to r4d_sample_chain_f32's fused form.
// Both exist and neither replaces the other: the fused one is right when the order is fixed and is
// what a speculative verify wants; these are what an engine that composes the chain per request
// needs, and they carry the stages the fused one has no room for -- penalties, DRY, a grammar
// bitmask, locally-typical, XTC.
//
// `params` is an array of abi/rad_sample.h's RadSampleParams, one row per sampled
// position.  This unit includes that header rather than restating the struct, so there is nothing
// for a caller to check the two layouts of.  Logits are f32.  A dropped candidate is index -1,
// value -infinity, and the buffers never shrink.
//
// The entry points are r4d_rad_sample_{temp,mask,penalties,dry,argmax,topk,topp,minp,typical,xtc,
// pick,merge_topk}, declared in libr4d/r4d_plugin.h with the plugin ABI's signature.  The
// flattened `(long lg, long params, int M, ...)` launch forms are private to
// r4d_sample_stages_f32.hip -- see the note at the entry points there.
//
// Exact top-k: a radix select over the monotone logit key, a second over the token id to break a
// tie at the threshold by the LOWER id, then a bitonic sort of the survivors -- descending, which
// every later stage assumes.  The optional `pairs` plane is the vocab-parallel path: (u32 global
// id, f32 logit) pairs with vocab_off already added, which is what the all-gather moves and what
// the merge reads.  The merge RE-APPLIES the row's top_k -- each rank ran its own before the
// gather, so the merged set holds up to world_size * k where the request asked for k, and nothing
// downstream narrows again.  At k = 1 that is the difference between an argmax and a draw.

}
