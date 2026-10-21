// The QSA indexer's BLOCK SCORE: how much one query wants each compressed block of key history.
//
//   score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
//
// The top `budget/ratio` of these per query is the set the attention then reads, so this is the
// whole of what makes Qwen3.8-Flash-Next's attention constant-cost past 2051 tokens rather than
// linear. docs/QSA.md is the decomposition; r4d_qsa_block_key_bf16.hip builds `bk`.
//
// THE RELU IS PER (BLOCK, HEAD) AND THE SUM IS OVER HEADS, in that order, and swapping them is a
// different model that still ranks blocks plausibly. The reference is `relu(scores).sum(dim=-1)`
// over a [n_blocks, heads] plane -- four independent votes, each of which can only ever argue FOR
// a block. Summing first would let one head's large negative cancel another's positive.
//
// `bt` IS A LIST OF PAGE IDS, NOT BLOCK INDICES, and that is the point of the whole design: the
// attention KV group is paged at `ratio` tokens, so a compress block and a page are the same
// thing, the block keys are stored per page, and the sequence's existing attention block table is
// already the list this scores. The top-k over the output is then a list of page ids, which is
// the block table the attention reads. Nothing is remapped anywhere.
//
// AN ABSENT ENTRY SCORES -inf RATHER THAN ZERO. A block table is padded to its declared width and
// a real score can be zero (every head vetoed the block), so zero would let padding outrank a
// live block the model chose to ignore.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kWave    = 32;
constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kMaxH    = 8;               // indexer query heads; 4 on this model
constexpr int kMaxVPL  = 8;               // n / kWave, so n <= 256
// One WAVE a block, and a workgroup covers a TILE OF QUERIES rather than one -- which is the
// whole of what makes this affordable at length.
//
// A block key is 256 bytes and every query in the step wants the same one. Scoring one query a
// workgroup reads the entire key history ONCE A QUERY ROW, and a speculative decode step is
// `1 + n_spec` rows -- at 200K of context that is dead linear in M, 88.3 us at M=1 and 298.6 at
// M=4. The keys are the only real traffic here, so holding a tile of queries in LDS and paying for
// them once is a straight divide by the tile.
//
// THE TILE IS FOUR BECAUSE A DECODE ROW-GROUP IS. Every sequence contributes exactly `1 + n_spec`
// query rows and `cu` is their cumulative count, so at the shipped depth of 3 a tile of 4 lands
// on a sequence boundary exactly. It is not required to: a tile that straddles two sequences is
// split at the boundary and the keys are read once per part, which is no worse than the one-query
// form does for every query anyway. The tile lives in registers -- see the scoring loop.
constexpr int kBPW     = 4;               // blocks a WAVE; a workgroup owns kWaves * kBPW = 32
constexpr int kQT      = 4;               // query rows a workgroup, and the grid's third axis

/* THE BUTTERFLY IS `r4d_wave_sum` AND ITS FIVE STEPS ARE SPELLED OUT BELOW, not called as one,
 * because the reduction is per HEAD and the five STEPS have to be hoisted out of the head loop:
 * a whole reduction inside it is `HEADS * 5` dependent lane exchanges with nothing to hide behind,
 * where five steps of `HEADS` independent ones is a quarter of the chain. Same lanes, same order,
 * same bits -- r4d_common.h has the measurement and why it is not on the LDS crossbar. */


// `m` IS A QUERY AND NOT A SEQUENCE, and `cu` is what tells the two apart. At decode they are the
// same thing up to the draft rows -- `1 + n_spec` queries a sequence -- but a prefill chunk is
// many queries over one history and every one of them selects its own set (docs/QSA.md, Stage B).
// The block table is the SEQUENCE's, so the scan of `cu` is how a query row finds its table row.
//
// A lane owns `n / 32` consecutive dims of both the query heads and the block key, so a head's
// dot product is one wave reduction and the block key is read once for all four heads AND all
// `kQT` queries -- which is why the head loop and the query loop are both inside the block loop.
//
// `nc` BOUNDS THE BLOCK LOOP AND IS NOT OPTIONAL. `nb` is the smaller of two DECLARED widths --
// the score plane's and the block table's pitch -- and both are sized for max_ctx, so scoring to
// `nb` on a 200-token prompt would read 65536 padding entries a query. The sequence's complete
// block count is an upper bound for every query in the chunk and is within `T / ratio` of each
// one's own; `qsa_select` applies the exact per-query bound, which it can because it has `pos`.
template <int VPL, int HEADS>
__global__ __launch_bounds__(kThreads) void qsa_score_kernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ bk,
    const int32_t* __restrict__ bt, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ nc, float* __restrict__ out,
    int n, int heads, int nb, int nseq, int q_ld, int bk_ld, int bt_ld, int out_ld,
    int m_offset) {

    __shared__ int sh_seq;
    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const float inv = rsqrtf((float)n);
    /* HOW WIDE A LANE MAY LOAD. A lane's dims are consecutive, so the widest vector that divides
     * both `vpl` and the row pitch is legal and the narrow cases still work -- a KV cache row
     * pitch is not required to be a multiple of anything. */
    const int kvec = ((VPL & 3) == 0 && (bk_ld & 3) == 0) ? 4
                   : ((VPL & 1) == 0 && (bk_ld & 1) == 0) ? 2 : 1;
    const int m0 = (int)blockIdx.y * kQT;

    /* The tile walks its queries a SEQUENCE at a time: everything below the query loop -- the
     * block table row, the complete-block count, the grid's block bound -- belongs to the
     * sequence and not to the query. At the shipped depth the loop runs once. */
    for (int m = m0; m < m0 + kQT; ) {
        /* The sequence of query `m`. Exactly one `i` can match, so the store is deterministic
         * however the threads are scheduled -- which matters because this tree compares generated
         * text across builds byte for byte. A query past the batch matches nothing and ends the
         * tile. */
        const int gm = m + m_offset;
        if (threadIdx.x == 0) sh_seq = -1;
        __syncthreads();
        for (int i = (int)threadIdx.x; i < nseq; i += kThreads)
            if (gm >= cu[i] && gm < cu[i + 1]) sh_seq = i;
        __syncthreads();
        const int seq = sh_seq;
        if (seq < 0) return;
        const int cu_end_local = cu[seq + 1] - m_offset;
        const int mend = cu_end_local < m0 + kQT ? cu_end_local : m0 + kQT;
        const int nq   = mend - m;

        const int nbs = nc[seq] < nb ? nc[seq] : nb;
        /* The block axis is sized from the DECLARED width, so most workgroups on a short context
         * have nothing to do; they skip here, before the query load. It is a SKIP and not a
         * return because the count is the sequence's: a tile straddling two of them can be past
         * the end of the first and inside the second. Every thread takes the same branch -- the
         * bound is uniform -- so the barriers below stay matched.
         *
         * THE SKIP STILL WAITS AT A BARRIER. The next part begins with thread 0 resetting
         * `sh_seq`, and a wave that has not yet read this part's value would read the reset and
         * leave the tile early, its blocks of the next sequence never scored. The barrier at the
         * bottom of the loop is what orders that on the scoring path; this is the same one for
         * the path that skips it, taken only when there is a next part to reset for. */
        const int b0 = (int)blockIdx.x * (kWaves * kBPW);
        if (nbs <= 0 || b0 >= nbs) {
            m = mend;
            if (m < m0 + kQT) __syncthreads();
            continue;
        }
        const int b1 = (b0 + kWaves * kBPW) < nbs ? (b0 + kWaves * kBPW) : nbs;

        /* THE QUERY TILE IN REGISTERS, loaded once for every block this workgroup owns. A lane
         * needs only its own VPL dims of each head of each query, and with the head count, the
         * dims a lane owns and the tile all compile-time that is kQT * HEADS * VPL registers --
         * 64 at this model's shape. Held in LDS instead, every block re-reads all of them, 64
         * `ds_read_b32` a block a lane, and LDS issue is what this kernel queues on. A query
         * past the tile's part reads the part's last row and is never scored. */
        float qv[kQT][HEADS][VPL];
#pragma unroll
        for (int mi = 0; mi < kQT; ++mi) {
            const int row = m + (mi < nq ? mi : nq - 1);
#pragma unroll
            for (int h = 0; h < HEADS; ++h) {
                const __hip_bfloat16* qp = q + (size_t)row * q_ld + (size_t)h * n + lane * VPL;
#pragma unroll
                for (int jj = 0; jj < VPL; ++jj) qv[mi][h][jj] = __bfloat162float(qp[jj]);
            }
        }

        /* ALL OF THIS WAVE'S BLOCKS ARE ASKED FOR BEFORE ANY IS SCORED. A block is two dependent
         * reads -- its page id out of the block table, then its key out of that page -- and taken
         * a block at a time a wave waits out both, kBPW times over: eight memory latencies in a
         * row, with the key history streaming at a third of what the part can deliver at 200K of
         * context. So every page id is read first, then every key, and only then does the scoring
         * begin. A block past the end or without a page reads page 0's key, which is always there,
         * and is skipped below. */
        int   pg[kBPW];
        float kvs[kBPW][VPL];
#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            const int b = b0 + wave + i * kWaves;
            pg[i] = b < b1 ? bt[(size_t)seq * bt_ld + b] : -1;
        }
#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            float* kv = kvs[i];
            /* ONE LOAD A LANE, NOT `vpl` OF THEM, and the difference is the whole bus. A lane owns
             * `vpl` CONSECUTIVE dims, so a scalar loop issues `vpl` two-byte loads each of which
             * spans the wave's entire 256-byte page: four cache lines touched to use 64 bytes of
             * them, four times over, which reads the key history at about 145 GB/s on a 621 GB/s
             * part. The vector form is the same bytes in the same order into the same registers --
             * only the instruction changes, so the sum below is bit-identical. */
            const uint16_t* kp = (const uint16_t*)bk + (size_t)(pg[i] >= 0 ? pg[i] : 0) * bk_ld
                               + lane * VPL;
            if (kvec == 4) {
#pragma unroll
                for (int j = 0; j < VPL; j += 4) {
                    const uint2 v = *(const uint2*)(kp + j);
                    kv[j + 0] = __uint_as_float((v.x & 0xFFFFu) << 16);
                    kv[j + 1] = __uint_as_float(v.x & 0xFFFF0000u);
                    kv[j + 2] = __uint_as_float((v.y & 0xFFFFu) << 16);
                    kv[j + 3] = __uint_as_float(v.y & 0xFFFF0000u);
                }
            } else if (kvec == 2) {
#pragma unroll
                for (int j = 0; j < VPL; j += 2) {
                    const uint32_t v = *(const uint32_t*)(kp + j);
                    kv[j + 0] = __uint_as_float((v & 0xFFFFu) << 16);
                    kv[j + 1] = __uint_as_float(v & 0xFFFF0000u);
                }
            } else {
#pragma unroll
                for (int j = 0; j < VPL; ++j)
                    kv[j] = __uint_as_float((uint32_t)kp[j] << 16);
            }
        }

#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            const int b = b0 + wave + i * kWaves;
            if (b >= b1) break;
            if (pg[i] < 0) {
                if (lane == 0)
                    for (int mi = 0; mi < nq; ++mi)
                        out[(size_t)(m + mi) * out_ld + b] = -INFINITY;
                continue;
            }
            const float* kv = kvs[i];
#pragma unroll
            for (int mi = 0; mi < kQT; ++mi) {
                if (mi >= nq) break;
                /* EVERY HEAD'S DOT FIRST, THEN ONE BUTTERFLY OVER ALL OF THEM. A wave reduction is
                 * five lane exchanges and each depends on the last, so a reduction per head inside
                 * the head loop is `HEADS * 5` exchanges in a single dependency chain with nothing
                 * to hide behind. Hoisting the five STEPS out leaves five dependent
                 * ones with `HEADS` independent shuffles inside each -- the same instructions in
                 * the same order on the same values, so the sum is bit-identical, and the chain
                 * is a quarter as long.
                 *
                 * `HEADS` is a template parameter because `dv[]` -- and `qv` above -- have to
                 * live in registers: indexed by a runtime bound they become scratch, and a spilled
                 * accumulator would cost more than the chain it is here to shorten. */
                float dv[HEADS];
#pragma unroll
                for (int h = 0; h < HEADS; ++h) {
                    float d = 0.0f;
#pragma unroll
                    for (int j = 0; j < VPL; ++j) d = fmaf(qv[mi][h][j], kv[j], d);
                    dv[h] = d;
                }
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] += swap16(dv[h]);       /* xor 16 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x168>(dv[h]);   /* xor 8 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x164>(dv[h]);   /* xor 4 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x162>(dv[h]);   /* xor 2 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x161>(dv[h]);   /* xor 1 */
                /* Every lane holds the reduced values, so the relu and the head sum are the same
                 * on all of them and only lane 0 stores. A partial reduction would be a score
                 * that depends on which lane asked. */
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < HEADS; ++h) s += fmaxf(dv[h], 0.0f);
                if (lane == 0) out[(size_t)(m + mi) * out_ld + b] = s * inv;
            }
        }
        m = mend;
        /* The next part of a straddling tile resets `sh_seq`, so nobody may still be reading it. */
        __syncthreads();
    }
}

/* ================================================================== the matrix form
 *
 * THE SAME SCORE AS A MATRIX PRODUCT, for a batch with many queries on one history -- a prefill
 * chunk. Per head the scores of 16 queries against 16 blocks are one 16x16 tile of
 * K_tile . Q_h^T, eight bf16 WMMA steps over the 128 dims, and the relu and the head sum then
 * happen in the accumulator registers: a lane holds one query and eight consecutive blocks of
 * every head's tile, so the four votes of a (query, block) pair are already side by side.
 *
 * WHY THE FORM ABOVE LOSES AT PREFILL. It is built for a decode step's four rows: a workgroup holds
 * four queries in registers and every block costs a five-step lane reduction per head. A 2048-row
 * chunk is 512 of those tiles, so the key history is read 512 times and the reductions outnumber
 * the multiplies; at 80K of context that is 9.3 ms a call, thirteen calls a step. Here the key
 * history is read once per 16 queries, there is no reduction at all, and the multiplies run on
 * the matrix units.
 *
 * AND IT WINS AT DECODE TOO, which is why it is the form for every batch it is built for and not
 * only for prefill. --perfqsa, one scattered history (us a call):
 *
 *      ctx       M=1            M=16            M=32            M=2048
 *      2000      5.28 -> 5.54   6.38 -> 5.50    8.03 -> 5.50    276 -> 17.4
 *      80000     25.0 -> 7.9 (M=4)                              10849 -> 535
 *      200000    29.9 -> 12.7   220 -> 16.2     437 -> 27.2     27711 -> 1356
 *
 * The scalar form's cost is linear in the rows -- a four-row tile per workgroup reads the history
 * again for every four rows -- so eight sequences at depth 3 and 200K of context cost 5.7 ms a
 * step in this op across its thirteen calls. The one row it still wins is a single query at a short
 * context, by a quarter of a microsecond.
 *
 * ONE FORM FOR EVERY BATCH ALSO KEEPS A SCORE A FUNCTION OF ITS QUERY. The two forms multiply the
 * same bf16 pairs exactly in f32 but sum them in different orders -- the WMMA's internal order over
 * sixteen dims and then eight steps, against a lane's four-dim chain and a butterfly -- so they
 * differ in the last bit (rel_l2 ~4e-8 between them), and a selection at an exact tie could differ
 * with it. Choosing the form by batch size would make a query's selection depend on what else was
 * in its step; within one form, which rows share a tile does not change a row's score.
 *
 * ONE WAVE OWNS 16 QUERY ROWS AND A RANGE OF `tpw` BLOCK TILES. The query rows are loaded once,
 * into registers -- HEADS x 8 fragments, 128 VGPRs at four heads -- and the wave then streams its
 * block tiles past them, one page-id read and eight fragment loads a tile. A workgroup is four
 * waves on four consecutive block ranges of the same query rows, so the rows they share are
 * fetched once into the WGP's cache. There is no LDS and no barrier.
 *
 * ROWS OF TWO SEQUENCES IN ONE TILE ARE SCORED A SEQUENCE AT A TIME, exactly as the scalar form
 * does it: each part walks its own sequence's table, and only the rows of that part are stored.
 * A tile of decode rows beside a prefill chunk is the only place that happens. */
constexpr int kMxWaves = 4;     // waves a workgroup, each on its own block range
constexpr int kMxQ     = 16;    // query rows a wave: the WMMA tile's N
constexpr int kMxKS    = 8;     // 16-wide K steps: n = 128, the only width this form is built for
/* A wave's block range: enough of them that the grid holds about this many waves, which is a
 * couple of rounds of the part's resident capacity (64 CUs x 4 SIMDs x ~6 waves at this form's
 * register count) ... */
constexpr long long kMxWavesTarget = 4096;
/* ... and never fewer tiles than this, because a wave loads its sixteen query rows -- 16 KiB at
 * four heads -- before its first tile, and at one tile a wave that load is most of what it reads. */
constexpr long long kMxMinTiles    = 4;

template <int HEADS>
__global__ __launch_bounds__(kMxWaves * kWave) void qsa_score_mx_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ bk,
    const int32_t* __restrict__ bt, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ nc, float* __restrict__ out,
    int M, int nb, int nseq, int tpw, int q_ld, int bk_ld, int bt_ld, int out_ld, int vec,
    int m_offset) {

    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int m0 = (int)blockIdx.y * kMxQ;
    const int wb0 = ((int)blockIdx.x * kMxWaves + wave) * tpw * 16;
    if (m0 >= M || wb0 >= nb) return;       /* wave-uniform, and nothing below synchronises */
    const float inv = rsqrtf((float)(kMxKS * 16));

    /* The B operand: query row `col` of the tile, every head, every K step. A row past the batch
     * reads the last one and is never stored. */
    v8bf qf[HEADS][kMxKS];
    {
        const int qr = m0 + col < M ? m0 + col : M - 1;
        const unsigned short* qp = q + (size_t)qr * q_ld + 4 * half;
#pragma unroll
        for (int h = 0; h < HEADS; ++h)
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) qf[h][ks] = fragP(qp + h * kMxKS * 16 + ks * 16);
    }

    const int mlim = m0 + kMxQ < M ? m0 + kMxQ : M;
    const int row = m0 + col;
    for (int m = m0; m < mlim; ) {
        /* The sequence of query `m`: exactly one entry of `cu` brackets it. */
        const int gm = m + m_offset;
        int seq = -1;
        for (int i0 = 0; i0 < nseq; i0 += kWave) {
            const int i = i0 + lane;
            const unsigned long long hit = __ballot(i < nseq && gm >= cu[i] && gm < cu[i + 1]);
            if (hit) { seq = i0 + __ffsll((long long)hit) - 1; break; }
        }
        if (seq < 0) return;                /* past the batch: so is every later row */
        const int cu_end_local = cu[seq + 1] - m_offset;
        const int mend = cu_end_local < mlim ? cu_end_local : mlim;
        const bool live = row >= m && row < mend;
        const int nbs  = nc[seq] < nb ? nc[seq] : nb;
        const int bend = wb0 + tpw * 16 < nbs ? wb0 + tpw * 16 : nbs;
        const int32_t* tab = bt + (size_t)seq * bt_ld;

        for (int b0 = wb0; b0 < bend; b0 += 16) {
            /* The A operand: block `b0 + col`'s key, one page-id read and eight fragment loads. A
             * block past the end or without a page reads page 0's key, which always exists, and
             * is overwritten or skipped at the store. */
            const int b  = b0 + col;
            const int pg = b < bend ? tab[b] : -1;
            const unsigned short* kp = bk + (size_t)(pg >= 0 ? pg : 0) * bk_ld + 4 * half;
            v8bf kf[kMxKS];
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) kf[ks] = fragP(kp + ks * 16);
            /* Which of the tile's sixteen blocks have no page: lanes 0-15 each read one. */
            const unsigned absent = (unsigned)__ballot(pg < 0) & 0xFFFFu;

            v8f acc[HEADS];
#pragma unroll
            for (int h = 0; h < HEADS; ++h)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[h][e] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks)
#pragma unroll
                for (int h = 0; h < HEADS; ++h) acc[h] = mma(kf[ks], qf[h][ks], acc[h]);

            /* D[block][query]: this lane holds query `col` against blocks b0 + 8*half + e. */
            if (!live) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < HEADS; ++h) s += fmaxf(acc[h][e], 0.0f);
                v[e] = ((absent >> (8 * half + e)) & 1u) ? -INFINITY : s * inv;
            }
            float* op = out + (size_t)row * out_ld + b0 + 8 * half;
            if (vec && b0 + 16 <= bend) {
                *(float4*)op       = make_float4(v[0], v[1], v[2], v[3]);
                *(float4*)(op + 4) = make_float4(v[4], v[5], v[6], v[7]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (b0 + 8 * half + e < bend) op[e] = v[e];
            }
        }
        m = mend;
    }
}

/* ================================================================== the packed matrix form
 *
 * PACKED MATRIX FORM FOR DECODE SHAPES:
 * 4 queries x 4 heads packed into the 16 columns of a single WMMA tile (col = 4*h + q_idx).
 *
 * Sequences are strictly segregated: blockIdx.y maps to sequence s.
 * Matrix A is loaded ONLY from sequence s's block table, and Matrix B is loaded ONLY
 * from sequence s's queries (q_start = cu[s], count <= 4).
 * This eliminates all cross-sequence contamination and ballot searching.
 *
 * Within the wave, the 4 heads are reduced into lanes 0..3 via two DPP instructions:
 *   s = r4d_dpp_xor_add<0x168>(s);  // XOR 8
 *   s = r4d_dpp_xor_add<0x164>(s);  // XOR 4
 * Each lane col < q_cnt then writes its accumulated score directly to out[q0 + col].
 * Instruction count drops by 4x (8 vs 32 WMMAs per tile), VGPR pressure drops by 4x (32 vs 128 VGPRs).
 */
__global__ __launch_bounds__(kMxWaves * kWave) void qsa_score_packed_mx_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ bk,
    const int32_t* __restrict__ bt, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ nc, float* __restrict__ out,
    int M, int nb, int nseq, int tpw, int q_ld, int bk_ld, int bt_ld, int out_ld, int vec,
    int m_offset) {

    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int seq = (int)blockIdx.y;
    if (seq >= nseq) return;

    const int q_start_all = cu[seq] - m_offset;
    const int q_end_all   = cu[seq + 1] - m_offset;
    const int q_start = q_start_all < 0 ? 0 : q_start_all;
    const int q_end   = q_end_all > M ? M : q_end_all;
    if (q_start >= q_end) return;

    const int nbs  = nc[seq] < nb ? nc[seq] : nb;
    const int wb0  = ((int)blockIdx.x * kMxWaves + wave) * tpw * 16;
    if (wb0 >= nbs) return;
    const int bend = wb0 + tpw * 16 < nbs ? wb0 + tpw * 16 : nbs;

    const float inv = rsqrtf((float)(kMxKS * 16));
    const int32_t* tab = bt + (size_t)seq * bt_ld;

    for (int q0 = q_start; q0 < q_end; q0 += 4) {
        const int q_cnt = q_end - q0 < 4 ? q_end - q0 : 4;

        /* Operand B: 4 queries x 4 heads in 16 columns.
         * head = col >> 2 (0..3), q_idx = col & 3 (0..3) */
        const int head  = col >> 2;
        const int q_idx = col & 3;
        const bool q_live = (q_idx < q_cnt);
        const int qr = q_live ? (q0 + q_idx) : q0;
        const unsigned short* qp = q + (size_t)qr * q_ld + (size_t)head * (kMxKS * 16) + 4 * half;
        v8bf qf[kMxKS];
#pragma unroll
        for (int ks = 0; ks < kMxKS; ++ks) qf[ks] = fragP(qp + ks * 16);

        for (int b0 = wb0; b0 < bend; b0 += 16) {
            const int b  = b0 + col;
            const int pg = b < bend ? tab[b] : -1;
            const unsigned short* kp = bk + (size_t)(pg >= 0 ? pg : 0) * bk_ld + 4 * half;
            v8bf kf[kMxKS];
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) kf[ks] = fragP(kp + ks * 16);

            const unsigned absent = (unsigned)__ballot(pg < 0) & 0xFFFFu;

            v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.0f;

#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) acc = mma(kf[ks], qf[ks], acc);

            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = fmaxf(acc[e], 0.0f);
                s = r4d_dpp_xor_add<0x168>(s);
                s = r4d_dpp_xor_add<0x164>(s);
                v[e] = ((absent >> (8 * half + e)) & 1u) ? -INFINITY : s * inv;
            }

            if (col < q_cnt) {
                const int q_row = q0 + col;
                float* op = out + (size_t)q_row * out_ld + b0 + 8 * half;
                if (vec && b0 + 16 <= bend) {
                    *(float4*)op       = make_float4(v[0], v[1], v[2], v[3]);
                    *(float4*)(op + 4) = make_float4(v[4], v[5], v[6], v[7]);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        if (b0 + 8 * half + e < bend) op[e] = v[e];
                }
            }
        }
    }
}

/* WHICH FORM A LAUNCH TAKES: 0 is the rule below, 1 pins the scalar form, 2 the matrix form,
 * 3 the packed matrix form. A test hook, like r4d_gemm_fp8a8_pin_mw. */
int g_qsa_form = 0;

}  // namespace

extern "C" void r4d_qsa_score_pin_form(int form) { g_qsa_form = form; }

//   q    [M][heads*n]    bf16, `q_ld` elements a row (0 = heads*n)      -- M is QUERIES
//   bk   [pages][n]      bf16, `bk_ld` elements a row (0 = n)
//   bt   [nseq][nb]      i32, `bt_ld` elements a row (0 = nb); negative = no block
//   cu   [nseq+1]        i32, cumulative query counts: which sequence a query row belongs to
//   nc   [nseq]          i32, complete blocks a sequence; the block loop's bound
//   out  [M][nb]         f32, `out_ld` elements a row (0 = nb)
static int r4d_qsa_score_bf16_emit(long q, long bk, long bt, long cu, long nc, long out,
                                  int M, int n, int heads, int nb, int nseq,
                                  int q_ld, int bk_ld, int bt_ld, int out_ld, const R4dSink& sk,
                                  int m_offset = 0) {
    int emit_rc = RAD_OK;
    if (!q || !bk || !bt || !cu || !nc || !out) return -1;
    if (M <= 0 || n <= 0 || heads <= 0 || nb <= 0 || nseq <= 0) return -2;
    if (n % kWave || n / kWave > kMaxVPL) return -3;   // a lane owns n/32 dims, at most 8
    if (heads > kMaxH) return -5;
    if (!q_ld) q_ld = heads * n;
    if (!bk_ld) bk_ld = n;
    if (!bt_ld) bt_ld = nb;
    if (!out_ld) out_ld = nb;
    /* The block axis is sized from the DECLARED width because the live count is device data; a
     * workgroup past the live end exits on its first compare, which is what `nb` inside the kernel
     * is bounded to. The QUERY axis is tiled the same way and for the same reason: a tile whose
     * first row is past the batch ends on its `cu` scan. */
    const bool mx_fits = n == kMxKS * 16 && heads <= 4;
    const bool mx_packed_fits = n == kMxKS * 16 && heads == 4 && (M <= 4 * nseq);
    if (g_qsa_form == 3 && !mx_packed_fits) return -3;
    if (g_qsa_form == 2 && !mx_fits) return -3;
    const bool use_packed = (g_qsa_form == 3) || (g_qsa_form == 0 && mx_packed_fits);
    if (use_packed) {
        const long long tiles = (nb + 15) / 16;
        long long tpw = (tiles * nseq + kMxWavesTarget - 1) / kMxWavesTarget;
        if (tpw < kMxMinTiles) tpw = kMxMinTiles;
        const unsigned gx = (unsigned)((tiles + tpw * kMxWaves - 1) / (tpw * kMxWaves));
        const int vec = (out_ld % 4 == 0) && ((out & 15) == 0);
        emit_rc = R4D_EMIT(sk, qsa_score_packed_mx_kernel, dim3(gx, (unsigned)nseq), dim3(kMxWaves * kWave),
                           0, (const unsigned short*)q, (const unsigned short*)bk,
                           (const int32_t*)bt, (const int32_t*)cu, (const int32_t*)nc,
                           (float*)out, M, nb, nseq, (int)tpw, q_ld, bk_ld, bt_ld, out_ld, vec, m_offset);
        return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
    }
    const bool mx = g_qsa_form == 1 ? false : mx_fits;
    if (mx) {
        /* THE BLOCK RANGE A WAVE OWNS is sized so the grid holds about kMxWavesTarget waves, and
         * never less than kMxMinTiles tiles, below which loading the query rows costs more than
         * the tiles they are used for. */
        const long long qt = (M + kMxQ - 1) / kMxQ, tiles = (nb + 15) / 16;
        long long tpw = (tiles * qt + kMxWavesTarget - 1) / kMxWavesTarget;
        if (tpw < kMxMinTiles) tpw = kMxMinTiles;
        const unsigned gx = (unsigned)((tiles + tpw * kMxWaves - 1) / (tpw * kMxWaves));
        const int vec = (out_ld % 4 == 0) && ((out & 15) == 0);
#define RAD_QSA_MX(H)                                                                             \
        emit_rc = R4D_EMIT(sk, (qsa_score_mx_kernel<H>), dim3(gx, (unsigned)qt), dim3(kMxWaves * kWave), \
                           0, (const unsigned short*)q, (const unsigned short*)bk,                \
                           (const int32_t*)bt, (const int32_t*)cu, (const int32_t*)nc,            \
                           (float*)out, M, nb, nseq, (int)tpw, q_ld, bk_ld, bt_ld, out_ld, vec, m_offset)
        switch (heads) {
            case 1: RAD_QSA_MX(1); break;
            case 2: RAD_QSA_MX(2); break;
            case 3: RAD_QSA_MX(3); break;
            case 4: RAD_QSA_MX(4); break;
            default: return -5;
        }
#undef RAD_QSA_MX
        return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
    }
    const unsigned nbg = (unsigned)((nb + kWaves * kBPW - 1) / (kWaves * kBPW));
    const unsigned nmg = (unsigned)((M + kQT - 1) / kQT);
    /* THE DIMS A LANE OWNS ARE A TEMPLATE PARAMETER, and that is not tidiness. `n / 32` is a
     * runtime value, so as a function argument the dot is a ROLLED loop -- the whole kernel
     * disassembles to ONE `v_fmac_f32` with a dependent `ds_load_b32` in front of it, four times a
     * head, and the block key sits in a runtime-indexed array the compiler cannot keep in
     * registers. Fixing the count at compile time unrolls both and turns the dot into register
     * FMAs. Four
     * instantiations cover every `n` and every head count the op's own checks admit
     * (n % 32 == 0, n <= 256, heads <= 8). */
#define RAD_QSA_GO(V, H)                                                                       \
    emit_rc = R4D_EMIT(sk, (qsa_score_kernel<V, H>), dim3(nbg, nmg), dim3(kThreads), 0,              \
                       (const __hip_bfloat16*)q, (const __hip_bfloat16*)bk, (const int32_t*)bt, \
                       (const int32_t*)cu, (const int32_t*)nc,                                  \
                       (float*)out, n, heads, nb, nseq, q_ld, bk_ld, bt_ld, out_ld, m_offset)
#define RAD_QSA_HEADS(V)                                        \
    switch (heads) {                                            \
        case 1: RAD_QSA_GO(V, 1); break;                        \
        case 2: RAD_QSA_GO(V, 2); break;                        \
        case 3: RAD_QSA_GO(V, 3); break;                        \
        case 4: RAD_QSA_GO(V, 4); break;                        \
        case 5: RAD_QSA_GO(V, 5); break;                        \
        case 6: RAD_QSA_GO(V, 6); break;                        \
        case 7: RAD_QSA_GO(V, 7); break;                        \
        case 8: RAD_QSA_GO(V, 8); break;                        \
        default: return -5;                                     \
    }
    switch (n / kWave) {
        case 1: RAD_QSA_HEADS(1); break;
        case 2: RAD_QSA_HEADS(2); break;
        case 3: RAD_QSA_HEADS(3); break;
        case 4: RAD_QSA_HEADS(4); break;
        case 5: RAD_QSA_HEADS(5); break;
        case 6: RAD_QSA_HEADS(6); break;
        case 7: RAD_QSA_HEADS(7); break;
        case 8: RAD_QSA_HEADS(8); break;
        default: return -3;
    }
#undef RAD_QSA_HEADS
#undef RAD_QSA_GO
    return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
}

extern "C" int r4d_qsa_score_bf16_emit(long q, long bk, long bt, long cu, long nc, long out,
                                  int M, int n, int heads, int nb, int nseq,
                                  int q_ld, int bk_ld, int bt_ld, int out_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return r4d_qsa_score_bf16_emit(q, bk, bt, cu, nc, out, M, n, heads, nb, nseq, q_ld, bk_ld, bt_ld, out_ld, sk);
}

// ============================================================== the plugin ABI entry point

enum { QS_Q = 0, QS_BK = 1, QS_BT = 2, QS_CU = 3, QS_NC = 4, QS_OUT = 5 };

static int r4d_rad_qsa_score_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* q  = r4d_opt(a, QS_Q);
    const RadTensor* bk = r4d_opt(a, QS_BK);
    const RadTensor* bt = r4d_opt(a, QS_BT);
    const RadTensor* cu = r4d_opt(a, QS_CU);
    const RadTensor* nc = r4d_opt(a, QS_NC);
    const RadTensor* o  = r4d_opt(a, QS_OUT);
    if (!q || !bk || !bt || !cu || !nc || !o) return R4D_NO(RAD_E_INVAL);
    /* `bk` MAY BE A KV CACHE AND IS THEN RANK FOUR -- see r4d_qsa_block_key_bf16.hip on why the
     * block-key store's geometry is a container. A page holds `n` contiguous values either way. */
    if (bk->rank < 2 || bt->rank != 2 || o->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long heads = r4d_geti_or(a, "heads", 0);
    if (n <= 0 || heads <= 0) return R4D_NO(RAD_E_INVAL);
    const long long M = o->shape[0];
    /* HOW MANY BLOCKS TO SCORE IS WHAT BOTH SIDES CAN NAME, and taking the smaller is safe rather
     * than approximate: a query's real block count is bounded by max_ctx/ratio and both the table
     * and the score plane are sized for that, so the minimum is never below the live count. It is
     * the MINIMUM and not one of them because the two come from different places -- the score
     * plane is the architecture's buffer and the table's pitch is the KV manager's, and requiring
     * them equal would refuse a configuration where they merely round differently. `qsa_select`
     * bounds the range it reads by `nc` regardless, so a score past the live count is never looked
     * at. */
    const long long nb = bt->shape[1] < o->shape[1] ? bt->shape[1] : o->shape[1];
    if (M < 1 || nb < 1) return R4D_NO(RAD_E_SHAPE);
    /* THE SEQUENCE COUNT COMES OFF `cu` AND NOT OFF THE TABLE'S HEIGHT, which is max_seqs -- the
     * same rule qsa_work and qsa_tail_store follow. The table must be at least that tall. */
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq) return R4D_NO(RAD_E_SHAPE);
    if (bk->shape[0] < 1 || r4d_numel(bk) / bk->shape[0] != n) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) < M * heads * n) return R4D_NO(RAD_E_SHAPE);
    const long long qp = r4d_row_pitch(q), kp = r4d_row_pitch(bk);
    const long long tp = r4d_row_pitch(bt), op = r4d_row_pitch(o);
    if (qp < 0 || kp < 0 || tp < 0 || op < 0) return R4D_NO(RAD_E_STRIDE);
    long long m_offset = r4d_geti_or(a, "m_offset", 0);
    const long long tp_part = r4d_geti_or(a, "tp_partition", 0);
    if (tp_part && a->world_size == 2 && a->rank == 1) {
        m_offset = M;
    }
    const int e = r4d_qsa_score_bf16_emit(r4d_addr(q), r4d_addr(bk), r4d_addr(bt), r4d_addr(cu),
                                     r4d_addr(nc), r4d_addr(o),
                                     (int)M, (int)n, (int)heads, (int)nb, (int)nseq,
                                     (int)qp, (int)kp, (int)tp, (int)op, sk, (int)m_offset);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

R4D_ENTRY(r4d_rad_qsa_score, r4d_rad_qsa_score_body)
