// BLOCK-SCALED FP8 weight, fp8 activation. Two entry points from one translation unit, because
// they are one K loop with two tile shapes and keeping two copies of this much staging code is how
// they drift:
//
//   r4d_gemm_fp8a8_nt_m16   M <= 16, the decode and speculative-verify band. One 16-wide token
//                           tile, split-K over the caller's scratch.
//   r4d_gemm_fp8a8_tiled    any M, the prefill band. 128x128 tile, eight waves.
//
// WHY THE LIBRARY NEEDS IT. Every other quantised GEMM here reads a weight this library's own
// quantiser produced. This one reads the weight A CHECKPOINT SHIPS: Qwen3.8-27B-FP8 stores
// `<linear>.weight` as F8_E4M3 [N][K] with a companion `<linear>.weight_scale_inv` in BF16 on a
// [ceil(N/128)][ceil(K/128)] grid -- `quantization_config.weight_block_size = [128, 128]`,
// DeepSeek-style block scaling. Without an entry point that reads it, serving that checkpoint means
// dequantising it to bf16 at conversion, which is twice the weight bytes of a 27B model across two
// 32 GB cards. There is no repacking and no scale conversion: the bytes in the safetensors file are
// the bytes these kernels read.
//
// Four properties of the K loop, each marked where it happens:
//
//   1. THE SCALE DTYPE. The checkpoint carries a bf16 scale per 128x128 tile; it decodes as
//      `__uint_as_float(s << 16)` -- one shift and one bitcast. See r4d_fp8_wmma.h.
//
//   2. THE OPERAND ROLES ARE A TEMPLATE PARAMETER. With the WEIGHT on the A operand the kernel
//      computes Y[rows][nb], which is C^T for this library's `C[M,N] = A[M,K] @ W[N,K]^T`
//      convention. See the `AW` comment on the body.
//
//   3. SPLIT-K USES THE CALLER'S SCRATCH. Nothing on an R4D launch path allocates, so the partial
//      plane is an argument and `r4d_gemm_fp8a8_nt_m16_scratch()` sizes it. The summation order is
//      fixed.
//
//   4. bf16 C, NOT f32. Every GEMM in this library writes bf16 and its consumers expect that. At
//      prefill's 1536x17408 that is 53 MB a store instead of 107.
//
// LOAD-BEARING: the padded LDS row pitch (r4d_fp8_wmma.h, `lds_pitch`), the one-K-tile-ahead
// register prefetch, and the two accumulator sets. The reason for each is at the line.
//
// THE STAGING ASSIGNMENT. Two threads a row at every tile width (`tid >> 1`) costs a third of this
// card's read bandwidth on the narrow tile, which is the whole of decode. The assignment used, and
// the measurement behind it, are at `kTPR` in the body.
//
// THIS ENTRY POINT RETURNS A NEGATIVE CODE rather than throwing. It is reachable from radiance's
// kernel plugin, where an error is a value and an exception crossing an .so boundary is the
// failure mode that boundary exists to avoid.

#include "r4d.h"
#include "r4d_fp8_args.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace {

using namespace r4d_fp8;

constexpr uint32_t kWave = 32;

// THE BLOCK, on both axes. One constant and not two because the checkpoint's scale grid, the
// activation quantiser's group (r4d_quant_act_fp8) and this fold must agree, and a disagreement is
// silently wrong output rather than a link error. It is NOT a build flag like R4D_GEMM_W4A8_GROUP:
// that group is a choice this library's quantiser makes, and this one is a property of the file on
// disk.
constexpr uint32_t kBlock = 128;

// kFN 128 and 8 waves. A block re-reads its whole row band of the weight once per M tile, so weight
// traffic is `N*K * ceil(M/FM)`, and at M = 512 that is four passes over the weight -- which LOOKS
// like the wall and is not one. Two things say so: a shape whose whole weight fits in cache
// (N=1024, 5 MB) runs at the same TFLOP/s as one that cannot (N=34816, 178 MB), and doubling FM,
// which halves that traffic outright, is SLOWER. The four passes are absorbed by the 64 MB MALL,
// because the grid is x-fastest and the M tiles of one N band are dispatched together.
//
// What the tile really loses to is LDS BANK CONFLICTS on the fragment read, and that is fixed in
// r4d_fp8_wmma.h's row pitch rather than here -- see the bank arithmetic at `lds_pitch`.
//
// Past that, removing each remaining per-iteration cost in turn (arithmetically wrong but
// timing-valid) shows where the rest of the time is. The four that matter are the
// per-128-column fold, the global fetch, seven of the eight B fragment reads and the fold's 8-way
// per-lane activation-scale gather; each is worth 4-13% on its own and all four together are worth
// about half. So with EVERY per-iteration cost deleted this loop shape reaches about 65% of the
// rate the WMMA issues at on its own -- the gap from where it runs to that ceiling is not one
// thing. Roughly a third of it is these four, and the rest is the loop itself: the barrier eight
// waves rendezvous on once per K tile, the staging stores, the loop arithmetic.
constexpr uint32_t kFN = 128, kFK = 64;   // kFK is BYTES per LDS row
// One gfx1201 cacheline -- 256 B, not 128. This is the tile width for the narrow instantiation and
// it carries a second job: `kTPR` is FK/32, so a 256-byte tile puts EIGHT threads on a row and
// that is what makes the weight stream coalesce. 64 would put two there and cost a third of the
// bandwidth. 128 (four threads) is also full speed; 512 does not build at all, the eight staged
// granule pairs a thread being 64 VGPRs of prefetch on top of the accumulators.
//
// The prefill tile stays at 64 rather than following: it is not fetch-bound, and widening it is
// slower at every width, because its LDS tile is 128 rows and 256 bytes of that is 32 KiB of A
// alone.
constexpr uint32_t kFKNarrow = 256;
constexpr uint32_t kFWaves = 8;
// 1 M tile x 8 N tiles a wave, so the 8 waves cover 8 DISTINCT row tiles.
//
// The register tile is the two of these together and it is a genuine knob -- the body takes it as a
// template parameter so the narrow instantiation can keep 1x1 while this one is set independently.
// EIGHT ACCUMULATOR TILES IS A HARD CAP: a tile is two f32 accumulator sets, so MT*NT past 8 is
// past 128 VGPRs of accumulator and the spill is immediate. 1x8 and 2x4 are the two shapes at the
// cap and they tie; 2x8 spills 233 bytes and runs at a third the rate, 4x4 and 4x8 spill thousands
// and collapse.
//
// 1x4 ANSWERS THE RESIDENCY QUESTION. FN = 64 takes the LDS from
// 36868 B to 27652, which is FOUR blocks in a WGP's 128 KiB rather than three, and the accumulators
// from 128 registers to 64, which takes the kernel from 203 VGPRs to 136. It is about 20% SLOWER.
// So the fourth block does not pay for halving the work a wave does, and residency joins iteration
// size and instruction count on the list of things that are not what binds this loop.
//
// 2x4 IS THE INTERESTING ONE and the wave grid has to be held fixed to read it. Reshaping the
// register tile alone changes the block tile too (FN = NT*16), so a plain 2x4 is a 128x64 block and
// not a squarer wave. With a second wave axis -- grid (MW/NW) x NW, so 8 waves as 4x2 hold the SAME
// 128x128 block with 2x4 a wave -- the fragment reads fall from 9 per 8 WMMA to 6, a third off the
// LDS read traffic, and it is a WASH.
//
// That is why the second wave axis is not in the code: the fragment reads, which do not
// bank-conflict, are not the wall, and the ablation agrees -- removing SEVEN of the nine is worth
// ~10%, so removing three is worth about nothing.
//
// Also slower or neutral: widening the tile so a whole 128-column scale block is one iteration
// (which makes the fold position compile-time and deletes 39 SALU and four guarded fold sites) --
// the bigger scheduling window costs 44 VGPRs and two waves a SIMD, and loses; the same via a
// maintained tile counter rather than an unroll, also a loss; clearing `raw` by feeding a zero
// accumulator to the first WMMA of a block instead of 64 v_mov in the fold, neutral; explicit
// `sched_group_barrier` interleavings of DS reads against WMMA, neutral to slightly worse; and an
// explicit one-slot software pipeline of the fragment reads, slightly worse, the compiler having
// already done it.
//
// And one that r4d_common.h's lds_barrier() makes look free and is not: swapping this loop's
// __syncthreads() for the LDS-scoped fence the attention kernels use costs a few percent on both
// gate_up and ffn_down. The vector-cache invalidate it removes is worth having here, which is the
// opposite of what it is worth in an attention kernel streaming KV.
//
// ---- ...AND THE WAVE GRID CANNOT BE TRANSPOSED TO REACH A WIDER TILE, AND IT IS THE ------------
// ---- ACTIVATION SCALE THAT SAYS SO --------------------------------------------------------------
//
// The cap above bounds MT*NT a WAVE, not the block, so the obvious way round it is to put the wave
// count on FN instead of FM: a 1xMW grid with MT = 4, NT = 2 gives a 64x256 block at the same eight
// tiles a wave as this 128x128 does at 1x8. That shape is what a SwiGLU epilogue needs here -- 256
// weight columns is 128 gate and 128 up, so a block would emit exactly one 128-column scale group,
// which is what the narrow tile's `GATED` epilogue does. That grid loses badly.
//
// THE GRID IS THE COST. Running the SAME 128x128 tile with the same eight tiles a wave on a
// 1xMW grid -- only which wave owns which quarter of it changes -- is about FOUR TIMES slower than
// the MWx1 grid, at every shape. So the cost is not the tile shape and no tile shape reachable this
// way can pay for it.
//
// WHY, and it is the reason the activation is on A in the first place: the AW=false scale fold
// below builds `as[MT][8]`, one f32 per activation ROW the wave owns, because an activation carries
// a scale per token. MT is exactly what a transposed grid has to raise -- 1 -> 8 for the same tile,
// 1 -> 4 for the wide one -- and the cost scales with it: MT=8 is about twice MT=4's time, and
// 64 against 32 live f32 on top of 128 VGPRs of accumulator. The MWx1 grid exists so
// that a wave owns ONE row tile and eight column tiles, and the weight's scale, which is one
// scalar for 128 columns, is the operand that tolerates being spread.
//
// So the epilogue fold has a decode half and no prefill half, and the reason is a register file
// rather than a fixable oversight. See GATED on the body.
constexpr uint32_t kFMTiles = 1, kFNTiles = 8;

// Blocks the split is aiming for. A TUNING DEFAULT, deliberately NOT derived from the device's CU
// count: `ksplit` sets the ORDER the K partials are summed in, so deriving it would make the
// emitted text a function of which card the binary is on.
//
// `gx * mw` is N/16 whatever MW is, so this rule is a function of N alone -- which is why the
// target has to be fitted against the shapes whose GRIDS fall near it and not only against the
// largest and the smallest. A target of 512 leaves the attention projections (4096 and 5120 output
// rows, a third of decode's GEMM time) running at about 60% of what the same kernel does when it is
// given enough blocks, and one 2-way split recovers it. A target of 256 leaves the mid-sized
// projections (3072 and 2048 rows, a fifth of the decode GEMM) short in the same way, by 3-12%.
// Past 8192 rows the grid is already big and splitting costs, which is why this is a target and not
// a multiplier: the guard below is what stops it there.
//
// AND IT IS 128 ACROSS THE WHOLE BAND, not only at narrow M. At M = 64 a larger target makes trunk
// shapes split that do not at 128, and every one of them loses 30-70% by it. The reason is the one
// the narrow-M reading gives from the other side: the split's cost is a partial plane of M*N f32,
// which at 64 rows is EIGHT TIMES what it is at 8, while the parallelism it buys is unchanged. A
// wider band wants a LOWER target, not a higher one, and 128 is already below every grid a trunk
// shape produces. Do not re-sweep it.
constexpr uint32_t kSplitTarget = 128u;
/* AND IT DOES NOT WANT TO FALL PAST EIGHT ROWS, although isolation says it does. The split's
 * BENEFIT is parallelism and its COST is the reduction -- `ksplit * M * N` floats written and read
 * back -- and only the cost scales with M, so a target expressed as a tile count is blind to the
 * row count. `gx * mw` is N/16 whatever the wave count is, so this decides only for N < 16*tgt,
 * and AMONG THE SHAPES THIS MODEL ISSUES THE ONLY ONE THAT SPLITS AT ALL is N = 512, the attention
 * k and v projections, fifty-two calls a step. At N = 512 the target picks ksplit 4 (128), 3 (96)
 * or 2 (64). Best of seven, us, --perf 512 5120 M:
 *
 *     M            8     16     24     32     40     48     56     64
 *     tgt  64   14.1   15.3   18.3   20.1   26.0   26.1   29.2   30.0
 *     tgt  96   12.4   15.3   17.8   20.4   24.5   25.1   28.0   29.1
 *     tgt 128   12.0   16.0   18.8   21.8   25.1   26.2   29.1   31.1
 *
 * Four ways wins at M = 8 and three wins at every M above it, by 0.7 to 2.0 us in isolation --
 * which across fifty-two calls looks like most of a millisecond a step and, in the engine, is about
 * a tenth of one.
 *
 * THE ISOLATED HARNESS OVERSTATES A SMALL WEIGHT. It rotates a 768 MB working set to defeat the
 * cache; in the engine this weight is 2.6 MB and stays resident, and --profile-ops reads the call
 * about a fifth faster than isolation does. What is left after that is a quarter of a percent of a
 * step.
 *
 * AND IT IS NOT FREE: a split repartitions K, so the k/v projections at M > 8 accumulate in a
 * different order and the emitted text moves. Decode at one sequence is M = 8 and untouched, but a
 * short PREFILL chunk is not. A tenth of a millisecond is not a reason to spend the model's output,
 * so the lower target is NOT used. A change here needs a gain worth a change in the output bits.
 *
 * A rule that DID want this would also have to fix the scratch query: it is asked once at declare
 * with the band's upper bound and is safe only because `ksplit * M * N` falls as M does. A target
 * that gives M <= 8 a larger split breaks that on both factors, and at a narrow band -- an upper
 * bound of 9, say -- the answer at the bound (3*9) is under the need at M = 8 (4*8). The launch
 * then degrades to ksplit 1 silently, which is the failure that query exists to prevent. */
// ...and the cap when the OUTPUT is thin, which is a different question: a shape with `M*N < 8192`
// makes a reduce that can be thinner than the GEMM the split was widening. The reduce keeps eight
// loads in flight rather than running as a dependent chain, which is what lets the cap sit this
// high.
constexpr uint32_t kSplitThinCap = 64u;


// ---- WHY THE WEIGHT IS IN FRAGMENT ORDER -------------------------------------------------------
//
// From a row-major weight this tile reaches 515 GB/s at gate_up's decode shape where
// r4d_gemm_fp8a16_nt_m1 reaches 620 on the identical bytes, and the difference is the SHAPE OF THE
// READ. A 16x16 fragment out of a row-major [N][K] plane is 8 bytes every `w_ld`, so the weight has
// to go global -> LDS -> fragment: a staging pass, a barrier a K tile, and 36 KiB of LDS a block.
// Stored in FRAGMENT ORDER the same bytes are contiguous and none of that is needed. The order is a
// container property -- the layout hook applies it at convert time -- so it has to serve the
// prefill tile and the matvec too.
//
// THE REQUEST WIDTH IS MOST OF THE ANSWER. At 8 bytes a lane the fragment read is a 256 B request
// and LOSES to the staged path's 1 KiB on the wide shapes; at 32 bytes a lane -- four fragments,
// 1 KiB a wave -- it wins everywhere, by around 13% at the shape decode spends a third of its GEMM
// time in and 2-3% elsewhere. The layout does not win on its own; as with the staging assignment
// below, the request width does.
// ---- the body ----------------------------------------------------------------------------------
//
// fp8 x fp8 -> f32. Nothing is converted anywhere in the loop: both operands are already E4M3 bytes
// and both go global -> LDS -> fragment as raw bytes. That is the signature of a correct fp8 GEMM
// on this card -- zero 16-bit WMMA and zero in-loop dequant v_cvt in the disassembly.
//
// Both operands carry a scale per 128 along K, so the two are constant together over a 128-column
// block and are applied ONCE per block to the partial sum rather than per element. That is why
// there are two accumulator sets: `raw` collects 128 columns of K unscaled, then folds into `out`.
// It costs 64 extra VGPRs and it is what keeps the inner loop free of per-element arithmetic.
//
// Both operands are ROW-major [row][k] byte planes, which is what makes staging a pure byte copy: a
// thread wants 32 consecutive k for one row, which is 32 consecutive bytes. A dim-major activation
// would make it 32 separate strided reads.
//
// ---- AW: which operand is the weight, and why it is a template parameter -----------------------
//
// The two operands are NOT interchangeable, because their scales have different shapes: a weight
// carries one scale per 128 ROWS, an activation one per TOKEN. And the accumulator is
// column-distributed, so a lane owns ONE B row and EIGHT A rows -- which makes a per-B-row scale
// one scalar a lane and a per-A-row scale eight.
//
//   AW = true    A is the weight. Then D is [N][M] and the epilogue TRANSPOSES on the way out:
//                sixteen lanes write sixteen addresses `N` elements apart, which is scattered. Used
//                only at M <= 16, where the output is at most 16*N against a whole-weight read of
//                N*K bytes -- 0.6% of the traffic at gate_proj's shape. Named because it is a real
//                cost, bounded because the band is. It buys the decode tile shape: the small
//                dimension lands on B, where a 16-wide tile is one N tile, and the M grid is over
//                N, which is where the blocks are.
//
//   AW = false   A is the activation. Then D is already [M][N] and the epilogue is a plain
//                coalesced store. This is prefill. It is correct ONLY at FN == 128, because the
//                weight's block scale then covers exactly the block's 128 B rows and the fold can
//                treat it as one scalar; static_assert'd below.
// ---- GATED: the SwiGLU and the quantiser folded into the epilogue -----------------------------
//
// `gated_quant_fp8` reads this kernel's [M, 2n] bf16 output back and writes [M, n] E4M3 plus a
// scale per (token, 128 columns). At M = 64 that is 2.23 MB re-read, 13.1 us of launch and pass,
// and 2.23 MB of SCATTERED bf16 the transposing store had to write in the first place -- around a
// sixtieth of a wide-batch step for the quantiser alone, and the largest op left that is not
// already at a wall.
//
// THE SHAPE FITS THE NARROW TILE EXACTLY, which is the whole reason this exists here and not in
// the prefill tile (where it is dead: see kSplitTarget's neighbour, the 8-accumulator cap). A
// block owns FM = 256 weight rows and a wave owns 16 of them. Give the FIRST half of the waves
// the gate rows `[c0, c0+128)` and the second half the up rows `[n + c0, n + c0 + 128)`, and:
//
//   * a block emits exactly 128 fused output columns -- ONE 128-column scale group, so the
//     quantiser's amax closes inside the block and no second pass is needed;
//   * wave w and wave w + MW/2 hold the SAME 16 output columns for the SAME tokens, in the same
//     lane, because the accumulator's (frag_c, acc_row) mapping does not depend on which weight
//     rows were multiplied. So the gate stash is a straight per-lane copy through LDS with no
//     permutation and no bank conflict;
//   * `acc_row(lane, t) = (lane >> 4) * 8 + t` is EIGHT CONSECUTIVE output columns for one token,
//     so the E4M3 store is one 8-byte access rather than the dword `quant_group` manages;
//   * the block count and the total WMMA are UNCHANGED -- 8704/128 blocks against 17408/256.
//
// THE STASH COSTS NO LDS. `s_b` is dead once the K loop ends, and MW/2 * kWave * NT * 8 bf16 is
// 16 KiB at NT = 4 against the 16.5 KiB it already holds. Only the amax plane is new, 2 KiB.
//
// AND IT IS BYTE-IDENTICAL TO THE PAIR, which is a requirement and not a hope: the accumulator is
// rounded to bf16 FIRST and the silu and the multiply run on the round-tripped values, because
// that is what a GEMM which stores a bf16 tensor and a quantiser which reads it back compute.
// The amax is an fmaxf reduction, which is exact and order-free, so folding it across waves rather
// than across a wave's lanes cannot move a bit.
// ---- AN EVICT-FIRST HINT ON THE NARROW PATH'S WEIGHT STREAM, AND WHY IT IS OFF ----------------
//
// The two operands of a decode GEMM have opposite reuse and the cache cannot tell. On the narrow
// path A is the WEIGHT: every block reads a DIFFERENT 16-row slice, once, and no block ever reads
// it again. B is the ACTIVATION: every block reads ALL of it, so at gate_up's 68 blocks a 320 KiB
// activation is read 68 times -- 22 MiB of requests off 320 KiB of data, which is a cache hit if
// and only if the 89 MiB of weight streaming past it has not evicted it. LRU says it has.
//
// So the hint goes on the weight load and not on the activation load. Nothing about the arithmetic
// moves; it is a replacement policy hint on one of the two loads.
//
// NOT ON THE PREFILL TILE, where the sharing is the other way round: its grid is
// ceil(M/128) x ceil(N/128) and the M blocks of one N band all read the SAME weight tile, 32 of
// them at a 4096-token chunk. Hinting a shared operand evict-first is the opposite of this.
//
// IT IS OFF. In isolation the hint is worth 2-7% at every shape and in both bands; in the engine
// the step does not move at all. The isolated benchmark runs one shape 50 times over 20 weight
// copies with ONE activation buffer, so the activation is hot across iterations and the hint
// measures its own repetition. The engine calls each GEMM once, on an activation the previous
// kernel has just written -- already resident, with nothing to protect it from. The re-read model
// above ("68 blocks x 320 KiB") is right about the request count and wrong about where those
// requests are served: they already hit.
//
// It stays as one template argument, off, so the measurement stays with the code -- as
// r4d_attn_decode's O_KNT does for the KV stream.
constexpr bool kNarrowWeightNT = false;

/* I8 IS THE SAME KERNEL AT INT8 CODES (gemm_i8a8_*): both operands i8 bytes in the same staging
 * and fragment order, v_wmma_i32_16x16x16_iu8 for the fp8 mma, and the WEIGHT's scale per ROW per
 * 128 K -- bf16, tile-major [N/16][K/128][16] (r4d_fp8_layout.cpp) -- where the fp8 weight's is
 * per 128x128 tile. Only the fold reads the difference: one weight scale per accumulator row on
 * the narrow path, one per lane's column on the prefill tile. The fp8 instantiations are
 * unaffected. */
template <bool Full, bool AW, uint32_t NT, uint32_t MT, uint32_t MW, uint32_t FK,
          bool GATED = false, bool I8 = false>
__global__ __launch_bounds__(MW * kWave) void gemm_fp8a8_kernel(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck,
    uint8_t* __restrict__ Q = nullptr, float* __restrict__ Sc = nullptr,
    uint16_t* __restrict__ Yb = nullptr,
    uint32_t q_ld = 0, uint32_t s_ld = 0, uint32_t y_ld = 0) {
    constexpr uint32_t FN = NT * 16;              // B rows a block
    constexpr uint32_t FM = MW * MT * 16;   // A rows a block

    // The AW=false fold reads the weight's scale as ONE scalar for the block's B rows, so those rows
    // must sit in one 128-row scale row: FN divides 128 and n0 is FN-aligned, which gives it.
    static_assert(AW || kBlock % FN == 0, "AW=false folds the weight scale as one scalar per block");
    // Only the AW branch folds the A-side scale as one scalar; with AW=false the A side is the
    // activation and carries a scale per ROW, so FM is free of the block. THE UNIT IS THE WAVE AND
    // NOT THE BLOCK: the fold below indexes at `(m0 + wm) / kBlock`, so what has to sit in one
    // 128-row scale row is a wave's MT*16 rows, not the block's FM of them. Requiring it of the
    // block would cap FM at 128 and so MW at 8 -- and the block's activation re-read is the whole
    // cost past 16 token rows.
    static_assert(!AW || (MT * 16 <= kBlock && kBlock % (MT * 16) == 0),
                  "a wave's A rows must sit in one scale row");
    // ...and m0 is FM-aligned with FM a multiple of MT*16, so a wave's rows never straddle either.
    static_assert(!AW || FM % (MT * 16) == 0, "FM must be a whole number of wave row tiles");

    // DOUBLE BUFFERED, on the prefill tile only. A single buffer needs two barriers a K tile -- one
    // to know every wave has finished READING the tile before it is overwritten, one to publish the
    // new one -- and there are only 32 WMMA between them, so eight waves rendezvous twice per tile.
    // With two buffers the overwrite lands somewhere nobody is reading and only the publish barrier
    // remains, which is worth about 5% at gate_up's prefill shape -- and it pays even though the
    // extra LDS drops the block from 4 resident a WGP to 3.
    //
    // The narrow tile keeps one buffer: it is 256 B wide and two of them are 74 KiB against the
    // 64 KiB a workgroup may hold. It loses nothing -- decode waits on HBM, not on a barrier -- and
    // the two-barrier structure is preserved there by the `else` at the loop foot.
    constexpr uint32_t kBufs = AW ? 1u : 2u;
    __shared__ uint8_t s_b[kBufs][FN * lds_pitch<FK>()];
    // NO LDS FOR A WHERE A IS THE WEIGHT. Fragment order delivers a lane its own fragment bytes
    // straight from global, so there is no 16-row transpose to stage -- and the narrow tile needs
    // 4 KiB of LDS a block rather than 38, which is occupancy on top of the fetch. One byte rather
    // than zero because a zero-length array is not a type.
    __shared__ uint8_t s_a[kBufs][AW ? 1u : FM * lds_pitch<FK>()];
    uint32_t buf = 0;

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    // Split-K. `ck` is a multiple of both 128 (the scale block) and FK (the staged tile), and
    // ksplit is derived FROM it at the launcher, so every slice has work and starts on a scale
    // boundary.
    const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
    const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
    if (kbeg >= kend) return;

    const uint32_t m0 = blockIdx.x * FM;
    const uint32_t n0 = blockIdx.y * FN;
    // Wave grid MW(M) x 1(N): a wave owns MT*16 A rows and all FN B rows. A 4x2 grid at the same
    // 128x128 tile -- 2x4 a wave, so 6 fragment reads per 8 WMMA rather than 9 -- is a WASH, which
    // says the fragment reads, which do not bank-conflict, are not the wall. There is no
    // second wave axis in the code for that reason.
    const uint32_t wm = wave * (MT * 16), wn = 0;
    /* THE ONE THING GATED CHANGES BEFORE THE EPILOGUE: which weight rows a wave owns. The gate
     * half and the up half are `rows/2` apart, so the assignment stops being contiguous and every
     * later use of `m0 + wm` -- the fragment tile, the liveness test, the A-side block scale --
     * goes through this instead. A wave still owns 16 consecutive rows, 16 still divides 128, so
     * the "a wave's rows sit in one scale row" invariant that lets FM be 256 is untouched. */
    constexpr uint32_t kGHalf = GATED ? (MW / 2u) : MW;
    const uint32_t wrow0 =
        GATED ? ((wave < kGHalf ? 0u : rows / 2u) + blockIdx.x * (FM / 2u)
                     + (wave % kGHalf) * 16u)
              : (m0 + wm);
    static_assert(!GATED || (AW && MT == 1u && (MW % 2u) == 0u),
                  "the gated fold is the narrow tile's epilogue: A is the weight, one M tile a "
                  "wave, and the waves split in half");
    static_assert(!(GATED && I8), "the gated epilogue quantises to E4M3 and has no int8 form");

    f8v raw[MT][NT], out[MT][NT];
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) { raw[i][j] = acc_zero(); out[i][j] = acc_zero(); }

    // ---- THE STAGING ASSIGNMENT, which is a bandwidth decision and not a bookkeeping one --------
    //
    // A thread stages one 32-byte granule pair; the question is WHICH one, and the answer is worth
    // 1.5x of the weight stream. The threads that share a row are CONSECUTIVE, so `kTPR` of them
    // cover a whole row of the tile and a wave's single load instruction asks for
    // `32 * 32 / kTPR` rows of `kTPR * 32` contiguous bytes each.
    //
    // With `tid >> 1` -- two threads a row at every FK -- a wave asks for SIXTEEN 64-byte pieces of
    // sixteen different rows, each row `ld` bytes from the last. That is sixteen separate
    // transactions off a 256-byte cacheline. A kernel that reads the same bytes in the same order
    // and does nothing else (rows of 5120 B, 1 GiB total, gfx1201):
    //
    //     threads a row      2         4         8        16        32
    //     GB/s             422       626       633       635       636      (plain stream: 638)
    //
    // So two threads a row leaves a third of the card's bandwidth on the table, and four already
    // recovers effectively all of it. Since a thread's granule pair is 32 B, kTPR is FK/32 -- which
    // is 2 for the prefill tile, where the fetch is not the limit, and 8 for the 256-byte narrow
    // tile, where it IS the limit: decode reads every weight in the
    // model exactly once and does nothing else.
    //
    // Rows a pass covers is then blockDim/kTPR, and the tile takes as many passes as it needs. The
    // register cost is that of two threads a row: a thread holds kAP granule pairs at one column
    // across kAP rows rather than spread along one row. Same bytes, same count, same LDS stores.
    constexpr uint32_t kSeg = 32;                       // bytes a thread stages, contiguous
    constexpr uint32_t kTPR = FK / kSeg;                // threads covering one row
    constexpr uint32_t kRPP = (MW * kWave) / kTPR;      // rows one pass covers
    constexpr uint32_t kAP  = AW ? 0u : (FM + kRPP - 1) / kRPP;   // passes over the A tile
    // Fragment-order supertiles a staged K tile spans, when A is the weight: one 32-byte load a
    // lane each, and each carries FOUR fragments (r4d_fp8_frag.h says why four and not one).
    constexpr uint32_t kAF  = AW ? FK / r4d_fp8::kSuperK : 0u;
    constexpr uint32_t kBP  = AW ? (FN + kRPP - 1) / kRPP : 0u;   // ... and over the B tile
    static_assert(kTPR >= 1 && kRPP >= 1, "FK must be at least 32 B and no wider than the block");

    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;
    const uint32_t am = sn;   // A staging, identical assignment

    // The B rows this lane owns, one per N tile -- a lane owns column `frag_c` of each of its NT
    // accumulators, and those are 16 columns apart. Folding a per-B-row scale with only the first
    // one is wrong for the others, which reads as a ~2.0 relative error rather than as noise.
    const uint32_t brow0 = n0 + wn + frag_c(lane);

    // ---- B, WHEN B IS THE WEIGHT, comes from the fragment-order plane ------------------------
    //
    // A wave reads ONE (16 row, 64 k) tile per pass: 32 lanes x 32 contiguous bytes is the whole
    // 1 KiB tile, which is the request width the layout exists for. The de-permutation happens in
    // the LDS store, so the tile in LDS is the same row-major padded one every fragment read in
    // this family expects.
    constexpr uint32_t kBSK = FK / r4d_fp8::kSuperK;              // k supertiles a staged tile
    constexpr uint32_t kBTL = (FN / 16u) * (kBSK ? kBSK : 1u);    // 16-row tiles in it
    constexpr uint32_t kBFP = AW ? 0u : (kBTL * 32u + MW * kWave - 1u) / (MW * kWave);
    const uint32_t btile = tid / 32u, blane = tid % 32u;
    const uint8_t* bsrc = Bp + (size_t)(n0 + sn) * b_ld + sk;
    const uint8_t* asrc = Ap + (size_t)(m0 + am) * a_ld + sk;
    // A AS FRAGMENTS. A wave owns ONE 16-row tile of the weight (MT is 1 on the narrow path), so
    // its whole A operand for 64 k is the 32 bytes this lane holds in that tile. `ksuper` is the
    // plane's tile stride; `arow` is this wave's row tile, which is out of range only when N is
    // not a multiple of the block's FM rows.
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    const uint32_t arow   = wrow0 / 16u;
    const bool     alive  = !AW || wrow0 < rows;
    const uint8_t* afrag  = Ap + (size_t)lane * r4d_fp8::kLaneBytes;

    // One K tile ahead, in registers. The load for tile c0+FK is issued before the WMMA for tile c0
    // and consumed after it, so the memory latency sits under the arithmetic instead of in front of
    // a barrier. Nothing about what is computed changes -- same operands, same order.
    //
    // `sk` is the same column for every pass, so the liveness test against `cols` is too, and the
    // per-pass part of the predicate is only whether that row exists.
    const bool k_live = Full || (kbeg + sk + kSeg <= cols);
    // `pa` stages A row-major into LDS; `pf` is A read as fragments. Exactly one of them is live,
    // and the other is a zero-length array the compiler drops.
    /* `pf` is the ONLY operand the mma loop reads straight from registers -- everything else has
     * been stored to LDS by then -- so its prefetch cannot land in the registers the CURRENT tile
     * is about to be multiplied out of. `pfn` takes the next tile and the loop foot rotates.
     * Without it the prefetch destroys the tile it sits in front of, which is invisible wherever
     * split-K makes each block run ONE K iteration and the prefetch never fires. */
    Stage8 pb[kBP ? kBP : 1u], pa[kAP ? kAP : 1u], pw[kBFP ? kBFP : 1u];
    Stage8 pf[kAF ? kAF : 1u], pfn[kAF ? kAF : 1u];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + kbeg,
                            k_live && r < FN && (Full || (n0 + r) < nb));
    }
#pragma unroll
    for (uint32_t p = 0; p < kBFP; ++p) {
        const uint32_t ti = btile + p * ((MW * kWave) / 32u);
        const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
        const uint32_t st = kbeg / r4d_fp8::kSuperK + ks;
        pw[p] = stage8_load(Bp + (size_t)((n0 / 16u + rt) * ksuper + st) * r4d_fp8::kTileBytes
                                + (size_t)blane * r4d_fp8::kLaneBytes,
                            ti < kBTL && (n0 + rt * 16u) < nb && st < ksuper);
    }
#pragma unroll
    for (uint32_t p = 0; p < kAP; ++p) {
        const uint32_t r = am + p * kRPP;
        pa[p] = stage8_load(asrc + (size_t)p * kRPP * a_ld + kbeg,
                            k_live && r < FM && (Full || (m0 + r) < rows));
    }
#pragma unroll
    for (uint32_t p = 0; p < kAF; ++p) {
        const uint32_t st = kbeg / r4d_fp8::kSuperK + p;
        pf[p] = stage8_load<kNarrowWeightNT>(
            afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes, alive && st < ksuper);
    }

    for (uint32_t c0 = kbeg; c0 < kend; c0 += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b[buf], r, sk, pb[p]);
        }
#pragma unroll
        for (uint32_t p = 0; p < kBFP; ++p) {
            const uint32_t ti = btile + p * ((MW * kWave) / 32u);
            if (ti >= kBTL) continue;
            const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
            stage8_store_frag<FK>(s_b[buf], rt * 16u + (blane & 15u),
                                  ks * r4d_fp8::kSuperK + (blane >> 4) * 8u, pw[p]);
        }
#pragma unroll
        for (uint32_t p = 0; p < kAP; ++p) {
            const uint32_t r = am + p * kRPP;
            if (r < FM) stage8_store<FK>(s_a[buf], r, sk, pa[p]);
        }
        __syncthreads();
        const uint32_t c1 = c0 + FK;
        if (c1 < kend) {
            const bool live = Full || (c1 + sk + kSeg <= cols);
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r < FN)
                    pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + c1,
                                        live && (Full || (n0 + r) < nb));
            }
#pragma unroll
            for (uint32_t p = 0; p < kBFP; ++p) {
                const uint32_t ti = btile + p * ((MW * kWave) / 32u);
                const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
                const uint32_t st = c1 / r4d_fp8::kSuperK + ks;
                pw[p] = stage8_load(Bp + (size_t)((n0 / 16u + rt) * ksuper + st) * r4d_fp8::kTileBytes
                                        + (size_t)blane * r4d_fp8::kLaneBytes,
                                    ti < kBTL && (n0 + rt * 16u) < nb && st < ksuper);
            }
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = am + p * kRPP;
                if (r < FM)
                    pa[p] = stage8_load(asrc + (size_t)p * kRPP * a_ld + c1,
                                        live && (Full || (m0 + r) < rows));
            }
#pragma unroll
            for (uint32_t p = 0; p < kAF; ++p) {
                const uint32_t st = c1 / r4d_fp8::kSuperK + p;
                pfn[p] = stage8_load<kNarrowWeightNT>(
                    afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes,
                    alive && st < ksuper);
            }
        }

#pragma unroll
        for (uint32_t ks = 0; ks < FK; ks += kK) {
            i2v af[MT], bf[NT];
            // A FROM REGISTERS WHERE IT IS THE WEIGHT: the 32 bytes this lane loaded for the
            // 64-k supertile ARE its four fragments, at +0, +8, +16 and +24. `ks` is a constant
            // after unrolling, so the selection folds away entirely.
            if (AW) {
                af[0] = frag_of(pf[ks / r4d_fp8::kSuperK], (ks % r4d_fp8::kSuperK) / 16u);
            } else {
#pragma unroll
                for (uint32_t i = 0; i < MT; ++i)
                    af[i] = frag<FK>(s_a[buf], wm + i * 16, ks, lane);
            }
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j)       bf[j] = frag<FK>(s_b[buf], wn + j * 16, ks, lane);
#pragma unroll
            for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[i][j] = mma8<I8>(af[i], bf[j], raw[i][j]);

            // End of a 128-column scale block: fold both scales in and reset. INSIDE the ks loop,
            // because at FK = 256 one staged tile spans two scale blocks and folding once would
            // apply the second block's scale to the first block's products.
            //
            // THE TEST IS NOT COMPILE-TIME AND THE BODY RUNS AT EVERY ks. `c0` is a runtime value,
            // so the modulo is a runtime test at each unrolled position and the compiler
            // IF-CONVERTS it rather than branching: `libr4d/isa/loop.sh` shows all four of the
            // prefill tile's ks blocks carrying eight scale loads with a 64-bit address each, 64
            // scaled adds and 64 resets -- ~150 of ~190 instructions against eight WMMA, where once
            // every OTHER staged tile is what the arithmetic asks for.
            //
            // GATING IT IS LEGAL, BYTE-IDENTICAL, AND ABOUT 16% SLOWER on the prefill tile, which
            // is the same wall the tile-widening variant in the header hits and a sharper reading
            // of it. Both entry points require K to be a multiple of kBlock and of FK, so the
            // position IS constant after unrolling -- at FK <= kBlock only a tile's last step can
            // fire, at FK > kBlock the boundaries sit at fixed ks. Gated, the loop is 310
            // instructions instead of 847 at the same 32 WMMA (the narrow tiles 295 instead of 989
            // and 513 instead of 2140), the narrow tiles do not move, and the prefill tile loses.
            // Hoisting the scale loads to the top of the staged tile as well, so a whole tile of
            // WMMA covers them, changes nothing further.
            //
            // The header blames 44 VGPRs and two waves a SIMD for the widening variant. Gating
            // widens nothing and still takes the kernel from 203 to 253 VGPRs, so it is the
            // REGISTERS the fold's live range costs and not the iteration size -- and deleting
            // instructions from this loop is not the lever. See libr4d/isa/README.md for the
            // issue-rate arithmetic that says the same thing from the other side.
            if ((c0 + ks + kK) % kBlock == 0 || c0 + ks + kK >= kend) {
                const uint32_t kb = (c0 + ks) / kBlock;
                if (AW && I8) {
                    // A is the int8 weight: a bf16 scale per ROW, and a lane's eight accumulator
                    // slots are eight CONSECUTIVE rows of one 16-row tile -- which the tile-major
                    // plane, [N/16][K/128][16], stores side by side: one 16-byte load. B is the
                    // activation, as below.
                    const uint32_t nkb8 = (cols + kBlock - 1) / kBlock;
                    float wsr[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i) {
                        const uint32_t tr = wrow0 / 16u + i;
                        uint4 w8 = make_uint4(0u, 0u, 0u, 0u);
                        if (Full || tr * 16u < rows)
                            w8 = *(const uint4*)((const uint16_t*)Sa +
                                                 ((size_t)tr * nkb8 + kb) * 16u + (lane >> 4) * 8u);
                        const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t)
                            wsr[i][t] = bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                    }
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t tok = brow0 + j * 16;
                        const float s = (Full || tok < nb)
                                            ? ((const float*)Sb)[(size_t)tok * sb_ld + kb] : 0.0f;
#pragma unroll
                        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += acc_val<I8>(raw[i][j][t]) * (wsr[i][t] * s);
                                raw[i][j][t] = 0.0f;
                            }
                    }
                } else if (I8) {
                    // B is the int8 weight: a bf16 scale per row, and a lane owns ONE weight row a
                    // tile -- its column, frag_c -- so one scale a lane a tile. A is the activation
                    // with its f32 scale per (token, 128 K), eight a lane a tile.
                    const uint32_t nkb8 = (cols + kBlock - 1) / kBlock;
                    float wsc[NT];
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t col = n0 + wn + j * 16 + frag_c(lane);
                        wsc[j] = (Full || col < nb)
                                     ? bscale(((const uint16_t*)Sb)[((size_t)(col / 16u) * nkb8 + kb)
                                                                        * 16u + col % 16u])
                                     : 0.0f;
                    }
                    float as[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            const uint32_t r = m0 + wm + i * 16 + acc_row(lane, t);
                            as[i][t] = (Full || r < rows)
                                           ? ((const float*)Sa)[(size_t)r * sa_ld + kb] : 0.0f;
                        }
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += acc_val<I8>(raw[i][j][t]) * (as[i][t] * wsc[j]);
                                raw[i][j][t] = 0.0f;
                            }
                } else if (AW) {
                    // A is the weight: one bf16 scale for THIS WAVE's A rows. A wave owns MT*16 of
                    // them, 16 divides 128 and m0 + wm is a multiple of MT*16, so those rows sit in
                    // one scale row whether or not the whole block does -- which is what lets FM be
                    // 256. B is the activation: one f32 scale per (token, 128-column block), one
                    // scalar a lane a tile.
                    const float ws =
                        bscale(((const uint16_t*)Sa)[(size_t)(wrow0 / kBlock) * sa_ld + kb]);
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t tok = brow0 + j * 16;
                        const float s = ws * ((Full || tok < nb)
                                                  ? ((const float*)Sb)[(size_t)tok * sb_ld + kb]
                                                  : 0.0f);
#pragma unroll
                        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += raw[i][j][t] * s;
                                raw[i][j][t] = 0.0f;
                            }
                    }
                } else {
                    // B is the weight: one bf16 scale for the block's 128 B rows. A is the
                    // activation: one f32 scale per token, and a lane owns EIGHT A rows through
                    // acc_row, so eight scalars a lane a tile. Same load count as the AW branch.
                    const float ws = bscale(((const uint16_t*)Sb)[(size_t)(n0 / kBlock) * sb_ld + kb]);
                    float as[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            const uint32_t r = m0 + wm + i * 16 + acc_row(lane, t);
                            as[i][t] = (Full || r < rows)
                                           ? ws * ((const float*)Sa)[(size_t)r * sa_ld + kb]
                                           : 0.0f;
                        }
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += raw[i][j][t] * as[i][t];
                                raw[i][j][t] = 0.0f;
                            }
                }
            }
        }
        // Two buffers: nothing to wait for, the next store lands in the other one. One buffer: this
        // is the barrier that says every wave is done reading before it is overwritten.
        /* The next tile's fragments become this one's, after the mma has consumed them. */
#pragma unroll
        for (uint32_t p = 0; p < kAF; ++p) pf[p] = pfn[p];
        if (kBufs > 1u) buf ^= 1u; else __syncthreads();
    }

    if constexpr (GATED) {
        constexpr uint32_t kStash = kGHalf * kWave * NT * 8u;          // bf16 gate values
        static_assert(kStash * 2u <= FN * lds_pitch<FK>(),
                      "the gate stash must fit in the staging buffer it reuses");
        uint16_t* const gst = (uint16_t*)&s_b[0][0];
        __shared__ float s_amx[kGHalf][FN];
        const uint32_t half = wave / kGHalf, w2 = wave % kGHalf;
        const uint32_t c = frag_c(lane);

        // s_b is still being read by the last K tile's fragments when this is reached.
        __syncthreads();
        if (half == 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (!Full && n0 + wn + j * 16u >= nb) continue;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    gst[((w2 * kWave + lane) * NT + j) * 8u + t] = f32_to_bf16(out[0][j][t]);
            }
        }
        __syncthreads();

        // Every arithmetic step below is r4d_fused_quant_fp8.hip's gated_quant_fp8_kernel, in its
        // order: the gate and the up value BOTH round-tripped through bf16, silu in f32, the
        // product rounded to bf16, and the quantiser reading that.
        float v[NT][8];
        if (half != 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                /* A WHOLLY DEAD TOKEN TILE IS SKIPPED BY THE WHOLE WAVE, which is the only
                 * saving available here: `tok` is a function of the LANE, so the per-element
                 * guard in the store below is divergence and costs what it skips. `n0 + j*16` is
                 * uniform, and at M = 48 with NT = 4 the last tile is entirely past the end. */
                if (!Full && n0 + wn + j * 16u >= nb) continue;
                float m = 0.0f;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const float g = bf16_to_f32(gst[((w2 * kWave + lane) * NT + j) * 8u + t]);
                    const float a = g / (1.0f + expf(-g));
                    const uint16_t h =
                        f32_to_bf16(a * bf16_to_f32(f32_to_bf16(out[0][j][t])));
                    v[j][t] = bf16_to_f32(h);
                    m = fmaxf(m, fabsf(v[j][t]));
                }
                // A wave's 16 output columns for one token live in lanes `c` and `c ^ 16`; the
                // group's other 112 columns are the other MW/2 - 1 waves, and fmaxf is exact and
                // order-free, so this reduction is the wave one `quant_group` does, regrouped.
                m = fmaxf(m, r4d_xor_lane<16>(m));
                if (lane < 16u) s_amx[w2][j * 16u + lane] = m;
            }
        }
        __syncthreads();
        if (half != 0u) {

            const uint32_t col0 = blockIdx.x * (FM / 2u) + w2 * 16u + (lane >> 4) * 8u;
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const uint32_t tok = n0 + wn + j * 16u + c;
                if (!Full && tok >= nb) continue;
                float mm = 0.0f;
#pragma unroll
                for (uint32_t u = 0; u < kGHalf; ++u) mm = fmaxf(mm, s_amx[u][j * 16u + c]);
                const float sc = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
                if (w2 == 0u && (lane >> 4) == 0u) Sc[(size_t)tok * s_ld + blockIdx.x] = sc;
                const float inv = 1.0f / sc;
                // The clamp is r4d_quant_act_fp8.hip's and it is not belt and braces: the builtin
                // answers an out-of-range input with a NaN rather than saturating, and
                // `amax * (1/scale)` is 448 in exact arithmetic but can land an ulp above it.
                float qc[8];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    qc[t] = fminf(fmaxf(v[j][t] * inv, -448.0f), 448.0f);
                int pk[2] = { 0, 0 };
#pragma unroll
                for (uint32_t d = 0; d < 2; ++d) {
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 0], qc[d * 4 + 1], pk[d],
                                                            false);
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 2], qc[d * 4 + 3], pk[d],
                                                            true);
                }
                // Eight consecutive columns, so one 8-byte store where the standalone quantiser
                // manages a dword. `q_ld` is a multiple of 128 and col0 of 8, so it is aligned.
                *(uint2*)(Q + (size_t)tok * q_ld + col0) =
                    make_uint2((uint32_t)pk[0], (uint32_t)pk[1]);
                if (Yb) {
                    __attribute__((aligned(16))) uint16_t hy[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) hy[t] = f32_to_bf16(v[j][t]);
                    uint16_t* const yr = Yb + (size_t)tok * y_ld + col0;
                    *(uint4*)yr = *(const uint4*)hy;
                }

            }
        }
        return;
    }

    // Split-K writes an f32 partial plane; without it the store is the final bf16 C.
    float* const part = ck ? (float*)Y + (size_t)blockIdx.z * rows * nb : nullptr;
    uint16_t* const c16 = (uint16_t*)Y;

    const uint32_t c = frag_c(lane);
    /* ---- THE STORE IS EIGHT CONSECUTIVE ELEMENTS AND ONE INSTRUCTION ---------------------------
     *
     * `acc_row(lane, t)` is `(lane>>4)*8 + t`, so a lane's eight accumulator slots are eight
     * CONSECUTIVE output columns; with AW the output index is `col * rows + r`, which makes them
     * eight consecutive addresses. Writing them one at a time is a 2-byte store instruction each,
     * and the eight of them land in ONE 16-byte piece of one line -- so the wave issues eight
     * requests where one would do, and the neighbouring lanes it shares a line with are `rows`
     * elements away and cannot help it.
     *
     * WHAT THAT COSTS, AND HOW TO SEE IT. Fix N and sweep K: the SLOPE is 610 GB/s at M = 8 and at
     * M = 64 alike, and the whole M penalty is in the INTERCEPT -- at N = 8192 the M = 8 to M = 64
     * difference is the same ~10 us at K = 512 as at K = 8192. A cost that does not move with K is
     * not bandwidth and not a re-read; it is the epilogue, whose only M-proportional work is this.
     *
     * The GATED epilogue above already writes its eight as one `uint2` (fp8) or one `uint4`
     * (bf16). This is the same store, and it is the same bytes at the same addresses: bit-identical
     * by construction.
     *
     * Two things gate it. It needs AW -- with B as the weight the index is `r * nb + col` and the
     * eight are `nb` apart, not adjacent. And it needs the eight to be 16-byte aligned, which they
     * are exactly when `rows` is a multiple of 8: `r` is already one (m0, wm, i*16 and
     * `(lane>>4)*8` all are), so `col * rows + r` is too. */
    const bool vec8 = AW && (rows & 7u) == 0u;
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t col = n0 + wn + j * 16 + c;
            if (!Full && col >= nb) continue;
            const uint32_t rb = m0 + wm + i * 16;
            const uint32_t r0 = rb + acc_row(lane, 0);
            if (vec8 && (Full || r0 + 8u <= rows)) {
                const size_t o = (size_t)col * rows + r0;
                if (ck) {
                    __attribute__((aligned(16))) float pv[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) pv[t] = out[i][j][t];
                    *(uint4*)(part + o)     = *(const uint4*)pv;
                    *(uint4*)(part + o + 4) = *(const uint4*)(pv + 4);
                } else {
                    __attribute__((aligned(16))) uint16_t hy[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) hy[t] = f32_to_bf16(out[i][j][t]);
                    *(uint4*)(c16 + o) = *(const uint4*)hy;
                }
                continue;
            }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t r = rb + acc_row(lane, t);
                if (!Full && r >= rows) continue;
                // AW: A held the weight, so [r][col] is [N][M] and C is [M][N]. See the AW comment.
                const size_t o = AW ? (size_t)col * rows + r : (size_t)r * nb + col;
                if (ck) part[o] = out[i][j][t];
                else    c16[o]  = f32_to_bf16(out[i][j][t]);
            }
        }

    // ---- THE SPLIT'S FINISHER, WHICH IS WHY THERE IS USUALLY NO SECOND LAUNCH ------------------
    //
    // Back-to-back DEPENDENT dispatches on this part cost 2.8 us of nothing between them, even for
    // an empty kernel, and a hipGraph takes that to 2.3 -- so it is the command processor and not
    // the host, and the only lever on it is launching fewer kernels. A decode step issues ~1600,
    // and a separate reduce for this GEMM would be 272 of them: 1.2 ms of a 41 ms step to sum two
    // f32 planes of at most 2048 elements a block.
    //
    // So the LAST z block of a column reduces that column itself. `ctr[blockIdx.x]` counts the
    // arrivals; the block that takes it to gridDim.z resets it to zero for the next launch and does
    // the sum. That keeps the property the separate kernel had and the reason it exists: the z loop
    // is in a FIXED ascending order and no atomic touches an accumulator, so the answer is
    // bit-identical to the unfused one and to itself from run to run.
    //
    // The counter lives in a small per-device, PER-STREAM allocation this file owns, NOT in the
    // caller's scratch, because the scratch is one shared arena region and any other op's kernel
    // would write over it between our launches. The self-reset is what keeps it zeroed; it is
    // sound only because two launches of this kernel never overlap -- they are ordered by the same
    // stream. A caller that runs two concurrently on ONE DEVICE needs a counter block each, and
    // `rad_lane` is exactly such a caller, so `split_ctr` is keyed on the stream: two launches on
    // one stream cannot overlap and two streams cannot collide.
    if (ck && Ctr) {
        // Every store above must be visible device-wide BEFORE the arrival is published, and to
        // every thread of the block, so the fence is per-thread and the barrier follows it.
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&Ctr[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gridDim.z) {
            if (tid == 0)
                __hip_atomic_store(&Ctr[blockIdx.x], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();                       // acquire the other slices' partials
            // This block owns A rows [m0, m0+FM) and every one of the nb output columns; with
            // AW the output index is col * rows + r, so the two loops are over exactly what the
            // stores above wrote and nothing else.
            const uint32_t rlo = m0, rhi = min(m0 + FM, rows);
            const float* const w0 = (const float*)Y;
            uint16_t* const cy = (uint16_t*)C;
            const uint32_t nr = rhi - rlo;
            if (AW && nb >= 4 && (nr & 7u) == 0u) {
                const uint32_t nr8 = nr >> 3;
                const uint32_t total_chunks = nr8 * nb;
                const size_t stride_z = (size_t)rows * nb;
                for (uint32_t cidx = tid; cidx < total_chunks; cidx += MW * kWave) {
                    const uint32_t col = cidx / nr8;
                    const uint32_t r = rlo + (cidx % nr8) * 8u;
                    const size_t o = (size_t)col * rows + r;
                    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    const float4* p_cur = (const float4*)(w0 + o);
                    float4 v0 = p_cur[0];
                    float4 v1 = p_cur[1];
                    for (uint32_t z = 1; z < gridDim.z; ++z) {
                        const float4* p_next = (const float4*)(w0 + (size_t)z * stride_z + o);
                        float4 next_v0 = p_next[0];
                        float4 next_v1 = p_next[1];
                        acc[0] += v0.x; acc[1] += v0.y; acc[2] += v0.z; acc[3] += v0.w;
                        acc[4] += v1.x; acc[5] += v1.y; acc[6] += v1.z; acc[7] += v1.w;
                        v0 = next_v0;
                        v1 = next_v1;
                    }
                    acc[0] += v0.x; acc[1] += v0.y; acc[2] += v0.z; acc[3] += v0.w;
                    acc[4] += v1.x; acc[5] += v1.y; acc[6] += v1.z; acc[7] += v1.w;
                    __attribute__((aligned(16))) uint16_t hy[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) hy[t] = f32_to_bf16(acc[t]);
                    *(uint4*)(cy + o) = *(const uint4*)hy;
                }
            } else {
                for (uint32_t idx = tid; idx < nr * nb; idx += MW * kWave) {
                    const uint32_t r = rlo + idx / nb, col = idx % nb;
                    const size_t o = AW ? (size_t)col * rows + r : (size_t)r * nb + col;
                    float acc = 0.0f;
                    for (uint32_t z = 0; z < gridDim.z; ++z) acc += w0[(size_t)z * rows * nb + o];
                    cy[o] = f32_to_bf16(acc);
                }
            }
        }
    }
}

// Sum the split-K partials into bf16 C. The z loop is in a FIXED order and there is no atomic
// anywhere. It is LATENCY bound, not bandwidth bound, which is why the loads are batched: written
// as one `acc +=` per iteration the adds are a dependent chain and each load sits behind the
// previous add, so the few blocks this grid has issue one small round of requests and wait.
// Batching eight loads before the adds breaks the chain without touching the ORDER of the adds.
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, uint16_t* __restrict__ y,
                                     uint32_t n_out, uint32_t ksplit) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)n_out;
    if (i >= n) return;
    const float* __restrict__ p = ws + i;
    constexpr uint32_t U = 8;
    float acc = 0.0f;
    uint32_t z = 0;
    for (; z + U <= ksplit; z += U) {
        float v[U];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) v[u] = p[(size_t)(z + u) * n];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) acc += v[u];
    }
    for (; z < ksplit; ++z) acc += p[(size_t)z * n];
    y[i] = f32_to_bf16(acc);
}

// A SWEEP HOOK and not a deployment knob. Both fp8a8 rows are NOVAR -- there is no variant list
// for rad-tune to search -- and what wants sweeping here is a LAUNCH heuristic, which is reachable
// only from inside this file. `R4D_F8_MW` pins the wave count and `R4D_F8_SPLITTGT` the split's
// block target, so `r4d_selftest --perf N K M` can compare the rules below without a rebuild, and
// so a pin can be applied to a SERVING engine and compared at a real step time on one build.
// `R4D_F8_MWLOG` prints the distinct (N, M) the narrow GEMM is asked for, which is what says WHICH
// shapes to measure: guessing them from the model dimensions is wrong by a factor of two here,
// because n_ff in a container is already per rank. Read once. Unset is the default rule, and the
// default rule is what a serve runs: nothing here reads the environment on a path that matters.
uint32_t env_u32(const char* k) {
    const char* v = std::getenv(k);
    return v ? (uint32_t)std::strtoul(v, nullptr, 10) : 0u;
}



// The staged tile's width in K is kFKNarrow and 256 B -- one whole cacheline a row -- is the value.
// A 128-byte tile loses everywhere, and by a lot, at every wave count. It halves the LDS a block
// holds, which is not what this kernel is short of; what it also halves is the threads that cover
// one row, and the staging table at `kTPR` says four threads a row is where the fetch starts giving
// bandwidth back. (128 is still reachable through narrow_fk for a K that 256 does not divide.)


// TWO BUFFERS ON THIS TILE ARE WORSE, although the prefill tile above gains 5% from double
// buffering and it is natural to assume the narrow tile skips it only because MW = 8's pair does
// not fit in 64 KiB. It fits at MW = 2 -- 25 KiB -- and it
// costs about 10% at both a narrow and a very wide N. The barrier it removes is not what this tile
// waits on; the LDS it spends is what the tile is short of.

// The finisher's arrival counters, one per grid.x column. See the finisher in the kernel for why
// they are here and not in the caller's scratch: the scratch is a shared arena region and any
// other op would write over them between two of our launches.
//
// Zeroed once, and kept zero by the finisher resetting its own counter. That invariant holds only
// while two launches sharing a counter block never run at once. Failure to allocate is not an
// error: the caller falls back to the separate reduce kernel, which computes the same bytes.
constexpr uint32_t kMaxCtr = 8192;

/* ONE COUNTER BLOCK A (STREAM, OUTPUT), which is what lets two of these GEMMs run at once.
 *
 * Two launches overlap in two ways: on two streams -- `rad_lane` runs a dense GEMM on lane 1
 * beside the routed experts on lane 0 -- and on ONE stream, when the engine starts a launch before
 * the independent ones in front of it have finished (rad_kernel_concurrent). Two launches sharing
 * a block would each see the other's arrivals: a hang or a wrong answer, whichever came first.
 *
 * Launches that overlap on one stream write disjoint outputs -- that is what made them
 * independent -- and launches on two streams are told apart by the stream, so the pair names a
 * block no concurrent launch shares. Launches that do share an output on one stream are ordered by
 * the dependency between them, and reuse the block safely. A step writes a few dozen distinct
 * outputs, so the blocks are a few dozen times 32 KiB, taken once. */
unsigned int* split_ctr(RadStream s, const void* out) {
    static std::mutex m;
    static std::map<std::pair<RadStream, const void*>, unsigned int*> blocks;
    std::lock_guard<std::mutex> lk(m);
    const auto key = std::make_pair(s, out);
    if (auto it = blocks.find(key); it != blocks.end()) return it->second;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)kMaxCtr * sizeof(unsigned int)) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, (size_t)kMaxCtr * sizeof(unsigned int)) != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    blocks.emplace(key, (unsigned int*)p);
    return (unsigned int*)p;
}
// The M tile for the narrow instantiation. MW is the wave count and it IS the tile: the wave grid
// is MW(M) x 1, so a block covers MW*16 rows of the weight and the grid over N is N/(MW*16). The
// rule is NOT monotone, so "throughput is a clean monotone function of the block count, so lower
// MW is free" is wrong past 8192 rows.
//
// IT ALSO DEPENDS ON kSplitTarget, which is why the two are fitted together: which shapes split,
// and how many ways, is what decides whether a narrow block or a wide one wins, so changing the
// split target invalidates the MW rule. At the current target, and across this model's own decode
// geometries at M <= 16, MW = 4 is best or within a microsecond of best at every
// point -- where a rule that answers 2 at 17408 rows takes the WORST of 2/4/8 there, and one that
// answers 8 at 512 rows costs a couple of microseconds on a call the model makes fifty times a
// step. MW = 8 also degrades as M grows through that band while MW = 4 is flat.
//
// ---- ...AND PAST 16 ROWS IT IS THE OTHER NUMBER, BECAUSE THE ACTIVATION IS RE-READ PER BLOCK ----
//
// The weight is the A operand and a lane takes its fragment from global, so the weight is streamed
// exactly once whatever MW is. THE ACTIVATION IS NOT: it is the B operand, every block stages the
// whole [M, K] of it into LDS, and there are `N / (MW * 16)` blocks. So the bytes a call moves are
//
//     weight  N * K                  +   activation  M * K * N / (MW * 16)
//
// and the second term is nothing at M = 8 -- 13% of the first at MW = 4 -- and is THE SAME SIZE AS
// THE WEIGHT at M = 64. It is served out of the last level of cache rather than HBM, which is why
// the kernel does not simply halve, but it is the whole of the fall from 586 GB/s at M = 8 to 460
// at M = 64 on gate_up. MW = 8 halves the block count and therefore halves that term, at no cost
// in parallelism: MW * (N / (MW * 16)) waves is N/16 either way.
//
// THE SHAPES THIS IS FITTED ON ARE THE ONES THE ENGINE ASKS FOR, read off `R4D_F8_MWLOG=1` on a
// running server and not derived from the model's dimensions: `n_ff` in a container is ALREADY
// per rank, so gate_up is N = 17408 on each card and not 8704, and a sweep of the model dims halved
// measures a model that does not exist. The set is gate_up, down, gdn in_qkv|z, the out
// projections, attn q|gate, the drafter's q, and k/v.
//
// Across it, MW = 8 beats MW = 4 at every shape from M = 24 up, by 5-20% at the wide end, and the
// crossover is exactly where NT stops being 1 -- which is not a coincidence, since NT is how many
// 16-token tiles the staged activation holds and so is the multiplier on the term above. Two shapes
// go the other way just past the threshold and both lose to the rest of the set.
//
// A CAUTION ON MEASURING THIS PART: it throws occasional +5 us outliers, and they are NOT
// independent between repetitions, so three reps can leave a whole cell polluted and invent a shape
// that wants the other wave count. A cell that breaks the monotonicity of its own row is a
// re-measurement, not a discovery.
//
// ---- ...AND PAST 32 ROWS IT IS SIXTEEN, ON EVERY SHAPE WIDE ENOUGH TO STILL FILL THE CARD ------
//
// Reading the weight's scale ONCE A BLOCK at `m0 / kBlock` would cap MW at eight, FM = MW*16 having
// to fit in one 128-row scale row. A wave owns 16 rows and 16 divides 128, so the AW branch reads
// it at `(m0 + wm) / kBlock` -- per WAVE -- which is identical wherever the per-block index is
// legal and correct where it is not. That lifts FM to 256 and halves the activation re-read again:
// 272 blocks at MW = 4 over gate_up, 136 at 8, 68 at 16.
//
// It stops there, and for the other reason a block count matters. 68 workgroups over 32 WGPs is
// 2.1 each, which still tiles; 34 would be 1.06 and the tail would be the whole kernel. So the
// rule takes 16 only where N is wide enough to keep the grid full, and the measurements say the
// same thing from the other side: 512 rows is four blocks at MW = 16 and loses everywhere, while
// the wide shapes lose to MW = 8 below 48 rows and win above it. Hence `m > 32 && n >= 2048`, and
// `n % 256` for the same ksplit invariant as above. In a served step that is worth a couple of
// percent at eight concurrent sequences and nothing at four -- four is M = 32, the wrong side of
// the threshold.
//
// ---- ...AND THE TOKEN-TILE COUNT HAS NO HOLE AT THREE ------------------------------------------
//
// Jumping NT 2 -> 4 at M = 32 would make the whole 33..48 band run a 64-token tile: at M = 40 that
// is three-eighths of the staged activation, the accumulator registers and the WMMA spent on rows
// that do not exist. M = 8*C with a depth-7 drafter, so the band is exactly five and six
// concurrent sequences. With a 64-token tile there, each added concurrent sequence costs 1.2-1.9 ms
// of step time everywhere except the one that crosses M = 32, which costs 3.3.
//
// NT = 3 IS LEGAL HERE AND ILLEGAL IN THE PREFILL TILE, which is why only this dispatch has it:
// `static_assert(AW || kBlock % FN == 0)` makes FN divide 128, and 48 does not. That assert guards
// the AW = false fold, which treats a block's weight scale as ONE SCALAR and so needs the block to
// sit inside one 128-row scale row. This dispatch is AW = true -- the scale is read per wave, the
// same change that lifted MW to 16 above -- so FN = 48 binds nothing.
//
// In isolation NT = 3 wins on every shape the engine issues, at every M in the band, and the MW
// rule holds under it: at M = 40/48/56 sixteen waves still beats eight on every shape wide enough
// for the rule to ask.
//
// In the engine it is worth most of a millisecond a step at the two concurrencies that land in the
// band. `R4D_F8_NT=4` prices the mechanism directly: forcing a 64-token tile onto M = 32 costs
// about a millisecond of step, so ONE dead token tile is the size of the effect.
//
// M = 56 IS ALREADY MINIMAL. ceil(56/16) is 4, so seven concurrent sequences have no smaller tile
// to take and their 12% of padding is not addressable by this rule.
//
// The bits do not move, and the selftest checks it the way it checks the wave count: a `pin_nt`
// hook and a sweep over every NT that covers M, memcmp'd against the rule's own output, at
// M = 1, 8, 24, 32, 40, 48 and 64.
// THE SPLIT IS UNCHANGED BY THIS AND SO ARE THE OUTPUT BITS. `narrow_ksplit` decides from
// `gx * mw`, which is N/16 whatever MW is -- but only when N divides MW*16, so the rule asks for
// `n % 128` rather than trusting it. With the same ksplit every output element is still one wave's
// accumulation over the same K tiles in the same order; MW moves which wave, not what it adds.
/* The test hook behind r4d_gemm_fp8a8_pin_mw. Zero is the rule; anything else overrides it for
 * every call until it is put back. It is separate from the environment pin because the environment
 * is read once a process and the bit check has to run BOTH settings inside one. */
std::atomic<uint32_t> g_mw_force{0u};

uint32_t narrow_mw(uint32_t n, uint32_t m) {
    static const uint32_t pin = env_u32("R4D_F8_MW");
    uint32_t mw = 4u;
    if (m > 16u && (n % 128u) == 0u) mw = 8u;
    if (m > 32u && n >= 2048u && (n % 256u) == 0u) mw = 16u;
    /* 10 and 12 are here for the BLOCK-COUNT question and not for the wave count: the narrow grid
     * is ceil(N / (16*MW)) blocks over 32 WGPs, so at N = 5120 sixteen waves is twenty blocks and
     * five-eighths of the card, while ten waves is thirty-two blocks and all of it. They are pins
     * only; `narrow_mw`'s rule never returns them. */
    /* THE NARROW BAND'S BLOCK COUNT IS NOT A LEVER, IN EITHER DIRECTION.
     *
     * The split's test is `gx * mw >= kSplitTarget`, and gx * mw is N/16 whatever MW is -- a WAVE
     * count. So below N = 2048 the split multiplies the grid and at N >= 4096 the grid is already
     * 64 blocks or more, but BETWEEN them it does neither: N = 2560 at MW 4 is forty blocks over
     * 32 WGPs, 1.25 each, which reads like a second scheduling round at a quarter occupancy. It
     * is the only shape this model asks for that lands there -- R4D_F8_MWLOG says the decode band
     * is N = 256, 640, 1280, 2560, 6144 and 8192, and only 2560 has both no split and under 64
     * blocks -- and it is a few percent of traced device time.
     *
     * Halving MW deals the same 160 waves into 80 blocks instead of 40, and doubling it into 20,
     * and BOTH are slower at the decode shape, with the text unchanged either way.
     *
     * The asymmetry is the explanation: finer costs about three times what coarser does, so what
     * is being paid is PER-BLOCK work -- the activation re-read, the scale plane, the prologue --
     * and not scheduling balance. 160 waves is 160 waves; how they are grouped is not the lever,
     * because with 4608 bytes of LDS and 112 VGPRs a WGP already holds many blocks at once and
     * there is no "round" to be half-empty. */
    if (pin == 1u || pin == 2u || pin == 4u || pin == 8u || pin == 10u || pin == 12u ||
        pin == 16u) mw = pin;
    if (const uint32_t f = g_mw_force.load(std::memory_order_relaxed)) mw = f;
    /* `R4D_F8_MWLOG` prints each DISTINCT (N, M) this rule is asked about, once, to at most 64 of
     * them. Shapes derived from the model's dimensions are not the shapes the engine asks for --
     * n_ff in a container is already per rank -- and a sweep of shapes that do not exist measures
     * nothing. */
    static const uint32_t mwlog = env_u32("R4D_F8_MWLOG");
    if (mwlog) {
        static std::mutex lm;
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)n << 32) | m;
        std::lock_guard<std::mutex> lk(lm);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr, "[mwlog] N=%u M=%u -> mw=%u\n", n, m, mw);
        }
    }
    return mw;
}

// SPLITTING N = 2560 IS FREE AND BUYS NOTHING, so the target is 128.
//
// At the decode band this model asks for N = 256, 640, 1280, 2560, 6144 and 8192 (R4D_F8_MWLOG).
// Of those, 2560 is the one the split declines while still being under 64 blocks: gx * mw is N/16
// = 160, over the 128 target, so it runs 40 blocks of 4 waves. It is second only to the 8192 row
// in traced device time, and 160 waves against 8192's 512 reads like the shape that is short of
// parallelism.
//
// R4D_F8_SPLITTGT=256 makes it split, and the kernel does not move: twice the waves, the same time
// per dispatch. So this shape is not parallelism-starved and the target is not leaving anything on
// the table -- the same answer the MW rule gives from the other direction: at the narrow band the
// grid of this kernel is settled.
//
// READ grid.z, NOT grid.x. The split lives in the Z dimension: a fired split leaves gx at 40 and
// only Z moves, so a per-grid table that divides Grid_Size_X alone reports "40 blocks" either way
// and shows the split as never having fired.
//
// The split, given the narrow grid. Factored out so the sizing query and the launch cannot disagree
// about how many planes the scratch holds -- a disagreement there is a read past the end of a
// buffer the caller sized from the other one.
/* THE STAGED TILE'S WIDTH, and 256 is not always it. K = 640 -- the shared expert's down
 * projection at n_ff_shared wide -- is a multiple of 128 and not of 256, and refusing it would send
 * a DECODE call to r4d_gemm_fp8a8_tiled: a kernel whose own header says it is for prefill,
 * computing 128 M rows for the four that are live at 190 GB/s.
 *
 * FK is a template parameter and the tiled instantiation already runs it at 64, so 128 is between
 * two widths that both work rather than a new regime. It also lands on the SIMPLER half of the
 * scale fold -- the loop's `FK <= kBlock` branch, where only a tile's last step can fire.
 *
 * ONE FUNCTION, because the scratch query and the launch both need this answer: if they drift,
 * the query answers 0, the launch forces ksplit to 1, and nothing reports it. */
inline uint32_t narrow_fk(uint32_t K) { return (K % kFKNarrow == 0u) ? kFKNarrow : 128u; }

uint32_t narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck_out) {
    *ck_out = 0;
    const uint32_t mw = narrow_mw(N, M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = (N + fm - 1) / fm;              // the M grid runs over N here: AW = true
    const uint32_t tgt = env_u32("R4D_F8_SPLITTGT") ? env_u32("R4D_F8_SPLITTGT") : kSplitTarget;
    if (K < 512u || gx * mw >= tgt) return 1;
    const uint32_t want = (tgt + gx * mw - 1) / (gx * mw);
    uint32_t ckw = ((K / want) + kBlock - 1u) / kBlock * kBlock;
    if (ckw < kBlock) ckw = kBlock;
    uint32_t k2 = (K + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)N * M < 8192) k2 = kSplitThinCap;
    if (k2 < 2) return 1;
    // Rounded to the STAGED TILE, not to the 128-column scale block: a narrow tile is 256 wide, so a
    // slice whose length is only a multiple of 128 runs one tile past its own end and accumulates
    // the next slice's columns as well -- a wrong answer that still decodes fluently.

    const uint32_t fk = narrow_fk(K);
    uint32_t ck = ((K + k2 - 1) / k2 + fk - 1u) / fk * fk;
    k2 = (K + ck - 1) / ck;
    if (k2 < 2) return 1;
    *ck_out = ck;
    return k2;
}

}  // namespace

// ================================================================== the entry points

extern "C" int r4d_gemm_fp8a8_block(void)          { return (int)kBlock; }
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, const R4dSink& sk,
                                  bool i8 = false);

extern "C" int r4d_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                     int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                     long scratch, long scratch_bytes, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, sk);
}

extern "C" int r4d_gemm_fp8a8_nt_m16_max_m(void)   { return 64; }

/* r4d.h says what this is for and why the environment pin cannot do the job. An mw the dispatch
 * does not instantiate would fall through to its last arm, so only the four it has are taken and
 * anything else clears the override rather than silently meaning 16. */
extern "C" void r4d_gemm_fp8a8_pin_mw(int mw) {
    const uint32_t v = (mw == 1 || mw == 2 || mw == 4 || mw == 8 || mw == 16) ? (uint32_t)mw : 0u;
    g_mw_force.store(v, std::memory_order_relaxed);
}

/* The same hook for the TOKEN-TILE count, and it needs one more guard than the wave count does.
 * `nt` may be pinned UP but never down: NT*16 is how many token columns the epilogue is allowed to
 * write, so a pin below ceil(M/16) does not run a slower variant, it drops rows. Both dispatches
 * read it, and both apply the same ceiling. */
std::atomic<uint32_t> g_nt_force{0u};

extern "C" void r4d_gemm_fp8a8_pin_nt(int nt) {
    const uint32_t v = (nt == 1 || nt == 2 || nt == 3 || nt == 4) ? (uint32_t)nt : 0u;
    g_nt_force.store(v, std::memory_order_relaxed);
}

// Bytes of f32 scratch one r4d_gemm_fp8a8_nt_m16 call needs, or 0 when it will not split. The
// caller may always pass more, and may pass none -- the launch then runs unsplit, which is correct
// and slow rather than wrong.
//
// THE BOUND IS max_m(), NOT A LITERAL, so the query and the band cannot drift apart. A query that
// refuses rows the band admits answers 0, and every such M is handed no scratch and runs with
// ksplit forced to 1. The launch below reports nothing when that happens; it is the
// correctness-preserving silent slowdown the plugin's hook warns about, arriving through the one
// door the hook does not guard.
extern "C" long r4d_gemm_fp8a8_nt_m16_scratch(int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0 || M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    uint32_t ck = 0;
    const uint32_t k2 = narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck);
    return k2 > 1 ? (long)k2 * M * N * (long)sizeof(float) : 0;
}

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, M <= 64.
//
//   a       [M][K]                     E4M3, row-major, `a_ld` bytes a row (0 = K)
//   ascale  [M][ceil(K/128)]           f32,  `sa_ld` elements a row (0 = ceil(K/128))
//   wq      [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws      [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c       [M][N]                     bf16
//
// Returns 0, or a negative code for a shape it does not serve. `scratch` may be 0.
/* The body takes a sink, so one set of geometry serves both the launch and the description.
 * `r4d.h`'s long-stream entry is libr4d's driver API and keeps its signature exactly. */
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, const R4dSink& sk, bool i8) {
    int emit_rc = RAD_OK;
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // THE STAGED TILE IS 256 BYTES WHERE IT CAN BE AND 128 WHERE IT CANNOT -- see narrow_fk.
    //
    // Refusing K % 256 outright costs one real shape, K = 640: the shared expert's down
    // projection at n_ff_shared wide, a multiple of 128 and not of 256. That refusal sends a
    // DECODE call to r4d_gemm_fp8a8_tiled, whose own header says it exists for prefill and
    // is "hundreds of blocks at any prefill shape". At M = 4 that kernel computes FM = 128 rows for
    // four live ones on a (1, 20) grid: about 190 GB/s where the part does 621, on thousands of
    // dispatches a run, ALL on one agent -- because the shared arm runs on rank 0 alone, the rank
    // the all-reduce is already waiting for.
    //
    // 128 IS NOT A NEW REGIME. FK is a template parameter and the tiled instantiation runs it at
    // 64, so this sits between two widths that already work, and it lands on the loop's
    // `FK <= kBlock` branch where only a tile's last step of the scale fold can fire. The split's
    // `ck` follows the same width through narrow_fk, so the slice cannot overrun its own end.
    //
    // So the tiled family does not appear in a decode trace at all, and the same work costs less
    // here: a fraction of a millisecond a step off the slower rank, and the text is unchanged.
    //
    // RELAXING THE RUNTIME GUARD ALONE DOES NOTHING: the DECLARED constraint has to move with it
    // -- see cGemmNtFp8A8M16 in r4d_rows.cpp. The resolver offers a call by what a row claims, not
    // by what its entry point will accept.
    if (K % 128) return -4;
    // A wave owns 16 rows of the weight and the transposing store writes `N`-strided columns, so a
    // ragged N would make the output pitch differ from the logical one.
    if (N % 16) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    const uint32_t mw = narrow_mw((uint32_t)N, (uint32_t)M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = ((uint32_t)N + fm - 1) / fm;

    uint32_t ck = 0;
    uint32_t ksplit = narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck);
    const long need = ksplit > 1 ? (long)ksplit * M * N * (long)sizeof(float) : 0;
    if (need && (!scratch || scratch_bytes < need)) { ksplit = 1; ck = 0; }

    const dim3 grid(gx, 1, ksplit);
    void* const dst = ksplit > 1 ? (void*)scratch : (void*)c;
    /* TOKEN TILES A WAVE. The activation is the B operand here, so NT of them cover NT*16 tokens
     * and the weight is read ONCE for all of them -- the same trade the lm head makes. */
    /* THREE IS A TILE COUNT THE BAND ACTUALLY ASKS FOR. The cost of this kernel is NT and not M --
     * NT decides the staged activation, the accumulator registers and the WMMA issued, and every
     * row past M is padding that costs all three. Jumping 2 -> 4 at M = 32 would make the whole
     * 33..48 band pay the 64-row price: in a steady-state step, going from four to five concurrent
     * sequences would cost 3.33 ms where every other added sequence costs 1.2-1.9. M = 8*C with
     * a depth-7 drafter, so 40 and 48 are five and six sequences -- see the token-tile note above
     * narrow_mw.
     *
     * NT = 3 IS LEGAL HERE AND NOT IN THE PREFILL TILE. `static_assert(AW || kBlock % FN == 0)`
     * is what makes FN divide 128, and it guards the AW = false fold that treats a block's weight
     * scale as one scalar. This dispatch is AW = true -- the scale is read per wave -- so FN = 48
     * binds nothing. The gate stash is the other size constraint and it scales with NT on both
     * sides (4096*NT bytes against FN * (256+8) = 4224*NT), so it holds unchanged. */
    static const uint32_t ntpin = env_u32("R4D_F8_NT");
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    /* A pin that drops rows is not a slower variant, it is a wrong answer. */
    if (ntpin >= 1u && ntpin <= 4u && (uint32_t)M <= ntpin * 16u) nt = ntpin;
    if (const uint32_t f = g_nt_force.load(std::memory_order_relaxed)) { if ((uint32_t)M <= f * 16u) nt = f; }
    /* `Full` DROPS THE COLUMN GUARD, so it has to mean "every one of the NT tiles is live" and not
     * "M is a multiple of 16". At M = 48 with NT = 4 the second is true and the first is false, and
     * the epilogue then writes columns 48..63 -- past the end of a split-K scratch sized for 48,
     * over the arrival counter, and the finisher spins forever: a hang, not a wrong answer. */
    const bool full = ((uint32_t)N % fm == 0) && ((uint32_t)M % (nt * 16u) == 0) &&
                      (K % (int)kBlock == 0);
    // The finisher needs one counter a column; past that, and if the allocation ever fails, the
    // separate reduce kernel runs instead and computes the same bytes -- as it does under
    // `R4D_F8_NOFUSE`, which is how the two are compared.
    unsigned int* const ctr =
        (ksplit > 1 && gx <= kMaxCtr && !env_u32("R4D_F8_NOFUSE")) ? split_ctr(sk.stream, (const void*)c) : nullptr;

#define R4D_F8D_NT(f, mw_, nt_)                                                                \
    emit_rc = R4D_EMIT(sk, (gemm_fp8a8_kernel<f, true, nt_, 1u, mw_, kFKNarrow>), grid,             \
                       dim3(mw_ * kWave),                                                      \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck,                                                     \
                       /* THE KERNEL'S SIX DEFAULTED PARAMETERS, SPELLED. The chevron fills a    \
                        * default argument in at the call; kernelParams is an array and cannot,  \
                        * so a described launch would supply fifteen values for twenty-one slots \
                        * and the kernel would read six arguments' worth of whatever followed.   \
                        * r4d_emit deduces the parameter list and static_asserts the arity,      \
                        * which is the only reason this is a build error and not a quantiser     \
                        * reading a garbage pointer. */                                          \
                       (uint8_t*)nullptr, (float*)nullptr, (uint16_t*)nullptr,                  \
                       (uint32_t)0, (uint32_t)0, (uint32_t)0)
#define R4D_F8D(f, mw_)                                                                        \
    do { if (nt == 4u)      R4D_F8D_NT(f, mw_, 4u);                                            \
         else if (nt == 3u) R4D_F8D_NT(f, mw_, 3u);                                            \
         else if (nt == 2u) R4D_F8D_NT(f, mw_, 2u);                                            \
         else               R4D_F8D_NT(f, mw_, 1u); } while (0)
#define R4D_F8D_MW(f)                                                                          \
    do {                                                                                       \
        if (mw == 1u)       R4D_F8D(f, 1u);                                                    \
        else if (mw == 2u)  R4D_F8D(f, 2u);                                                    \
        else if (mw == 4u)  R4D_F8D(f, 4u);                                                    \
        else if (mw == 8u)  R4D_F8D(f, 8u);                                                    \
        else if (mw == 10u) R4D_F8D(f, 10u);                                                   \
        else if (mw == 12u) R4D_F8D(f, 12u);                                                   \
        else                R4D_F8D(f, 16u);                                                   \
    } while (0)
/* THE 128-WIDE SET IS DELIBERATELY NARROWER. Every extra (f, mw, nt) triple is another
 * instantiation of a WMMA kernel, and mirroring the whole ladder would double this file's object
 * for one shape. `narrow_mw`'s RULE only ever answers 4, 8 or 16 -- 1, 2, 10 and 12 are pins --
 * so those three cover every K % 256 call the engine can make, and a pinned mw at such a K falls
 * back to the refusal, which is the same answer a pinned mw at an unsupported K always gets. */
#define R4D_F8D_NT128(f, mw_, nt_)                                                             \
    emit_rc = R4D_EMIT(sk, (gemm_fp8a8_kernel<f, true, nt_, 1u, mw_, 128u>), grid,             \
                       dim3(mw_ * kWave),                                                      \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck,                                                     \
                       /* THE KERNEL'S SIX DEFAULTED PARAMETERS, SPELLED. The chevron fills a    \
                        * default argument in at the call; kernelParams is an array and cannot,  \
                        * so a described launch would supply fifteen values for twenty-one slots \
                        * and the kernel would read six arguments' worth of whatever followed.   \
                        * r4d_emit deduces the parameter list and static_asserts the arity,      \
                        * which is the only reason this is a build error and not a quantiser     \
                        * reading a garbage pointer. */                                          \
                       (uint8_t*)nullptr, (float*)nullptr, (uint16_t*)nullptr,                  \
                       (uint32_t)0, (uint32_t)0, (uint32_t)0)
#define R4D_F8D128(f, mw_)                                                                     \
    do { if (nt == 4u)      R4D_F8D_NT128(f, mw_, 4u);                                         \
         else if (nt == 3u) R4D_F8D_NT128(f, mw_, 3u);                                         \
         else if (nt == 2u) R4D_F8D_NT128(f, mw_, 2u);                                         \
         else               R4D_F8D_NT128(f, mw_, 1u); } while (0)
#define R4D_F8D_MW128(f)                                                                       \
    do {                                                                                       \
        if (mw == 4u)       R4D_F8D128(f, 4u);                                                 \
        else if (mw == 8u)  R4D_F8D128(f, 8u);                                                 \
        else                R4D_F8D128(f, 16u);                                                \
    } while (0)
/* THE INT8 TWIN TAKES ONLY THE THREE WAVE COUNTS THE RULE ANSWERS, at both staged widths: the
 * fp8 ladder's pins are tuning probes, and mirroring them would double the object again. */
#define R4D_I8D_NT(f, mw_, nt_, fk_)                                                           \
    emit_rc = R4D_EMIT(sk, (gemm_fp8a8_kernel<f, true, nt_, 1u, mw_, fk_, false, true>), grid,  \
                       dim3(mw_ * kWave),                                                      \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck,                                                     \
                       (uint8_t*)nullptr, (float*)nullptr, (uint16_t*)nullptr,                  \
                       (uint32_t)0, (uint32_t)0, (uint32_t)0)
#define R4D_I8D(f, mw_, fk_)                                                                   \
    do { if (nt == 4u)      R4D_I8D_NT(f, mw_, 4u, fk_);                                       \
         else if (nt == 3u) R4D_I8D_NT(f, mw_, 3u, fk_);                                       \
         else if (nt == 2u) R4D_I8D_NT(f, mw_, 2u, fk_);                                       \
         else               R4D_I8D_NT(f, mw_, 1u, fk_); } while (0)
#define R4D_I8D_MW(f, fk_)                                                                     \
    do {                                                                                       \
        if (mw == 4u)       R4D_I8D(f, 4u, fk_);                                               \
        else if (mw == 8u)  R4D_I8D(f, 8u, fk_);                                               \
        else                R4D_I8D(f, 16u, fk_);                                              \
    } while (0)
    if (i8) {
        if (!(mw == 4u || mw == 8u || mw == 16u)) return -4;
        if (narrow_fk((uint32_t)K) == kFKNarrow) {
            if (full) R4D_I8D_MW(true, kFKNarrow); else R4D_I8D_MW(false, kFKNarrow);
        } else {
            if (full) R4D_I8D_MW(true, 128u); else R4D_I8D_MW(false, 128u);
        }
    } else if (narrow_fk((uint32_t)K) == kFKNarrow) {
        if (full) R4D_F8D_MW(true); else R4D_F8D_MW(false);
    } else {
        if (!(mw == 4u || mw == 8u || mw == 16u)) return -4;
        if (full) R4D_F8D_MW128(true); else R4D_F8D_MW128(false);
    }
#undef R4D_I8D_MW
#undef R4D_I8D
#undef R4D_I8D_NT
#undef R4D_F8D_MW128
#undef R4D_F8D128
#undef R4D_F8D_NT128
#undef R4D_F8D_MW
#undef R4D_F8D
    /* On the launch path this is the dispatch's status; on the describe path it says the
     * description fitted. Either way a failure here is a failure of the call. */
    if (emit_rc != RAD_OK) return -6;
    if (!sk.desc && hipGetLastError() != hipSuccess) return -6;

    if (ksplit > 1 && !ctr) {
        const uint32_t n = (uint32_t)M * (uint32_t)N;
        /* The second dispatch of this call; the descriptor array carries both. */
        if (R4D_EMIT(sk, splitk_reduce_kernel, dim3((n + 255) / 256), dim3(256), 0,
                     (const float*)scratch, (uint16_t*)c, n, ksplit) != RAD_OK) return -7;
    }
    return 0;
}

// ...AND THE SAME GEMM WITH gated_quant_fp8's PASS FOLDED INTO ITS EPILOGUE. The kernel comment at
// `GATED` says what the fold is and why it is byte-identical; this is what the shape has to satisfy
// for it to exist at all, and every one of these is a REFUSAL rather than a slow path, because a
// gated call that quietly ran ungated would write the wrong buffer.
//
//   q       [M][nout]                  E4M3, `q_ld` bytes a row (0 = nout)
//   s       [M][ceil(nout/128)]        f32,  `s_ld` elements a row
//   y       [M][nout] or 0             bf16, the silu product the unfused pair also wrote
//
// The weight is the gate_up plane [2*nout][K] with gate first, exactly as the unfused GEMM reads
// it, so nothing about the checkpoint or the layout hook changes.
extern "C" int r4d_gemm_fp8a8_gated_nt_m16_max_m(void) { return 64; }

extern "C" int r4d_gemm_fp8a8_gated_nt_m16(long a, long ascale, long wq, long ws,
                                           long q, long s, long y, int act,
                                           int M, int K, int nout,
                                           int a_ld, int sa_ld, int w_ld, int sw_ld,
                                           int q_ld, int s_ld, int y_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !q || !s) return -1;
    if (M <= 0 || K <= 0 || nout <= 0) return -2;
    if (M > r4d_gemm_fp8a8_gated_nt_m16_max_m()) return -3;
    if (K % (int)kFKNarrow) return -4;
    // The block emits FM/2 = 128 fused columns and that is ONE 128-column scale group, which is
    // the whole reason the quantiser's reduction closes inside it. A nout that is not a multiple
    // of 128 would put a group across two blocks and there is nowhere to finish it.
    if (nout % 128) return -5;
    // Only silu is compiled: `gated_quant_fp8`'s sigmoid form is a gated attention's o_proj, whose
    // gate is a column slice of another projection and reaches `gate_quant_fp8` instead.
    if (act != 0) return -8;
    const int N = 2 * nout;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;
    if (!q_ld)  q_ld  = nout;
    if (!s_ld)  s_ld  = nout / 128;
    if (!y_ld)  y_ld  = nout;

    // The fold has no partial plane to reduce and no second launch to fold into, so a shape whose
    // grid the split rule would widen is REFUSED rather than served unsplit -- at gate_up's shape
    // (N/16 = 1088 against a target of 128) it never fires, and a shape where it would is a shape
    // this kernel is the wrong answer for.
    uint32_t ck = 0;
    if (narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck) != 1u) return -9;

    constexpr uint32_t kGmw = 16u;                  // FM = 256: 128 gate rows and 128 up rows
    /* Same rule as the narrow dispatch above, and for the same reason. The epilogue skips a token
     * tile wholly past M in the gate stash ("at M = 48 with NT = 4 the last tile is entirely past
     * the end"), but a tile count above ceil(M/16) still pays for that tile in every K
     * iteration. */
    static const uint32_t ntping = env_u32("R4D_F8_NT");
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    if (ntping >= 1u && ntping <= 4u && (uint32_t)M <= ntping * 16u) nt = ntping;
    if (const uint32_t f = g_nt_force.load(std::memory_order_relaxed)) { if ((uint32_t)M <= f * 16u) nt = f; }
    const bool full = ((uint32_t)M % (nt * 16u) == 0) && (K % (int)kBlock == 0);
    const dim3 grid((uint32_t)(nout / 128), 1, 1);

#define R4D_F8G_NT(f, nt_)                                                                     \
    hipLaunchKernelGGL((gemm_fp8a8_kernel<f, true, nt_, 1u, kGmw, kFKNarrow, true>), grid,      \
                       dim3(kGmw * kWave),                                                     \
                       0, (hipStream_t)stream, (const uint8_t*)wq, (const void*)ws,            \
                       (const uint8_t*)a, (const void*)ascale, nullptr, nullptr, nullptr,       \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, 0u, (uint8_t*)q, (float*)s, (uint16_t*)y,               \
                       (uint32_t)q_ld, (uint32_t)s_ld, (uint32_t)y_ld)
#define R4D_F8G(f)                                                                             \
    do { if (nt == 4u)      R4D_F8G_NT(f, 4u);                                                 \
         else if (nt == 3u) R4D_F8G_NT(f, 3u);                                                 \
         else if (nt == 2u) R4D_F8G_NT(f, 2u);                                                 \
         else               R4D_F8G_NT(f, 1u); } while (0)
    if (full) R4D_F8G(true); else R4D_F8G(false);
#undef R4D_F8G
#undef R4D_F8G_NT
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// The same GEMM for PREFILL row counts. Same operands and same layouts; the difference is the tile
// shape and that the ACTIVATION lands on the A operand, which makes the epilogue a coalesced store
// instead of a scattered one. No split-K: the grid is already ceil(M/128) x ceil(N/128), which is
// in the hundreds of blocks at any prefill shape, and that is the condition the split exists for.
template <bool I8>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream);

extern "C" int r4d_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long stream) {
    return gemm_8a8_tiled<false>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The int8 twins: i8 codes for both operands, the weight's bf16 scale per row per 128 K -- ws is
// [N][ceil(K/128)] -- and otherwise the fp8 entry points' arguments, bands and refusals.
extern "C" int r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                   int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                   long stream) {
    return gemm_8a8_tiled<true>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

extern "C" int r4d_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long scratch, long scratch_bytes, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, sk, true);
}

template <bool I8>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // Required, not preferred: the fold at the end of the K loop assumes a whole 128-column scale
    // block, and the AW=false branch reads the weight's scale as one scalar for 128 B rows.
    if (K % (int)kBlock) return -4;
    if (N <= 0) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    constexpr uint32_t FM = kFWaves * kFMTiles * 16;
    const dim3 grid(((uint32_t)M + FM - 1) / FM, ((uint32_t)N + kFN - 1) / kFN, 1);
    const bool full = ((uint32_t)M % FM == 0) && ((uint32_t)N % kFN == 0);
    if (full)
        hipLaunchKernelGGL((gemm_fp8a8_kernel<true, false, kFNTiles, kFMTiles, kFWaves, kFK, false, I8>), grid,
                           dim3(kFWaves * kWave), 0, (hipStream_t)stream, (const uint8_t*)a,
                           (const void*)ascale, (const uint8_t*)wq, (const void*)ws, (void*)c, (void*)c, nullptr,
                           (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)w_ld,
                           (uint32_t)sa_ld, (uint32_t)sw_ld, 0u);
    else
        hipLaunchKernelGGL((gemm_fp8a8_kernel<false, false, kFNTiles, kFMTiles, kFWaves, kFK, false, I8>), grid,
                           dim3(kFWaves * kWave), 0, (hipStream_t)stream, (const uint8_t*)a,
                           (const void*)ascale, (const uint8_t*)wq, (const void*)ws, (void*)c, (void*)c, nullptr,
                           (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)w_ld,
                           (uint32_t)sa_ld, (uint32_t)sw_ld, 0u);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. See libr4d/r4d_args.h for why there is no adapter
// directory between the core and these launches.
extern "C" int64_t r4d_rad_gemm_fp8a8_m16_scratch(const RadArgs* a) {
    const long long M = aff_or(a, "M"), N = aff_or(a, "N"), K = aff_or(a, "K");
    if (M < 1 || N < 1 || K < 1) return 0;
    if (M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    return (int64_t)r4d_gemm_fp8a8_nt_m16_scratch((int)M, (int)K, (int)N);
}

static int rad_gemm_fp8a8_m16_body(const RadArgs* a, const R4dSink& sk) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    /* A short scratch is not an error: libr4d runs unsplit, which is correct and slow. Passing the
     * length is what lets it make that decision rather than reading past the end of an arena the
     * core sized for a different band. */
    const int e = gemm_fp8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_gemm_fp8a8_m16, rad_gemm_fp8a8_m16_body)

/* THE SAME GEMM WITH THE QUANTISER'S PASS IN ITS EPILOGUE. A shim of its own rather than a case of
 * `fp8_ops`, because that helper derives M and N from operand 4 -- which here is the QUANTISER's
 * [M, nout] output and not the GEMM's [M, 2*nout] one, so every extent it checks would be off by
 * the factor the fold exists to remove. Operand indices are `gemm_nt_q_gated`'s, from
 * oGemmNtQGated in r4d_rows.cpp. */
enum { GG_A = 0, GG_AS = 1, GG_B = 2, GG_BS = 3, GG_Q = 4, GG_S = 5, GG_Y = 6 };

extern "C" int r4d_rad_gemm_fp8a8_gated_m16(const RadArgs* a, RadStream s) {
    const RadTensor* x   = r4d_opt(a, GG_A);
    const RadTensor* xs  = r4d_opt(a, GG_AS);
    const RadTensor* w   = r4d_opt(a, GG_B);
    const RadTensor* wsc = r4d_opt(a, GG_BS);
    const RadTensor* q   = r4d_opt(a, GG_Q);
    const RadTensor* sc  = r4d_opt(a, GG_S);
    const RadTensor* y   = r4d_opt(a, GG_Y);
    if (!x || !xs || !w || !wsc || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (w->rank != 2 || wsc->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (x->rank < 2 || xs->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);

    const long long blk = r4d_gemm_fp8a8_block();
    if (blk != R4D_FP8_BLOCK) return R4D_NO(RAD_E_SHAPE);
    if (r4d_geti_or(a, "group", blk) != blk) return R4D_NO(RAD_E_SHAPE);

    /* The FUSED width is the quantiser's, and the weight is twice it. Taking nout from `q` rather
     * than from the `N` parameter is deliberate: the buffer is what the kernel writes into. */
    const long long M = q->shape[0], nout = q->shape[1], K = w->shape[1];
    if (M < 1 || nout < 1 || K < 1) return R4D_NO(RAD_E_SHAPE);
    if (M > r4d_gemm_fp8a8_gated_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (w->shape[0] != 2 * nout) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + blk - 1) / blk;
    if (wsc->shape[0] != (2 * nout + blk - 1) / blk || wsc->shape[1] != nkb)
        return R4D_NO(RAD_E_SHAPE);
    if (x->shape[0] != M || r4d_numel(x) != M * K) return R4D_NO(RAD_E_SHAPE);
    if (xs->shape[0] != M || xs->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != (nout + blk - 1) / blk) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != nout)) return R4D_NO(RAD_E_SHAPE);

    const long long ap = r4d_row_pitch(x), asp = r4d_row_pitch(xs);
    const long long wp = r4d_row_pitch(w), wsp = r4d_row_pitch(wsc);
    const long long qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    if (ap < 0 || asp < 0 || wp < 0 || wsp < 0 || qp < 0 || sp < 0 || yp < 0)
        return R4D_NO(RAD_E_STRIDE);

    /* `act` is required by this schema, unlike gated_quant_fp8's, and only silu is compiled. */
    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "silu")) return R4D_NO(RAD_E_UNSUPPORTED);

    const int e = r4d_gemm_fp8a8_gated_nt_m16(
        r4d_addr(x), r4d_addr(xs), r4d_addr(w), r4d_addr(wsc),
        r4d_addr(q), r4d_addr(sc), y ? r4d_addr(y) : 0, 0,
        (int)M, (int)K, (int)nout, (int)ap, (int)asp, (int)wp, (int)wsp,
        (int)qp, (int)sp, (int)yp, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* ---- the int8 twins' shims: the same operands with i8 codes and the weight's scale plane per
 * row, which fp8_ops checks at [N][K/128] when asked (FP8_OPS_ROW_SCALE). */
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs* a) {
    return r4d_rad_gemm_fp8a8_m16_scratch(a);
}

static int rad_gemm_i8a8_m16_body(const RadArgs* a, const R4dSink& sk) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = gemm_fp8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, sk, true);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_gemm_i8a8_m16, rad_gemm_i8a8_m16_body)

extern "C" int r4d_rad_gemm_i8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_gemm_i8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_fp8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    const int e = r4d_gemm_fp8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

