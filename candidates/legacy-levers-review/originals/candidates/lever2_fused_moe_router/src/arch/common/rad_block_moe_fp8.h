/* rad_block_moe_fp8.h -- the ROUTED feed-forward block with block-scaled FP8 experts.
 *
 *   rmsnorm -> quant_act_fp8 -> router gemm -> router_topk -> moe_scatter
 *                            -> gate_up moe_gemm_q -> silu_mul + quant -> down moe_gemm_q
 *                            -> moe_gather
 *                            -> [shared expert: gate_up -> silu_mul + quant -> down -> gate]
 *                            -> add -> [all_reduce] -> residual add
 *
 * IT IS rad_block_mlp_fp8.h WITH THE FEED-FORWARD ROUTED, and everything that block's header
 * argues -- why the gate/up fusion survives block scaling, why the norm and the residual add and
 * the collective are one op, why the activations between ops stay bf16 -- is argued there and not
 * repeated. What follows is only what routing changes.
 *
 * ============================== THE SHARED EXPERT IS NOT OPTIONAL HERE ==============================
 *
 * Qwen3.5-MoE runs a dense expert BESIDE the routed ones on every token, gated by a one-output
 * projection: `y = routed(x) + sigmoid(shared_gate(x)) * shared(x)`. It is `mlp.shared_expert.*`
 * and `mlp.shared_expert_gate.weight` in the checkpoint, and a block that dropped it would produce
 * a plausible distribution and the wrong model. `moe_gather` does the multiply as it sums the
 * routed slots -- see docs/OPS.md's `scale_rows` for why `mul` cannot do it, and note that the
 * gate is applied to the SCALAR before the multiply, not to the product.
 *
 * A family without one passes `Src::shared` null and the whole arm disappears, buffers included.
 *
 * ============================== WHY THE BLOCK SLICES ITS ROWS ==============================
 *
 * The expert path's intermediates are `top_k` times as tall as the step: at an 8192-token chunk and
 * top_k 8 the down-projection's input alone is 65536 rows, which at n_embd 2048 is 256 MiB of
 * transient for ONE buffer. So the block runs in passes of at most `Config::rows` tokens and its
 * buffers are declared at that height. Routing is per pass and has to be -- `expert_offset` and
 * `sorted_tok` describe the rows of one pass -- which is not an approximation: a token's routing
 * depends on that token alone, so a pass computes exactly the rows it names.
 *
 * THE DEFAULT IS 2048 AND THAT IS THE CHUNK, not a tuning choice: core/sched/geometry.cpp clamps
 * the prefill chunk to the linear-state checkpoint interval, which on this family is 2048, so the
 * shipped configuration is one pass and the slicing is inert. It exists so that a deployment with a
 * larger chunk does not need this file edited.
 *
 * ============================== THE EXPERT WEIGHTS ARE EIGHT RUNS ==============================
 *
 * `moe_gemm_q` takes its weights as TABLES (RAD_OPD_WTAB), and a table is a run of CONSECUTIVELY
 * DECLARED handles -- `RAD_WTAB(first, n)` names the first and the count. It takes each
 * projection's experts as TWO tables by parity, the even experts and the odd ones, because every
 * entry of a table has one geometry and a rank holding an uneven slice of each expert has two (see
 * Config::ff_lo). So the arrays are declared in separate loops -- gate_up codes, gate_up scales,
 * down codes, down scales, each for the even experts and then the odd ones -- and NOT as a group
 * per expert. Interleaving them would give runs of stride two or four, which the operand cannot
 * express; the failure would be a table whose second entry is a scale plane. Where four ranks split
 * every expert the classes are four, by number mod 4 (`moe_gemm_q_mod4`, Config::ff_lo4), and the
 * runs sixteen, declared the same way class by class.
 *
 * The codes run and the scale run of a projection are two VIEWS of one stored weight -- its codes
 * plane and its scale plane (RadWeightDecl::source) -- so an expert holds two logical weights,
 * gate_up and down, and is its own movement unit (`grp_expert_slot`, slots 0 and 1) and
 * RAD_ACCESS_CONDITIONAL, which is what lets the placement planner tier them and the heat engine
 * promote the hot ones.
 */
#ifndef RAD_BLOCK_MOE_FP8_H
#define RAD_BLOCK_MOE_FP8_H

#include "rad_fp8.h"
#include "rad_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {
namespace arch {

/* THE MOST ROUTING SLOTS THE SORT SERVES IN ONE WORKGROUP, which is what bounds
 * `router_topk_scatter`. It is libr4d's `kScatterSmall` and it is stated here rather than asked
 * of the plugin for the reason every other such bound in this library is: the architecture has to
 * decide, at declare time and identically on both ranks, whether the op exists at all. */
static const int64_t kMoeSortRows = 1024;

/* The skinny bf16 GEMM's decode band, which is where the fused op's GEMM is that GEMM's own
 * arithmetic; past it the router runs the tiled kernel and the pair is issued as before. */
static const int64_t kMoeRouterFusedRows = 64;



/* ---- RADIANCE_DEBUG_ROUTING: the routing WEIGHTS of token 0, which nothing else reports.
 *
 * The core's histogram (core/runtime/ctx.cpp) says which experts a step chose and rad_arch.h's
 * dbg_resid says how big each arm's contribution was; neither shows the weights themselves, and a
 * router whose top-k sums to 0.4 instead of 1 is a block that is right in every other respect and
 * scaled down by more than half. It SYNCHRONISES THE STREAM and is off unless the variable is set:
 * a layer number, or negative for every layer. */
inline void moe_debug_weights(RadCtx* c_, int layer, int64_t top_k, rad_buf ids, rad_buf ew) {
    static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
    if (!dbg || !ids || !ew) return;
    const int want = (int)std::strtol(dbg, nullptr, 10);
    if (want >= 0 && layer != want) return;
    static int seen[256] = {0};
    if (seen[layer & 255]++) return;

    void* pi = rad_buf_ptr(c_, ids);
    void* pw = rad_buf_ptr(c_, ew);
    if (!pi || !pw) return;
    RadStream s = rad_stream(c_);
    std::vector<int32_t>  vi((size_t)top_k);
    std::vector<uint16_t> vw((size_t)top_k);
    rad_memcpy_async(vi.data(), pi, top_k * 4, s);
    rad_memcpy_async(vw.data(), pw, top_k * 2, s);
    rad_stream_sync(s);

    std::string line;
    double sum = 0.0;
    for (int64_t r = 0; r < top_k; ++r) {
        const uint32_t u = (uint32_t)vw[(size_t)r] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        sum += f;
        char b[48];
        std::snprintf(b, sizeof b, "%s%d:%.4f", r ? " " : "", vi[(size_t)r], f);
        line += b;
    }
    std::fprintf(stderr, "D moe layer %d token 0 top_k = %s  (sum %.4f)\n",
                 layer, line.c_str(), sum);
}

/* THE MOST EXPERTS A LAYER PROTECTS ON ONE RANK (MoeFP8::prot): libr4d's sort takes at most eight
 * in its `protect` list. */
constexpr int64_t kMaxProtected = 8;

struct MoeFP8 {
    struct Config {
        /* ROUTED EXPERTS THIS RANK OWNS. Equal to `n_expert_all` in every deployment that does not
         * use expert parallelism -- every rank then holds a slice of every expert (see ff_lo). A
         * calibration run is the one that splits the experts themselves, because a Hessian is
         * over an expert's whole K. */
        int64_t n_expert     = 0;
        /* THE MODEL'S expert count -- what the ROUTER is wide in and what top_k picks from. 0 means
         * "the same as n_expert", the whole-model case.
         *
         * The router stays replicated under expert parallelism and that is the property the scheme
         * rests on: every rank scores every expert from the same residual stream and picks the
         * same top_k, so the ranks agree on the routing without a collective. Each then sorts only
         * the slots that landed in ITS range and the block's end-of-block all-reduce sums the
         * partials. Sharding the router instead would have two ranks disagree about where a token
         * went, and the all-reduce would sum two different models. */
        int64_t n_expert_all = 0;
        /* This rank's first expert id, in the model's numbering. Straight through to moe_scatter,
         * which subtracts it so the offsets, the counts and the weight table are all local. */
        int64_t expert_base  = 0;
        int64_t top_k        = 0;
        int64_t n_ff_exp     = 0;   /* one expert's intermediate width, PER RANK */
        /* THE ROUTED EXPERTS' SLICE ON THIS RANK WHEN THEIR WIDTH DOES NOT HALVE INTO WHOLE fp8
         * SCALE BLOCKS: expert e's intermediate columns [ff_lo[e % 2], ff_hi[e % 2]) in the model's
         * numbering, the slice depending on the expert's PARITY. hi 0 is the even split (or no
         * split). `n_ff_exp` is then the wider slice, which is what the buffers between the two
         * grouped GEMMs are.
         *
         * WHY BY PARITY. This family's expert is 640 columns, five 128-column blocks, so two ranks
         * split each expert three blocks and two. Giving every expert's three blocks to one rank
         * would make that rank read 3/5 of every routed expert; alternating which rank takes three
         * by the expert's parity makes the ranks equal on average, and unequal only by the
         * difference between how many even and odd experts a step routes -- a fifth of an expert
         * per unit of that difference, where splitting the EXPERTS between the ranks (expert
         * parallelism) is unequal by whole experts, the binomial spread of which rank the routed
         * experts land on. The block's all-reduce turns that spread into the faster rank waiting,
         * about a millisecond a step at one sequence.
         *
         * Every rank computes its slice of every routed expert: its gate/up rows and its K slice of
         * down, and the all-reduce sums the partials -- the routing weight multiplies each partial,
         * and the sum of the partials is the expert's output. Rounding moves (two partial sums
         * where there was one), so this is checked by agreement rather than byte for byte. */
        int64_t ff_lo[2] = { 0, 0 }, ff_hi[2] = { 0, 0 };
        /* OR BY THE EXPERT'S NUMBER MOD 4, where four ranks split it: expert e's columns
         * [ff_lo4[e % 4], ff_hi4[e % 4]), served by `moe_gemm_q_mod4`, and taking precedence over
         * the parity slices. hi 0 offers none.
         *
         * Parity gives the extra block of a five-block expert two places to go, so at four ranks
         * two of them carry three blocks of every even/odd pair and two carry two -- and the
         * block's all-reduce has the narrow ranks wait for the wide ones. Four classes give it
         * four: each rank takes two blocks of one class and one of each other, five of every four
         * experts. Any count of quantised experts serves: a layer that keeps two plain bf16 has
         * 510, and its first two classes hold one expert more than the last two (the declaration
         * gives a table op's first tables the remainder). `n_ff_exp` is the widest slice. "e" is
         * the expert's place in the sort, as for the parity classes. */
        int64_t ff_lo4[4] = { 0, 0, 0, 0 }, ff_hi4[4] = { 0, 0, 0, 0 };
        int64_t n_ff_shared  = 0;   /* the shared expert's, per rank; 0 = no shared expert */
        /* THE SHARED ARM'S SLICE ON THIS RANK, when its width does not halve into whole fp8 scale
         * blocks: [shared_lo, shared_hi) of its columns in the model's numbering, with
         * `n_ff_shared` then the slice's width. hi 0 is the even split (or no split at world 1);
         * an empty slice declares no arm on this rank.
         *
         * WHY A SLICE AND NOT ONE RANK. This family's shared expert is 640 columns, five
         * 128-column blocks, so two ranks cannot halve it -- and computing it on one rank alone
         * makes the ranks unequal at every routed layer, which the block's all-reduce turns into
         * the other rank waiting: about 20 us a layer at one sequence, which costs far more than
         * the arm's own bytes. Slicing it three blocks and two puts both ranks within one block of
         * each other: each computes its columns of the gate/up product, its own K slice of the
         * down projection, scales that partial by the replicated gate, and the all-reduce sums the
         * partials -- `sgate * (down_0 + down_1)` is the arm, since the gate is a per-token
         * scalar. Rounding moves (two partial sums where there was one), so this is checked
         * against the unsplit arm by agreement rather than byte for byte.
         *
         * The arm's stages run behind the routed ones on the same queue (see pass()): a routed
         * GEMM's tail leaves compute units idle, and the arm fills them. */
        int64_t shared_lo = 0, shared_hi = 0;
        int     norm_topk    = 1;   /* renormalise the kept routing weights to sum to one */
        /* THE EXPERT WEIGHT AT FOUR BITS rather than E4M3, which for Qwen3.8-Flash-Next is the
         * difference between a model that fits two cards and one that does not (the registry row
         * for `moe_gemm_w4a8` carries the placement report that says so). It is the STORED
         * encoding of the experts -- `i4*bf16[1x128]` against `fp8_e4m3*bf16[128x128]` -- so it is
         * not a choice made here: probe() reads it off the first expert's encoding, which is the
         * container's, the checkpoint's, or the recipe rad-convert is applying, and declare()
         * refuses an expert whose encoding is not the one probed.
         *
         * A packed 4-bit row is K/2 bytes, so a width shard has to fall on a scale group; the
         * per-rank checks in declare() say so by name. */
        int     expert_w4    = 0;
        /* AND ROTATED: a 128-wide Walsh-Hadamard along K baked into the stored expert weight and
         * into the activation that meets it -- the encoding's `fwht128` transform, read off it as
         * `expert_w4` is, and only meaningful with it (dtype `w4a8h`).
         *
         * WHAT IT COSTS AT RUNTIME, stated because it is the part that is easy to miss. The
         * rotation itself is free: it rides in the quantiser's registers on a launch already
         * being paid. The DOWN projection is free outright -- its input comes from this block's
         * own gated quantiser, which nothing else reads, so that op simply becomes the rotated
         * one. The GATE_UP projection is not: its input is the block's normed activation, which
         * the SHARED expert also reads at fp8 and which must therefore stay unrotated, so a
         * second (q, scale) pair and one extra `had_quant_act_fp8` launch a layer are the price
         * of keeping the shared arm byte-identical.
         *
         * ON ITS OWN A ROTATION IS WORSE FOR ROUNDING: it fixes RANGE, and group-128 absmax
         * already handles range -- 2x worse for RTN int4 on the 27B. It is the basis GPTQ's error
         * feedback quantises in, and the pair is what the served container carries. */
        int     expert_rot   = 0;
        /* AND A CODEBOOK: the rotated codes index a sixteen-entry `table` plane -- libquant's
         * `table=w4nl`, non-uniform levels fitted to the rotated weights -- rather than standing
         * for -8..7 (dtype `w4nla8h`). Only with `expert_rot`, which it implies; the scale grid,
         * the activation and the GEMM are the rotated form's, and the codes view selects the table
         * beside the codes so the kernel's relayout can check it is the table it decodes. */
        int     expert_nl    = 0;
        /* AND ITS SCALES A 64 OF K: one E4M3 byte under a fixed 2^-13 (dtype `w4nl64a8h`, libr4d's
         * kW4G64Scale) where the codebook form has one bf16 a 128 -- the same bytes. Only with
         * `expert_nl`. The scale views select the fixed second level beside the scales, which the
         * relayout checks is the kernel's constant. 2 for a 64, 4 for a 32 (`w4nl32a8h`, twice
         * the scale bytes): the scale groups a 128 of K. */
        int     expert_g64   = 0;
        /* AND ITS CODES FIVE BITS: libquant's 32-level `table=w5nl` at w4nl64a8h's scales (dtype
         * `w5nl64a8h`), the sign of each code its high bit. Only with `expert_g64` 2. This is the
         * PROBED layer's; the 64-group codebook forms may differ by layer and by projection, and
         * declare() resolves each layer's two (MoeFP8::fmt). */
        int     expert_w5    = 0;
        /* PLAIN bf16 -- the checkpoint's own experts, converted without a quantising recipe. Both
         * GEMMs are then `moe_gemm` over one table of whole experts and the bf16 activation, with
         * `silu_mul` between them, and the shared arm follows the trunk (Geom::w_bf16). An expert
         * cannot be sliced by parity here -- a bf16 expert is served whole on its rank, which is
         * also what keeps it one run of the container for the file tier. */
        int     expert_bf16  = 0;
        /* The layer and the GLOBAL expert whose gate_up probe() read the format from and therefore
         * mapped; declare() maps every other. -1 for the layer: none. */
        int64_t probed_layer = -1;
        int64_t probed_expert = 0;
        /* How many experts from `probed_expert` on probe() mapped: it reads past experts a
         * quantised layer keeps plain bf16 (see `prot`) to the first one in the layer's format. */
        int64_t probed_count = 1;
        /* THE GPTQ CALIBRATION TAP. On, this block declares an f32 [K, K] PERSIST accumulator for
         * each of its two expert projections and a `gram_accum` into it, running until
         * RADIANCE_CALIB_TOKENS tokens have passed and then writing the pair to
         * RADIANCE_CALIB_DIR and switching itself off.
         *
         * IT IS A SEPARATE RUN AND NOT A SERVING MODE. The accumulators are 1.34 GB across 48
         * layers at this model's widths, the dump synchronises the stream, and the two extra
         * launches a layer are pure overhead once the corpus is in. Nothing reads the files at
         * serve time -- rad-convert does, in another process, to pack the next container.
         *
         * IT TAPS THE QUANTISER'S OUTPUT, not the bf16 before it, and that is deliberate: the
         * Hessian GPTQ wants is of the operand the expert GEMM actually contracts. Under
         * `expert_rot` that operand is rotated, so the Hessian is in rotated coordinates -- which
         * is the same basis the packer quantises in, and the only basis in which the two agree. */
        int     calib        = 0;
        int64_t rows         = 0;   /* tokens per pass; 0 means the whole step */
    };

    /* WHICH BUFFERS ARE A PASS TALL AND WHICH ARE THE STEP'S, and the division is the whole of what
     * the slicing costs in bookkeeping. Only the four planes that carry one row per (token, slot)
     * are a pass tall -- they are `top_k` times the step and they are what the slicing exists to
     * bound. Everything else is one row per TOKEN, is a few megabytes at any chunk, and is sized
     * at max_tok and sliced by the pass offset like every other activation in this tree.
     *
     * That is not a size optimisation, it is what keeps the pass offset expressible: LinearFP8 and
     * GateQuantFP8 slice their INPUT and their OUTPUT with the same `r0`, so a shared expert whose
     * output buffer were a pass tall would be written past its end on the second pass. The shipped
     * chunk is one pass, so that overrun would show no symptom there. */
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActFP8  h{};           /* [max_tok, n_embd]  normed input, then the block's delta */
        /* THE ROTATED CODES OF THE SAME INPUT, and only `q` and `s` of it are used -- `x` is
         * `h`'s. Declared only when `expert_rot` is set; the routed gate_up reads this pair and
         * the shared expert goes on reading `h`'s. */
        ActFP8  hr{};          /* [max_tok, n_embd]  q + scale only */
        /* the routing -- one row per TOKEN, so the step's height */
        rad_buf logits  = 0;   /* [max_tok, n_expert_all] bf16 -- the MODEL's expert count */
        rad_buf ids     = 0;   /* [max_tok, top_k]      i32 */
        rad_buf ew      = 0;   /* [max_tok, top_k]      bf16 */
        rad_buf eoff    = 0;   /* [n_expert + 1]        i32 */
        rad_buf ecnt    = 0;   /* [n_expert]            i32 */
        /* the expert planes -- one row per (token, slot), so a PASS's height */
        rad_buf sorted  = 0;   /* [rows * top_k]               i32, pass-relative slot indices */
        rad_buf egu     = 0;   /* [rows * top_k, 2 * n_ff_exp] bf16 */
        ActFP8  eff{};         /* [rows * top_k, n_ff_exp] */
        rad_buf edn     = 0;   /* [rows * top_k, n_embd]       bf16, in SORTED order */
        /* back in token order, and the shared expert beside it: the step's height again */
        rad_buf moe     = 0;   /* [max_tok, n_embd]           bf16 */
        rad_buf sgu     = 0;   /* [max_tok, 2 * n_ff_shared]  bf16 */
        ActFP8  sff{};         /* [max_tok, n_ff_shared] */
        rad_buf sout    = 0;   /* [max_tok, n_embd]           bf16 */
        rad_buf sgate   = 0;   /* [max_tok, 1]                bf16 */
    };

    /* The checkpoint side. `experts` is a PREFIX -- "....mlp.experts" -- and the per-expert names
     * are built from it, because a caller cannot pass 256 of them and a format string with the
     * expert in it is a convention this file would have to parse. `shared` is the same for the
     * dense expert beside them; null means the family has none, and `shared_gate` must then be
     * null too.
     *
     * THE CONFIG TRAVELS IN HERE rather than as an argument to declare, and that is not tidiness:
     * it makes this block's declare() arity IDENTICAL to MlpFP8's, which is what lets MtpBlockFP8
     * be a template over the two. A routed feed-forward and a dense one are the same thing in the
     * same position of the same graph, and the only component that should know the difference is
     * the plugin that picks one. */
    struct Src {
        Config      cfg{};
        const char* norm        = nullptr;
        const char* router      = nullptr;   /* mlp.gate.weight, [n_expert, n_embd] bf16 */
        const char* experts     = nullptr;   /* prefix; ".<e>.{gate,up,down}_proj.weight" */
        const char* shared      = nullptr;   /* prefix; ".{gate,up,down}_proj.weight" */
        /* A STACKED CHECKPOINT NAMES TWO TENSORS AND NOT A PREFIX, which is the shape every
         * recent release uses: `mlp.experts.gate_up_proj` [n_expert, 2*n_ff, n_embd] and
         * `mlp.experts.down_proj` [n_expert, n_embd, n_ff], one tensor holding every expert.
         * Set these INSTEAD of `experts` and each declared expert maps to a dim-0 slice
         * (rad_arch.h's map_slice; core/format/checkpoint.cpp takes the slice as a view, and a
         * block-fp8 checkpoint's stacked `_scale_inv` plane at the same index).
         * Note that a stacked gate_up is ALREADY FUSED in the order this block wants -- gate
         * rows first -- so there is no concatenation left to do. */
        const char* experts_gate_up = nullptr;
        const char* experts_down    = nullptr;
        const char* shared_gate = nullptr;   /* mlp.shared_expert_gate.weight, [1, n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer = 0;

    rad_weight w_norm = 0, w_router = 0, w_sgate = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    /* The four runs. Only the FIRST handle and the count reach an issue (RAD_WTAB); the rest are
     * held so that a declaration failure can be reported against the expert it happened on. */
    std::vector<rad_weight> w_gu, w_gus, w_dn, w_dns;

    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_exp{}, gq_sh{};
    LinearFP8    sh_gate_up{}, sh_down{};
    rad_op op_router = 0, op_topk = 0, op_scatter = 0;
    /* `router_topk_scatter`, if it resolves: the two above in one launch. 0 leaves the pair, which
     * is declared either way -- the fused form is bounded by the sort's single-workgroup shape and
     * a prefill chunk is past it. */
    rad_op  op_topk_scatter = 0;
    int64_t topk_scatter_rows = 0;
    /* `router_gemm_topk_scatter`, if it resolves: the router GEMM in front of the pair above, up
     * to `router_fused_rows` tokens, and with `router_fused_gate` the shared expert's gate
     * (`op_sgate`'s work) as well. 0 -- the op did not resolve -- issues the router and the pair
     * as always, and so does a pass of more rows. */
    rad_op  op_router_fused = 0;
    int64_t router_fused_rows = 0;
    bool    router_fused_gate = false;
    /* The rotated re-quantisation of the block input, declared only when `expert_rot` is set and
     * the caller does not already write `w.hr`. */
    rad_op op_hq = 0;
    /* The caller's read writes the rotated pair (HyperConn::Config::rotate), so this block declares
     * no quantiser for it. Set before declare. */
    bool   hr_in = false;
    rad_op op_gu = 0, op_dn = 0, op_gather = 0;
    /* THE PROTECTED EXPERTS (declare): local experts a quantised layer keeps plain bf16 -- the few
     * whose output dwarfs the rest's (docs/MOE-W4.md). The scatter sorts them last (`protect`,
     * `prot_list`), the quantised GEMMs above serve the other `n_reg`, and a bf16 grouped GEMM
     * pair serves these over the trailing run of the offsets: gate_up into `b_pgu`, the gate into
     * `pff`, and down into the same `edn` rows the quantised down left zero. */
    std::vector<int64_t>    prot;
    const char*             prot_list = nullptr;
    int64_t                 n_reg = 0;
    /* How many weight-table classes the quantised GEMMs take: 2 by parity (`moe_gemm_q`), or 4 by
     * number mod 4 (`moe_gemm_q_mod4`, Config::ff_lo4). */
    int                     ncls = 2;
    /* THIS LAYER'S gate_up AND down FORMATS (format_of's numbers), resolved by declare(): the
     * probed layer's for both, except that among the codebook forms at E4M3 scales -- w4nl64a8h,
     * w4nl32a8h, w5nl64a8h -- each projection of each layer takes its own experts'. */
    int                     fmt[2] = { 0, 0 };
    std::vector<rad_weight> w_pgu, w_pdn;
    rad_buf                 b_pgu = 0;
    ActFP8                  pff{};
    GateQuantFP8            gq_prot{};
    rad_op                  op_pgu = 0, op_pdn = 0;
    /* The gather FOLDS the shared arm in wherever there is one: it takes `sout` and `sgate` and
     * writes the block's delta straight into `h.x`, so `w.moe` is never written. */
    bool   gfold = false;
    /* THE WRITE BEHIND THIS BLOCK TAKES THE GATHER up to this many tokens -- HyperConn's
     * gather_rows and gather_rows6, one a wire, set by the architecture after that write declared
     * its fused forms, 0 where it did not. Where gather_taken() says so the block leaves its routed
     * rows unsummed. */
    int64_t gather_out_rows = 0;
    int64_t gather_out_rows6 = 0;
    /* The calibration tap: two accumulators and the two ops that fill them. Zero unless
     * `Config::calib` is set. `calib_rows` and `calib_done` are mutable because `pass` is const
     * and the tap's whole state is a token count -- per block, so a block reaches its target on
     * the same pass every other block does and nothing has to know which layer is last. */
    rad_buf b_gram_gu = 0, b_gram_dn = 0;
    rad_op  op_gram_gu = 0, op_gram_dn = 0;
    /* The domain the tap sums in, which its files say (calib_write): the rotation's width when
     * the GEMMs read rotated codes, the model's own otherwise. */
    uint32_t gram_domain = 0;
    /* THE FILE IS NAMED AFTER THE WEIGHT IT WILL BE USED TO PACK, minus the expert index, and
     * that is the whole protocol between this and the GPTQ quantiser: the recipe hands it the
     * directory (`calib=`) and rad-convert the weight's logical name, and libquant strips
     * `.<e>.weight` off the name to get the file. A name of this tap's own -- "blk3.gate_up" --
     * would have made the quantiser carry a table mapping one to the other, for every
     * architecture. */
    const char* gram_gu_stem = nullptr;
    const char* gram_dn_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;
    /* THE DOWN PROJECTION'S HESSIAN AN EXPERT (rad_arch.h, CalibExperts), beside the pooled one:
     * a layer-sized scratch, `ne` squares of [n_ff_exp, n_ff_exp], a `gram_accum` into each
     * expert's square over that expert's rows, and the sum kept on the host. */
    rad_buf b_gram_dne = 0;
    rad_op  op_gram_dne = 0;
    mutable CalibExperts calib_dne;
    mutable bool         calib_dne_failed = false;
    void calib_experts_pass(RadCtx* c_, int64_t slots) const;
    rad_op op_sgate = 0;
    rad_op op_ar = 0, op_add = 0;
    bool   ar_out = false;
    /* Which wires the consumer of this block's all-reduce takes. Set before declare; see
     * AttnGatedFP8::ar_out_take. */
    int    ar_out_take = kArOutNorm;

    /* THE STORED EXPERT FORMAT, read off the encoding of layer `l`'s first expert's gate_up --
     * its name map declared here, which declare() then leaves out -- into `cfg->expert_w4` and
     * `cfg->expert_rot`. Asked before any layer is declared, because the architecture sizes the
     * rotated activation and tells the connection in front of the block to write it. Nothing
     * known about the weight (a builder with no model behind it) is block fp8. */
    static int probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg);
    /* The format `e` names: 0 block fp8, 1 four-bit, 2 four-bit rotated, 3 plain bf16, 4 the
     * rotated w4nl codebook, 5 and 6 that codebook at E4M3 scales a 64 and a 32 of K, 7 its
     * five-bit form at a 64; RAD_E_DTYPE for an encoding this block serves none of. */
    static int format_of(const RadEncoding& e);
    /* format_of's numbers as the grouped GEMMs' dtype strings. */
    static constexpr const char* kServed[8] = { "fp8a8", "w4a8", "w4a8h", "bf16", "w4nla8h",
                                                "w4nl64a8h", "w4nl32a8h", "w5nl64a8h" };

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                 const Wire& wire, const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* One pass of at most `c.rows` tokens. */
    void pass(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
    void shared_up(RadCtx* c, int64_t T, int64_t r0, int64_t rows, bool gate_done = false) const;
};

inline int MoeFP8::format_of(const RadEncoding& e) {
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return 3;
    if (!rad_enc_is(&e, "affine")) return RAD_E_DTYPE;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    /* The codebook at E4M3 scales a 64 or 32 of K under one fixed f32 (w4nl64a8h, w4nl32a8h), and
     * its five-bit form at a 64 (w5nl64a8h). */
    if (e.n_planes == 4 && sc && sc->dtype == RAD_F8E4M3) {
        const RadEncPlane* s2 = rad_enc_plane(&e, "scale.1");
        const RadEncPlane* t  = rad_enc_plane(&e, "table");
        if (!(s2 && s2->dtype == RAD_F32 && s2->block[0] == 0 && s2->block[1] == 0 && t &&
              t->kind == RAD_PLANE_TABLE && t->dtype == RAD_F32 && t->extent[0] == 1 &&
              sc->block[0] == 1 && std::strcmp(e.transform, "fwht128") == 0))
            return RAD_E_DTYPE;
        if (t->extent[1] == 16 && e.plane[0].dtype == RAD_U4 &&
            (sc->block[1] == RAD_FP8_BLOCK / 2 || sc->block[1] == RAD_FP8_BLOCK / 4))
            return sc->block[1] == RAD_FP8_BLOCK / 2 ? 5 : 6;
        if (t->extent[1] == 32 && e.plane[0].dtype == RAD_U5 && sc->block[1] == RAD_FP8_BLOCK / 2)
            return 7;
        return RAD_E_DTYPE;
    }
    if (!sc || sc->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (e.n_planes == 3) {
        const RadEncPlane* t = rad_enc_plane(&e, "table");
        if (t && t->kind == RAD_PLANE_TABLE && t->dtype == RAD_F32 && t->extent[0] == 1 &&
            t->extent[1] == 16 && e.plane[0].dtype == RAD_U4 && sc->block[0] == 1 &&
            sc->block[1] == RAD_FP8_BLOCK && std::strcmp(e.transform, "fwht128") == 0)
            return 4;
        return RAD_E_DTYPE;
    }
    if (e.n_planes != 2) return RAD_E_DTYPE;
    if (e.plane[0].dtype == RAD_F8E4M3 && !e.transform[0] && sc->block[0] == RAD_FP8_BLOCK &&
        sc->block[1] == RAD_FP8_BLOCK)
        return 0;
    if (e.plane[0].dtype == RAD_I4 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK) {
        if (!e.transform[0]) return 1;
        if (std::strcmp(e.transform, "fwht128") == 0) return 2;
    }
    return RAD_E_DTYPE;
}

/* WHETHER A CHECKPOINT SHIPS ONE LAYER'S EXPERTS STACKED OR ONE BY ONE, as the Src it fills.
 * Recent releases stack them -- `<prefix>.gate_up_proj` and `<prefix>.down_proj` -- and a
 * quantised export (AutoRound's) writes each expert on its own, `<prefix>.<e>.{gate,up,down}_proj`.
 * Each form is asked of the checkpoint through a probe name mapped to its first expert, which the
 * checkpoint answers or does not. Stacked is the answer when neither is found: a container is
 * read by declared name, the same under both maps, so it asks nothing here. `tag` keeps one
 * caller's probe name apart from another's for the same layer. */
inline int moe_expert_sources(RadBuilder* b, Names& nm, const char* tag, const char* prefix,
                              MoeFP8::Src* src) {
    const char* gu = nm.ckpt("%s.gate_up_proj", prefix);
    const char* probe = nm.f("probe.%s.%s.0", tag, prefix);
    RadEncoding e{};
    RAD_ARCH_TRY(map_concat(b, probe, { nm.ckpt("%s.0.gate_proj.weight", prefix),
                                        nm.ckpt("%s.0.up_proj.weight", prefix) }));
    if (weight_enc(b, probe, &e)) {
        src->experts = nm.ckpt("%s", prefix);
        src->experts_gate_up = src->experts_down = nullptr;
    } else {
        src->experts = nullptr;
        src->experts_gate_up = gu;
        src->experts_down    = nm.ckpt("%s.down_proj", prefix);
    }
    return RAD_OK;
}

/* The gate_up and down name maps of one expert: a dim-0 slice of the stacked tensors, or the
 * checkpoint's per-expert gate and up joined along rows. `gate_up` false maps down alone. */
inline int moe_map_expert(RadBuilder* b, Names& nm, const MoeFP8::Src& src, const char* guw,
                          const char* dnw, long long ge, bool gate_up) {
    if (src.experts_gate_up) {
        /* THE SLICE INDEX IS GLOBAL: the checkpoint holds every expert, and this rank declares a
         * window into it. Slicing by the local index would give every rank experts 0..E/W. */
        if (gate_up) RAD_ARCH_TRY(map_slice(b, guw, src.experts_gate_up, (int)ge));
        return map_slice(b, dnw, src.experts_down, (int)ge);
    }
    const char* gate = nm.ckpt("%s.%lld.gate_proj.weight", src.experts, ge);
    const char* up   = nm.ckpt("%s.%lld.up_proj.weight",   src.experts, ge);
    if (gate_up) RAD_ARCH_TRY(map_concat(b, guw, { gate, up }));
    return map_copy(b, dnw, nm.ckpt("%s.%lld.down_proj.weight", src.experts, ge));
}

inline int MoeFP8::probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg) {
    if (!cfg || (!src.experts && !src.experts_gate_up)) return RAD_E_INVAL;
    /* The caller names the expert, in the model's numbering: the first one this rank will own if
     * the experts are split between the ranks, which declare() then skips -- and an expert every
     * rank declares a slice of otherwise, which every rank skips. */
    /* A PLAIN bf16 EXPERT DOES NOT SETTLE IT: a quantised layer keeps up to kMaxProtected of its
     * experts plain, so the probe reads on past them to the first expert in another format, and
     * only a run of plain ones longer than any protected set says the model is bf16. */
    const long long g0 = (long long)cfg->probed_expert;
    const long long all = cfg->n_expert_all > 0 ? (long long)cfg->n_expert_all
                                                : (long long)cfg->n_expert;
    const long long gn = all > g0 ? all : g0 + 1;
    cfg->probed_layer = l;
    int f = 0;
    for (long long ge = g0; ge < gn && ge <= g0 + kMaxProtected; ++ge) {
        const char* guw = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge);
        if (src.experts_gate_up) RAD_ARCH_TRY(map_slice(b, guw, src.experts_gate_up, (int)ge));
        else RAD_ARCH_TRY(map_concat(b, guw, { nm.ckpt("%s.%lld.gate_proj.weight", src.experts, ge),
                                               nm.ckpt("%s.%lld.up_proj.weight", src.experts, ge) }));
        cfg->probed_count = ge - g0 + 1;
        RadEncoding e{};
        f = 0;
        if (weight_enc(b, guw, &e) && (f = format_of(e)) < 0) {
            char en[256];
            rad_enc_format(&e, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and the routed experts are served as block fp8 "
                            "(fp8_e4m3*bf16[128x128]), four-bit (i4*bf16[1x128], optionally "
                            "/fwht128; or u4 codes into a 16-entry f32 table, /fwht128) or plain "
                            "bf16; or the codebook at E4M3 scales a 64 or 32 of K under a fixed "
                            "f32, or its 32-entry five-bit form at a 64. "
                            "Convert it with a recipe that makes one of them.\n", guw, en);
            return RAD_E_DTYPE;
        }
        if (f != 3) break;
    }
    cfg->expert_w4   = (f == 1 || f == 2 || f == 4 || f == 5 || f == 6 || f == 7) ? 1 : 0;
    cfg->expert_rot  = (f == 2 || f == 4 || f == 5 || f == 6 || f == 7) ? 1 : 0;
    cfg->expert_nl   = (f == 4 || f == 5 || f == 6 || f == 7) ? 1 : 0;
    cfg->expert_g64  = (f == 5 || f == 7) ? 2 : f == 6 ? 4 : 0;
    cfg->expert_w5   = f == 7 ? 1 : 0;
    cfg->expert_bf16 = f == 3 ? 1 : 0;
    return RAD_OK;
}

inline int MoeFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                           const Wire& wire, const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; c = src.cfg; w = wire; layer = l;
    ar_out = ar_out_;
    /* 0 means "not expert-parallel": this rank owns the model's whole expert set. Normalised once,
     * here, so nothing below has to ask which case it is in. */
    if (c.n_expert_all <= 0) c.n_expert_all = c.n_expert;
    if (c.n_expert <= 0 || c.top_k <= 0 || c.top_k > c.n_expert_all || c.n_ff_exp <= 0)
        return RAD_E_INVAL;
    /* THE SLICE HAS TO BE INSIDE THE MODEL. A base past the end silently routes nothing to this
     * rank -- a model that serves and drops a fraction of its own feed-forward. */
    if (c.expert_base < 0 || c.expert_base + c.n_expert > c.n_expert_all) return RAD_E_INVAL;
    if (c.rows <= 0) c.rows = g.max_tok;
    /* A rank whose slice of the shared arm is empty (see Config) declares no arm, and the
     * all-reduce carries the others' share. */
    const bool shared = c.n_ff_shared > 0 && src.shared;
    /* Whether `w.sgate` is out of the sharing pool -- set where the shared arm says so, and what
     * the fused router's gate write relies on (see there). */
    bool sgate_concurrent = false;

    /* THE INPUT MAY BE PREPARED BY THE ARCHITECTURE, and `src.norm == nullptr` is how it says so.
     *
     * A pre-norm model's block owns its own norm: one op that adds the previous block's delta into
     * the residual stream, normalises it and quantises it (rad_fp8.h's NormQuantFP8). A GATED
     * RESIDUAL model's does not -- arch/common/rad_block_hc.h's `hc_read` produces the block input
     * from a stream four times as wide, with a data-dependent mix the block has no way to compute,
     * and there is nothing left for a norm here to do. So this block takes `wire.h` as ALREADY
     * normed and quantised and declares no norm weight at all.
     *
     * `ar_in` is refused in that mode rather than ignored: the fused all-reduce lives INSIDE the
     * norm, so with no norm there is nowhere to fold it and a caller that asked would silently get
     * an unreduced input. It emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    /* ---- THE EXPERTS' NAME MAPS, and from them which experts this layer PROTECTS. The maps are
     * how the model is asked what each expert is (rad_weight_encoding): one a LOGICAL weight,
     * gate_up and down, the scale views having none; the probed experts' gate_up is mapped
     * already. Every expert is the format the block was set up for -- one table of experts takes
     * one encoding -- except that a quantised layer may keep a few WHOLE experts plain bf16: the
     * ones whose output dwarfs the rest's, where the same relative rounding error is a large
     * absolute one (docs/MOE-W4.md). Those are sorted last and served by a bf16 GEMM pair of
     * their own; they are found here, before the sort is declared, because the sort takes them. */
    const int64_t ne = c.n_expert;
    for (int64_t e = 0; e < ne; ++e) {
        const long long ge = (long long)(c.expert_base + e);   /* the checkpoint's numbering */
        const bool probed = c.probed_layer == l && ge >= (long long)c.probed_expert &&
                            ge < (long long)(c.probed_expert + c.probed_count);
        RAD_ARCH_TRY(moe_map_expert(b, nm, src, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge),
                                    nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge), ge, !probed));
    }
    prot.clear();
    {
        const int want = c.expert_bf16 ? 3 : c.expert_w5 ? 7 : c.expert_g64 == 4 ? 6
                       : c.expert_g64 ? 5 : c.expert_nl ? 4
                       : c.expert_w4 ? (c.expert_rot ? 2 : 1) : 0;
        auto fmt_of = [&](int64_t e, RadEncoding (&enc)[2], int (&fe)[2], const char* (&nmv)[2]) {
            const long long ge = (long long)(c.expert_base + e);
            nmv[0] = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge);
            nmv[1] = nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge);
            for (int i = 0; i < 2; ++i) {
                fe[i] = want;
                if (weight_enc(b, nmv[i], &enc[i])) fe[i] = format_of(enc[i]);
            }
        };
        /* THE CODEBOOK FORMS AT E4M3 SCALES SHARE EVERYTHING BUT THE GEMM'S DTYPE -- the rotated
         * input, the scale views, the fixed second level, the accumulators -- so the bits can go
         * where the error is: a layer, and either projection of it, may be in any of them whatever
         * the probed layer's. The first expert of the layer kept in one of them decides its two;
         * every other must match. Outside that family a layer is the probed layer's form. */
        auto family = [](int f) { return f == 5 || f == 6 || f == 7; };
        fmt[0] = fmt[1] = want;
        if (family(want))
            for (int64_t e = 0; e < ne; ++e) {
                RadEncoding enc[2]{};
                int fe[2];
                const char* nmv[2];
                fmt_of(e, enc, fe, nmv);
                if (fe[0] == 3 && fe[1] == 3) continue;
                for (int i = 0; i < 2; ++i)
                    if (family(fe[i])) fmt[i] = fe[i];
                break;
            }
        for (int64_t e = 0; e < ne; ++e) {
            RadEncoding enc[2]{};
            int fe[2];
            const char* names[2];
            fmt_of(e, enc, fe, names);
            if (fe[0] == fmt[0] && fe[1] == fmt[1]) continue;
            if (want != 3 && fe[0] == 3 && fe[1] == 3) { prot.push_back(e); continue; }
            const int i = fe[0] != fmt[0] ? 0 : 1;
            char en[256];
            rad_enc_format(&enc[i], en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and this layer's experts are served as %s -- one "
                            "table of experts takes one encoding, and only a whole expert kept "
                            "plain bf16 is served beside it.\n", names[i], en, kServed[fmt[i]]);
            return RAD_E_DTYPE;
        }
    }
    n_reg = ne - (int64_t)prot.size();
    if (!prot.empty()) {
        if ((int64_t)prot.size() > kMaxProtected) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank; the sort "
                            "places at most %lld last\n", l, prot.size(), (long long)kMaxProtected);
            return RAD_E_INVAL;
        }
        /* The quantised GEMM takes its experts as two equal tables by parity -- at four classes
         * (Config::ff_lo4) as four tables, the first n_reg % 4 one expert longer, which any count
         * gives. */
        if (!(c.ff_hi[0] > 0 && c.ff_hi4[0] > 0) && n_reg % 2) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank, which "
                            "leaves %lld quantised; they come in pairs on a rank so the rest split "
                            "into two equal tables by parity\n", l, prot.size(), (long long)n_reg);
            return RAD_E_INVAL;
        }
        /* The tap's per-expert files are named by expert, and the sort here numbers by place. */
        if (c.calib) {
            fprintf(stderr, "radiance: calibrate the bf16 container: layer %d of this one keeps "
                            "experts plain bf16 and sorts them out of order\n", l);
            return RAD_E_INVAL;
        }
        std::string lst;
        for (int64_t e : prot) lst += (lst.empty() ? "" : ",") + std::to_string(e);
        prot_list = nm.ckpt("%s", lst.c_str());
    }

    /* THE ROUTER IS REPLICATED, NOT SHARDED, and that is the property the whole block rests on.
     * Every rank routes every token to the same experts -- the routing is computed from the full
     * residual stream, which every rank holds -- so `sorted_tok` and `expert_offset` are identical
     * across ranks and the expert GEMMs' row layout agrees without a collective. Sharding the
     * router would make two ranks disagree about which expert a token went to, and the all-reduce
     * at the end of the block would then sum two different models. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_inp.weight", l), src.router));
    w_router = decl_w(b, nm.f("blk.%d.ffn_gate_inp.weight", l), RAD_BF16,
                      {c.n_expert_all, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    op_router = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", c.n_expert_all),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_router)),
                  {w.h.x}, {w.logits});
    op_topk = rw(b, RAD_OP(b, "router_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert_all),
                               RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.logits}, {w.ids, w.ew});
    /* THE PROTECTED EXPERTS ARE SORTED LAST (`protect`), and only a layer that has them says so,
     * so every other layer's sort is declared without a protect list. */
    if (!prot_list)
        op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                      RAD_INT("top_k", c.top_k),
                                      RAD_INT("expert_base", c.expert_base)),
                           RAD_NOWEIGHTS),
                       {w.ids}, {w.sorted, w.eoff, w.ecnt});
    else
        op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                      RAD_INT("top_k", c.top_k),
                                      RAD_INT("expert_base", c.expert_base),
                                      RAD_STR("protect", prot_list)),
                           RAD_NOWEIGHTS),
                       {w.ids}, {w.sorted, w.eoff, w.ecnt});

    /* THE TWO ABOVE IN ONE LAUNCH, and it is worth asking for because NEITHER IS WORK. At a decode
     * shape each is a single 256-thread workgroup ranking about a kilobyte, so each costs a full
     * dispatch for no arithmetic, once per layer per step, twice over. The sort reads
     * `expert_ids` and nothing else and the selection writes it, so the launch between them buys a
     * barrier and nothing more. */
    /* R4-01: router_topk_scatter fuses router top-k and histogram generation across both decode
     * and prefill (M >= 128), eliminating the standalone scatter_hist_kernel launch across all
     * 48 MoE layers in prefill. topk_scatter_rows therefore covers all c.rows. */
    if (c.top_k > 0)
        topk_scatter_rows = c.rows;
    if (topk_scatter_rows > 0 && !prot_list)
        op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                RAD_PARAMS(RAD_RANGE("M", 1, topk_scatter_rows),
                                           RAD_INT("n_expert", c.n_expert_all),
                                           RAD_INT("n_local", c.n_expert),
                                           RAD_INT("top_k", c.top_k),
                                           RAD_INT("norm", c.norm_topk),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_INT("expert_base", c.expert_base)),
                                RAD_NOWEIGHTS),
                            {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});
    else if (topk_scatter_rows > 0)
        op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                RAD_PARAMS(RAD_RANGE("M", 1, topk_scatter_rows),
                                           RAD_INT("n_expert", c.n_expert_all),
                                           RAD_INT("n_local", c.n_expert),
                                           RAD_INT("top_k", c.top_k),
                                           RAD_INT("norm", c.norm_topk),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_INT("expert_base", c.expert_base),
                                           RAD_STR("protect", prot_list)),
                                RAD_NOWEIGHTS),
                            {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});

    /* EITHER a prefix or the two stacked tensors, never both and never neither. A checkpoint
     * cannot be read two ways at once and a caller that set both has not decided. */
    const bool stacked = src.experts_gate_up != nullptr;
    if (stacked != (src.experts_down != nullptr) || (stacked == (src.experts != nullptr))) {
        fprintf(stderr, "radiance: a routed feed-forward takes EITHER an `experts` prefix or "
                        "both stacked tensors; got prefix %s, gate_up %s, down %s\n",
                src.experts ? "yes" : "no", src.experts_gate_up ? "yes" : "no",
                src.experts_down ? "yes" : "no");
        return RAD_E_INVAL;
    }

    /* ---- the expert weights, in EIGHT RUNS: gate_up codes, gate_up scales, down codes and down
     * scales, each for the even experts and then the odd ones. See the header: a weight table is a
     * run of consecutive handles, and `moe_gemm_q` takes each projection's experts as two tables
     * by parity -- entry e / 2 of class e % 2 -- so these loops must not be merged. */
    if (ne % 2) {
        fprintf(stderr, "radiance: %lld routed experts on this rank; the grouped GEMM takes them as "
                        "two equal tables by parity, so the count must be even\n", (long long)ne);
        return RAD_E_INVAL;
    }
    /* The quantised experts by their PLACE in the sort -- the protected ones sorted last are not
     * among them -- which is also the expert slot each is declared in, so the routing counts the
     * sort writes in place order credit the right unit. */
    std::vector<int64_t> reg;
    reg.reserve((size_t)n_reg);
    for (int64_t e = 0; e < ne; ++e)
        if (std::find(prot.begin(), prot.end(), e) == prot.end()) reg.push_back(e);
    w_gu.resize((size_t)ne); w_gus.resize((size_t)ne);
    w_dn.resize((size_t)ne); w_dns.resize((size_t)ne);
    /* Each class's width on this rank: the Config's slice when the width is split unevenly, and
     * `n_ff_exp` for every class otherwise. The buffers between the two GEMMs are the widest
     * one's. FOUR CLASSES where the Config offers them (Config::ff_lo4), the parity classes
     * otherwise. */
    const bool split = c.ff_hi[0] > 0 || c.ff_hi4[0] > 0;
    ncls = c.ff_hi4[0] > 0 ? 4 : 2;
    if (ncls == 4 && n_reg < 4) {
        fprintf(stderr, "radiance: layer %d serves %lld quantised experts on this rank; four "
                        "classes need at least one each\n", l, (long long)n_reg);
        return RAD_E_INVAL;
    }
    int64_t qlo[4] = { 0, 0, 0, 0 }, qhi[4] = { 0, 0, 0, 0 };
    int64_t wq[4] = { c.n_ff_exp, c.n_ff_exp, c.n_ff_exp, c.n_ff_exp };
    if (split)
        for (int q = 0; q < ncls; ++q) {
            qlo[q] = ncls == 4 ? c.ff_lo4[q] : c.ff_lo[q];
            qhi[q] = ncls == 4 ? c.ff_hi4[q] : c.ff_hi[q];
            wq[q]  = qhi[q] - qlo[q];
        }
    if (split) {
        int64_t widest = 0;
        bool    bad = false;
        for (int q = 0; q < ncls; ++q) {
            bad = bad || wq[q] <= 0;
            widest = wq[q] > widest ? wq[q] : widest;
        }
        if (bad || c.n_ff_exp != widest) {
            fprintf(stderr, "radiance: the routed experts' %d class slices, the first [%lld, %lld) "
                            "and [%lld, %lld), do not describe a width of %lld\n", ncls,
                    (long long)qlo[0], (long long)qhi[0], (long long)qlo[1], (long long)qhi[1],
                    (long long)c.n_ff_exp);
            return RAD_E_INVAL;
        }
    }
    for (int q = 0; q < ncls; ++q)
        if (wq[q] % RAD_FP8_BLOCK || g.n_embd % RAD_FP8_BLOCK ||
            (split && qlo[q] % RAD_FP8_BLOCK)) {
            fprintf(stderr, "radiance: an expert is %lldx%lld per rank, which is not a whole number "
                            "of %dx%d scale blocks\n",
                    (long long)wq[q], (long long)g.n_embd, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
    /* The codes' dtype, which is what the codes view is declared at, and the planes it selects:
     * a codebook's table rides with its codes, so the relayout sees the table it is asked to
     * decode by. */
    auto edt_of = [&](int f) {
        return f == 7 ? (uint32_t)RAD_U5 : c.expert_nl ? (uint32_t)RAD_U4
             : c.expert_w4 ? (uint32_t)RAD_I4 : (uint32_t)RAD_F8E4M3;
    };
    const uint32_t edt[2] = { edt_of(fmt[0]), edt_of(fmt[1]) };
    const char* const ecodes = c.expert_nl ? "codes,table" : "codes";
    /* UNDER EXPERT PARALLELISM AN EXPERT IS WHOLE, so its weights are not sharded at all: this
     * rank holds all of the columns of the experts it owns and none of the others. Declaring them
     * RAD_SHARD_ROW would have the loader slice each one again -- it refuses by name, because the
     * declared row parts sum to the full width while the rank's share is half of it. */
    const bool ep = c.n_expert != c.n_expert_all;
    if (ep && split) {
        fprintf(stderr, "radiance: a rank either owns whole experts or a slice of every one; this "
                        "configuration asks for both\n");
        return RAD_E_INVAL;
    }
    /* A PACKED 4-BIT ROW SLICES ON ITS SCALE GROUPS AND NOWHERE ELSE: it is K/2 bytes with eight
     * codes a dword, so a width shard along K is a byte range of every row exactly when it falls on
     * a group of 128 -- which the per-rank check above already requires -- and a shard along the
     * rows is whole rows. The rotation is a 128-wide Hadamard inside each group and is cut by
     * neither. */
    /* THE ROTATION IS A PROPERTY OF THE 4-BIT PLANE AND HAS NO MEANING WITHOUT IT. There is no
     * rotated fp8 expert format in this tree -- the fp8 layout hook writes a 128x128 tile scale
     * and has no rotated form -- so `expert_rot` alone is a configuration that cannot be
     * built, and saying so is better than silently serving the unrotated one. */
    if ((c.expert_rot && !c.expert_w4) || (c.expert_nl && !c.expert_rot)) {
        fprintf(stderr, "radiance: rotated experts (expert_rot) require the 4-bit expert format, "
                        "and a codebook (expert_nl) the rotated one; there is no rotated E4M3 "
                        "expert plane and no unrotated codebook\n");
        return RAD_E_INVAL;
    }
    const int  sh_row = ep ? RAD_SHARD_NONE : RAD_SHARD_ROW;
    const int  sh_col = ep ? RAD_SHARD_NONE : RAD_SHARD_COL;
    const int  n_parts = ep ? 0 : 2;
    /* The scale grid the chosen weight format wants: one per 128x128 tile for E4M3, one per ROW
     * per group of 128 along K for 4-bit -- which is what a sixteen-level grid needs, and is the
     * only shape difference between the two declarations. `sdiv` is how many weight rows or
     * columns one scale covers along the split dimension. */
    const int64_t sdiv = c.expert_w4 ? 1 : RAD_FP8_BLOCK;
    /* A FORMAT A STRING, and the string is what selects the grouped GEMM; its layout hook then
     * accepts the experts' encoding or the kernel is not a candidate. One a projection. */
    const char* const edtype[2] = { kServed[fmt[0]], kServed[fmt[1]] };
    /* The scale views: bf16, one a (row, 128 of K) -- or a 128x128 tile for E4M3 -- or at
     * w4nl64a8h one E4M3 a (row, 64 of K), selected with the fixed second level beside it. TWO
     * PLANES MAKE IT A VIEW OF THE LOGICAL WEIGHT, as the codes view is (rad_builder.cpp): it is
     * declared at the weight's extents, its slice is in the weight's columns, and the stored
     * plane under it -- [rows, K/64] bytes -- is the layout hook's. `sgk` is the K one declared
     * scale column covers. */
    const uint32_t sdt    = c.expert_g64 ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
    const int64_t  sgk    = c.expert_g64 ? 1 : RAD_FP8_BLOCK;
    const char*    escale = c.expert_g64 ? "scale,scale.1" : "scale";
    /* A slice of each expert: the gate and up rows [lo, hi) of this class for the gate/up pair,
     * columns [lo, hi) for down, and the scale planes the same slice in their own units. */
    auto span = [&](rad_weight h, int q, int64_t unit) -> int {
        if (!split || !h) return RAD_OK;
        return rad_weight_shard_span(b, h, qlo[q] / unit, qhi[q] / unit);
    };
    const bool rot = c.expert_rot != 0;
    if (c.expert_bf16) {
        /* ---- PLAIN bf16 EXPERTS: ONE TABLE A PROJECTION, of whole experts in expert order --
         * `moe_gemm`'s one weight operand, expert e at entry e -- so each projection's handles are
         * declared in one loop and the two loops are not merged. An expert is its own movement
         * unit as it is in every other format, and its bytes in the container are what the kernel
         * reads, so the file tier can serve it as it lies. */
        if (split || rot || c.expert_w4) {
            fprintf(stderr, "radiance: bf16 experts are whole on their rank (expert parallel, or "
                            "one rank) and unrotated; this configuration %s\n",
                    split ? "slices every expert" : "rotates or packs them");
            return RAD_E_INVAL;
        }
        for (int64_t e = 0; e < ne; ++e) {
            const int64_t parts[2] = { c.n_ff_exp, c.n_ff_exp };
            w_gu[(size_t)e] = decl_w_n(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                               (long long)(c.expert_base + e)),
                                       RAD_BF16, {2 * c.n_ff_exp, g.n_embd}, RAD_ACCESS_CONDITIONAL,
                                       sh_row, grp_expert_slot(l, (int)e, 0), 0, parts, n_parts);
            if (!w_gu[(size_t)e]) return RAD_E_INVAL;
        }
        for (int64_t e = 0; e < ne; ++e) {
            w_dn[(size_t)e] = decl_w(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                             (long long)(c.expert_base + e)),
                                     RAD_BF16, {g.n_embd, c.n_ff_exp}, RAD_ACCESS_CONDITIONAL,
                                     sh_col, grp_expert_slot(l, (int)e, 1));
            if (!w_dn[(size_t)e]) return RAD_E_INVAL;
        }
        /* The same two row orders as the quantised GEMMs, and for the same reason (see a_order
         * above): gate_up gathers token rows, down reads its input in sorted order. */
        op_gu = rw(b, rad_decl_op(b, "moe_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * c.n_ff_exp),
                                     RAD_INT("K", g.n_embd), RAD_INT("n_expert", ne),
                                     RAD_INT("top_k", c.top_k), RAD_STR("a_order", "token"),
                                     RAD_STR("dtype", "bf16")),
                          w_gu.data(), (int)ne),
                   {w.h.x, w.sorted, w.eoff}, {w.egu});
        /* silu(gate) * up into the bf16 plane the down GEMM reads: the gate's bf16 form, whatever
         * the trunk's (rad_fp8.h, GateQuantFP8). */
        Geom gx = g;
        gx.w_bf16 = true;
        RAD_ARCH_TRY(gq_exp.declare(b, gx, w.egu, w.eff, c.n_ff_exp, "silu", c.rows * c.top_k,
                                    true, false));
        op_dn = rw(b, rad_decl_op(b, "moe_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                     RAD_INT("K", c.n_ff_exp), RAD_INT("n_expert", ne),
                                     RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                     RAD_STR("dtype", "bf16")),
                          w_dn.data(), (int)ne),
                   {w.eff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_gu || !op_dn) return RAD_E_INVAL;
    } else {
    /* ---- the eight runs -- sixteen at four classes -- declared as the views the header
     * describes, class by class. */
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            const int64_t parts[2] = { wq[q], wq[q] };
            rad_weight& h = w_gu[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                  (long long)(c.expert_base + e)),
                          nullptr, ecodes, edt[0], {2 * wq[q], g.n_embd},
                          RAD_ACCESS_CONDITIONAL, sh_row, grp_expert_slot(l, (int)s, 0),
                          parts, n_parts);
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, 1));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            const int64_t parts[2] = { wq[q] / sdiv, wq[q] / sdiv };
            rad_weight& h = w_gus[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.scale", l,
                                  (long long)(c.expert_base + e)),
                          nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                               (long long)(c.expert_base + e)),
                          escale, sdt,
                          {c.expert_w4 ? 2 * wq[q] : fp8_blocks(2 * wq[q]),
                           c.expert_w4 ? g.n_embd / sgk : fp8_blocks(g.n_embd)},
                          RAD_ACCESS_CONDITIONAL, sh_row, grp_expert_slot(l, (int)s, 0),
                          parts, n_parts);
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, sdiv));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            rad_weight& h = w_dn[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                  (long long)(c.expert_base + e)),
                          nullptr, ecodes, edt[1], {g.n_embd, wq[q]},
                          RAD_ACCESS_CONDITIONAL, sh_col, grp_expert_slot(l, (int)s, 1));
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, 1));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            rad_weight& h = w_dns[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_down_exps.%lld.scale", l,
                                  (long long)(c.expert_base + e)),
                          nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                               (long long)(c.expert_base + e)),
                          escale, sdt,
                          {c.expert_w4 ? g.n_embd : fp8_blocks(g.n_embd), wq[q] / sgk},
                          RAD_ACCESS_CONDITIONAL, sh_col, grp_expert_slot(l, (int)s, 1));
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, sgk));
        }

    /* ---- the two grouped GEMMs. `M` is the TOKEN count of a pass, not the sorted row count: the
     * kernels read the row count off `y` and the parameter is what the scratch hook and the
     * operand descriptions size from.
     *
     * THE TWO DISAGREE ABOUT `a_order` AND THAT IS THE WHOLE POINT OF THE PARAMETER. gate_up reads
     * the block's normed INPUT, which is one row per TOKEN, so a sorted row gathers
     * `a[sorted_tok[i] / top_k]`. down reads gate_up's output through the gated quantiser, which
     * is already one row per (token, slot) IN THIS ORDER, so a sorted row is row i. Gathering the
     * second one as well hands every slot of a token the FIRST slot's intermediate -- at decode,
     * where a step has one token, all top_k experts then read slot 0 -- and the model still runs
     * and answers plausibly and wrongly. If the reference and the kernel both gather they AGREE,
     * and a workbench that builds the operand at the extent the gather wants cannot see the
     * difference either -- this parameter is the only thing that states which order is meant. */
    /* THE ROUTED ARM'S OWN COPY OF THE INPUT, ROTATED. `w.h.q` is read by the shared expert too
     * and that arm's weight is unrotated E4M3, so this cannot be done in place -- see Config's
     * note on what the isolation buys and what it costs. Declared here, immediately before its
     * one consumer, so the graph order and the issue order agree. */
    if (rot) {
        if (!w.hr.q || !w.hr.s) {
            fprintf(stderr, "radiance: rotated experts need the `hr` (q, scale) pair on the wire; "
                            "the architecture declares the buffers\n");
            return RAD_E_INVAL;
        }
        if (!hr_in)
            op_hq = rw(b, RAD_OP(b, "had_quant_act_fp8",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.hr.q, w.hr.s});
    }
    /* A table op's weights, in its table operands' order: the even experts' codes and scales, then
     * the odd experts' (`w`, `w_scale`, `w_odd`, `w_odd_scale`) -- or at four classes class 0's,
     * then 1's, 2's and 3's. */
    auto tables = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
        std::vector<rad_weight> tab;
        tab.reserve((size_t)n_reg * 2);
        for (int q = 0; q < ncls; ++q) {
            for (int64_t s = q; s < n_reg; s += ncls) tab.push_back(wc[(size_t)s]);
            for (int64_t s = q; s < n_reg; s += ncls) tab.push_back(ws[(size_t)s]);
        }
        return tab;
    };
    /* THE GATE/UP OUTPUT IS TWO PARTS OF THE WIDER CLASS'S WIDTH, gate then up, and the narrower
     * class fills each part's leading columns and zeros the rest -- so the gated quantiser below
     * reads one layout for every row, and a row of the narrower class quantises to zeros past its
     * width, which its down projection does not read. */
    {
        const std::vector<rad_weight> tab = tables(w_gu, w_gus);
        rad_op h = ncls == 4
            ? rad_decl_op(b, "moe_gemm_q_mod4",
                  RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * wq[0]),
                             RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                             RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("a_order", "token"), RAD_STR("dtype", edtype[0]),
                             RAD_INT("N_1", 2 * wq[1]), RAD_INT("K_1", g.n_embd),
                             RAD_INT("N_2", 2 * wq[2]), RAD_INT("K_2", g.n_embd),
                             RAD_INT("N_3", 2 * wq[3]), RAD_INT("K_3", g.n_embd),
                             RAD_INT("parts", 2)),
                  tab.data(), (int)tab.size())
            : rad_decl_op(b, "moe_gemm_q",
                       RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * wq[0]),
                                  RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                  RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                  RAD_STR("a_order", "token"), RAD_STR("dtype", edtype[0]),
                                  RAD_INT("N_odd", 2 * wq[1]), RAD_INT("K_odd", g.n_embd),
                                  RAD_INT("parts", 2)),
                       tab.data(), (int)tab.size());
        op_gu = rw(b, h, {rot ? w.hr.q : w.h.q, rot ? w.hr.s : w.h.s, w.sorted, w.eoff}, {w.egu});
    }
    /* THE DOWN PROJECTION'S INPUT IS FREE TO ROTATE IN PLACE, because `w.eff` is written here and
     * read by exactly one op -- the down GEMM below, which takes the codes alone, so the bf16
     * product is not written. So the rotation is not an extra launch, it is the same launch
     * resolving to the rotated op. */
    RAD_ARCH_TRY(gq_exp.declare(b, g, w.egu, w.eff, c.n_ff_exp, "silu", c.rows * c.top_k,
                                false, rot));
    {
        const std::vector<rad_weight> tab = tables(w_dn, w_dns);
        rad_op h = ncls == 4
            ? rad_decl_op(b, "moe_gemm_q_mod4",
                  RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                             RAD_INT("K", wq[0]), RAD_INT("n_expert", n_reg),
                             RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("a_order", "sorted"), RAD_STR("dtype", edtype[1]),
                             RAD_INT("N_1", g.n_embd), RAD_INT("K_1", wq[1]),
                             RAD_INT("N_2", g.n_embd), RAD_INT("K_2", wq[2]),
                             RAD_INT("N_3", g.n_embd), RAD_INT("K_3", wq[3]),
                             RAD_INT("parts", 1)),
                  tab.data(), (int)tab.size())
            : rad_decl_op(b, "moe_gemm_q",
                       RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                  RAD_INT("K", wq[0]), RAD_INT("n_expert", n_reg),
                                  RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                  RAD_STR("a_order", "sorted"), RAD_STR("dtype", edtype[1]),
                                  RAD_INT("N_odd", g.n_embd), RAD_INT("K_odd", wq[1]),
                                  RAD_INT("parts", 1)),
                       tab.data(), (int)tab.size());
        op_dn = rw(b, h, {w.eff.q, w.eff.s, w.sorted, w.eoff}, {w.edn});
    }
    /* ---- THE PROTECTED EXPERTS' OWN PAIR. Their weights are the layer's, not expert slots: a
     * container's experts in a layer are one fixed-size unit each, and these are four times the
     * bytes -- so they are static, always resident, and joined to the layer's unit with the
     * router. One run of handles a projection, a table of `prot.size()`; the GEMMs are handed the
     * trailing run of the offsets, the rows the quantised down GEMM left zero, and write them.
     *
     * WHERE EVERY RANK HOLDS A SLICE OF EVERY EXPERT, IT HOLDS ONE OF THESE TOO, and an EVEN one:
     * the parity split exists because a 640-wide expert is five 128-wide scale blocks, and a bf16
     * expert has no blocks -- so rank r takes intermediate units [r*w, (r+1)*w) of each, rows of
     * gate and of up and columns of down, and the block's all-reduce sums the partials as it does
     * every other expert's. */
    if (!prot.empty()) {
        const int64_t np = (int64_t)prot.size();
        int64_t wp = c.n_ff_exp;
        int     psh_row = RAD_SHARD_NONE, psh_col = RAD_SHARD_NONE, pparts = 0;
        /* This rank's tile range within one plain expert, in the model's numbering; `phi` 0 is the
         * even split, which needs no span because the loader divides the entry itself. */
        int64_t plo = 0, phi = 0;
        if (split) {
            /* THE EXPERT'S WHOLE WIDTH, not this rank's slice of it: `n_ff_all` is the model's, and
             * where the parity tables end says the same thing at two ranks and nothing else at
             * more -- reading it off `ff_hi` gave a per-rank answer, which happened to be right
             * only when there were two of them. A plain expert has no scale blocks to land on, so
             * the unit here is the GEMM's 64-wide output tile. */
            const int64_t full = g.n_ff_all > 0 ? g.n_ff_all
                                                : (c.ff_hi[0] > c.ff_hi[1] ? c.ff_hi[0]
                                                                          : c.ff_hi[1]);
            /* ONE SLICE A RANK, and it does not have to be the same size on every rank: an uneven
             * share of the 64-wide tiles is still whole tiles, and an uneven share is the only
             * kind available when the width does not divide -- this family's 640 columns are ten
             * tiles, which four ranks take as 3, 3, 2, 2, the first ranks by index taking the
             * extra ones. Every tile has exactly one owner, each rank writes its own columns of
             * gate/up and reads its own of down, and the block's all-reduce sums the partials
             * exactly as it does for the quantised experts and for the shared arm. The even case
             * keeps the even-split declaration verbatim -- no span at all -- so a rank count that
             * divides declares exactly as it did before. */
            psh_row = RAD_SHARD_ROW;
            psh_col = RAD_SHARD_COL;
            pparts  = 2;
            if (full % g.world == 0 && (full / g.world) % 64 == 0) {
                wp = full / g.world;
            } else if (full % 64 || full / 64 < g.world) {
                fprintf(stderr, "radiance: layer %d's plain experts are %lld wide, which %d ranks "
                                "cannot share in 64-wide tiles\n", l, (long long)full, g.world);
                return RAD_E_INVAL;
            } else {
                const int64_t nt = full / 64, base = nt / g.world, extra = nt % g.world;
                const int64_t t0 = (int64_t)g.rank * base + std::min<int64_t>(g.rank, extra);
                const int64_t t1 = t0 + base + ((int64_t)g.rank < extra ? 1 : 0);
                plo = t0 * 64;
                phi = t1 * 64;
                wp  = phi - plo;
            }
        }
        w_pgu.assign((size_t)np, 0);
        w_pdn.assign((size_t)np, 0);
        for (int64_t j = 0; j < np; ++j) {
            const int64_t parts[2] = { wp, wp };
            w_pgu[(size_t)j] = decl_w_n(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                                (long long)(c.expert_base + prot[(size_t)j])),
                                        RAD_BF16, {2 * wp, g.n_embd}, RAD_ACCESS_PER_TOKEN,
                                        psh_row, grp_layer(l), 0, parts, pparts);
            if (!w_pgu[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pgu[(size_t)j], plo, phi));
        }
        for (int64_t j = 0; j < np; ++j) {
            w_pdn[(size_t)j] = decl_w(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                              (long long)(c.expert_base + prot[(size_t)j])),
                                      RAD_BF16, {g.n_embd, wp}, RAD_ACCESS_PER_TOKEN, psh_col,
                                      grp_layer(l));
            if (!w_pdn[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pdn[(size_t)j], plo, phi));
        }
        /* The gate_up output and the gate's, a sorted row each and in bf16: the rows of the
         * regular experts are never written and never read. */
        b_pgu = decl_b(b, nm.f("blk.%d.moe_prot_gu", l), RAD_BF16, {c.rows * c.top_k, 2 * wp});
        pff.x = decl_b(b, nm.f("blk.%d.moe_prot_ff", l), RAD_BF16, {c.rows * c.top_k, wp});
        if (!b_pgu || !pff.x) return RAD_E_INVAL;
        op_pgu = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * wp),
                                      RAD_INT("K", g.n_embd), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "token"),
                                      RAD_STR("dtype", "bf16")),
                           w_pgu.data(), (int)np),
                    {w.h.x, w.sorted, w.eoff}, {b_pgu});
        Geom gx = g;
        gx.w_bf16 = true;
        RAD_ARCH_TRY(gq_prot.declare(b, gx, b_pgu, pff, wp, "silu", c.rows * c.top_k, true,
                                     false));
        op_pdn = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                      RAD_INT("K", wp), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                      RAD_STR("dtype", "bf16")),
                           w_pdn.data(), (int)np),
                    {pff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_pgu || !op_pdn) return RAD_E_INVAL;
    }
    }
    /* ---- the calibration tap, declared LAST of the routed arm so its buffers cannot alias one
     * the arm is still using. They are PERSIST, so the planner would not alias them anyway; the
     * order is kept because declaration order IS the buffer plan and a reader should not have to
     * check which kind a name is to know that. */
    if (c.calib) {
        /* A rank holding a slice of each expert sees a slice of its K, and a Hessian of a slice is
         * not the Hessian the packer needs. Calibration runs at one rank. */
        if (split) {
            fprintf(stderr, "radiance: calibration runs at one rank; this rank holds a slice of "
                            "every expert's width, and its Hessians would be of the slice\n");
            return RAD_E_INVAL;
        }
        if (!calib_dir()) {
            fprintf(stderr, "radiance: calibration is configured but RADIANCE_CALIB_DIR is not "
                            "set; there is nowhere to write the Hessians\n");
            return RAD_E_INVAL;
        }
        /* Both K are multiples of 128 already, which `expert_rot` also requires; gram_accum needs
         * only a multiple of four, but the 128 is what makes a 64-wide output tile sit inside one
         * scale group and there is no width here that does not. */
        if (!c.expert_bf16 && ((g.n_embd % RAD_FP8_BLOCK) || (c.n_ff_exp % RAD_FP8_BLOCK)))
            return RAD_E_SHAPE;
        gram_gu_stem = nm.f("blk.%d.ffn_gate_up_exps", l);
        gram_dn_stem = nm.f("blk.%d.ffn_down_exps", l);
        b_gram_gu = decl_b(b, nm.f("blk.%d.gram_gu", l), RAD_F32, {g.n_embd, g.n_embd},
                           RAD_BUF_PERSIST);
        b_gram_dn = decl_b(b, nm.f("blk.%d.gram_dn", l), RAD_F32, {c.n_ff_exp, c.n_ff_exp},
                           RAD_BUF_PERSIST);
        /* bf16 experts read the model's own activation, and the tap sums exactly that; the
         * quantised ones read codes, rotated when the plane is, and the files say which. */
        gram_domain = rot ? (uint32_t)RAD_FP8_BLOCK : 0u;
        if (c.expert_bf16) {
            op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                           {w.h.x, b_gram_gu}, {b_gram_gu});
            op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                          RAD_INT("n", c.n_ff_exp), RAD_INT("group", RAD_FP8_BLOCK),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                           {w.eff.x, b_gram_dn}, {b_gram_dn});
        } else {
        op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                           RAD_NOWEIGHTS),
                       {rot ? w.hr.q : w.h.q, rot ? w.hr.s : w.h.s, b_gram_gu}, {b_gram_gu});
        /* The down tap's M is the SORTED height, `rows * top_k`, because `w.eff` is one row per
         * (token, slot). gate_up's is the token height -- its GEMM gathers internally, and every
         * token is routed to exactly top_k experts, so the gathered distribution is the token
         * distribution times top_k and H is used only up to a positive scale. */
        op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                      RAD_INT("n", c.n_ff_exp),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                           RAD_NOWEIGHTS),
                       {w.eff.q, w.eff.s, b_gram_dn}, {b_gram_dn});
        }
        /* One expert's Gram a launch: the same op over that expert's run of the sorted rows,
         * into that expert's square of the scratch. TRANSIENT, because nothing on the card reads
         * it after the pass -- it goes back to the host before the next op is issued. */
        b_gram_dne = decl_b(b, nm.f("blk.%d.gram_dn_experts", l), RAD_F32,
                            {c.n_expert * c.n_ff_exp, c.n_ff_exp});
        if (c.expert_bf16)
            op_gram_dne = rw(b, RAD_OP(b, "gram_accum",
                                RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                           RAD_INT("n", c.n_ff_exp),
                                           RAD_INT("group", RAD_FP8_BLOCK),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                             {w.eff.x, b_gram_dne}, {b_gram_dne});
        else
            op_gram_dne = rw(b, RAD_OP(b, "gram_accum",
                                RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                           RAD_INT("n", c.n_ff_exp),
                                           RAD_INT("group", RAD_FP8_BLOCK),
                                           RAD_STR("dtype", "fp8a8")),
                                RAD_NOWEIGHTS),
                             {w.eff.q, w.eff.s, b_gram_dne}, {b_gram_dne});
        if (!b_gram_dne || !op_gram_dne) return RAD_E_INVAL;
    }

    /* THE GATHER FOLDS THE SHARED ARM IN, so `w.moe` is never written: `scale_rows` and a
     * separate gather would be two dispatches a layer, with a 160 KiB round trip through the
     * buffer between them on top. */
    gfold = shared;
    {
        rad_buf grd[5] = { w.edn, w.ew, w.sorted, w.sout, w.sgate };
        rad_buf gwr[1] = { w.h.x };
        op_gather = RAD_OP(b, "moe_gather",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("top_k", c.top_k), RAD_STR("dtype", g.dtype),
                                      RAD_STR("act", gfold ? "sigmoid" : "none")),
                           RAD_NOWEIGHTS);
        if (!op_gather) return RAD_E_INVAL;
        rad_op_reads (b, op_gather, grd, gfold ? 5 : 3);
        rad_op_writes(b, op_gather, gwr, 1);
    }

    /* ---- the shared expert. Ordinary dense linears over the same normed input. */
    if (shared) {
        /* THE SECOND LANE NEEDS THESE OUT OF THE SHARING POOL, and rad_buf_concurrent is where
         * that is said. The arm runs beside the routed GEMMs, so a plan that laid `sgu` over
         * `egu` -- legal from the linear order, which puts one range entirely after the other --
         * would have two streams writing the same arena bytes. It costs a few MiB of arena. */
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgu));
        /* WHAT THE DOWN PROJECTION READS decides what the gate writes, so it is asked first: the
         * codes, or the bf16 plane when the weight is plain -- a bf16 model, or a quantised one
         * that keeps the shared expert at the checkpoint's precision. */
        const char* down_base = nm.f("blk.%d.ffn_down_shexp", l);
        RAD_ARCH_TRY(sh_down.probe(b, nm, g, down_base,
                                   { nm.ckpt("%s.down_proj.weight", src.shared) }));
        /* The code pair the gate writes, E4M3 or the int8 one (ActFP8::q8_fed) -- and an int8
         * down projection that quantises for itself writes the int8 pair on this lane too. */
        if (w.sff.cq()) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cq()));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cs()));
        }
        if (w.sff.q8 && !w.sff.q8_fed && sh_down.probed_i8) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.q8));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.s8));
        }
        if (!w.sff.q || (sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)))
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.x));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sout));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgate));
        sgate_concurrent = true;
        /* The gate/up rows and the down columns of this rank's slice -- the even split, or the
         * uneven one Config states -- and the split of gate from up stays, because that is a
         * property of the stacked checkpoint tensor and not of the rank count. */
        RAD_ARCH_TRY(sh_gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up_shexp", l),
                                        2 * c.n_ff_shared, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { nm.ckpt("%s.gate_proj.weight", src.shared),
                                          nm.ckpt("%s.up_proj.weight",   src.shared) },
                                        w.h, w.sgu, c.rows,
                                        { c.n_ff_shared, c.n_ff_shared },
                                        c.shared_lo, c.shared_hi));
        /* An int8 down projection fed the int8 pair reads no bf16 plane, so none is written. */
        RAD_ARCH_TRY(gq_sh.declare(b, g, w.sgu, w.sff, c.n_ff_shared, "silu", c.rows,
                                   sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)));
        RAD_ARCH_TRY(sh_down.declare(b, nm, g, down_base,
                                     g.n_embd, c.n_ff_shared, RAD_SHARD_COL, grp_layer(l),
                                     { nm.ckpt("%s.down_proj.weight", src.shared) },
                                     w.sff, w.sout, c.rows, {},
                                     c.shared_lo, c.shared_hi));
        /* The gate. REPLICATED like the router and for the same reason: it is computed from the
         * whole residual stream and a sharded copy would give two ranks two different gates for
         * one token. One output column, so the GEMM is skinny and the kernel clamps rather than
         * masks its N tile -- which is why N = 1 is legal here. */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), src.shared_gate));
        w_sgate = decl_w(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), RAD_BF16, {1, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        op_sgate = rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 1),
                                    RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_sgate)),
                     {w.h.x}, {w.sgate});
    }

    /* ---- THE ROUTER GEMM AND `router_topk_scatter` AS ONE OP, with the shared expert's gate in
     * the same launch wherever the layer has one. Asked for wherever the pair itself is declared,
     * up to the pair's own row bound and the skinny GEMM's decode band; a plugin that does not
     * serve it leaves the handle 0 and the router and the pair are issued as declared above.
     * Declared HERE, after the gate's weight, and not beside the router: the op names both
     * weights, and declaring it late only lengthens the planned lifetime of the buffers it
     * touches, never shortens one -- each of them is also declared by the op it stands in for, at
     * that op's place.
     *
     * EXCEPT `w.sgate`, WHICH THE FUSED OP WRITES AHEAD OF ITS DECLARED WRITER. The gate is issued
     * at the router's place, before `op_sgate`'s declared index, so a plan bounded by the linear
     * order could lend `w.sgate`'s bytes to a buffer that is live in between. It cannot here
     * because the shared arm took `w.sgate` out of the sharing pool (rad_buf_concurrent), and the
     * declaration refuses a fused gate on a layer where that has not happened.
     *
     * NOT rw()'s braced lists and not one RAD_OP per combination: the parameter list (a protect
     * list or not), the weights and the write set (the gate or not) are conditional, so each is
     * one array cut to its length. */
    if (op_topk_scatter) {
        router_fused_rows = topk_scatter_rows < kMoeRouterFusedRows ? topk_scatter_rows
                                                                    : kMoeRouterFusedRows;
        const bool gate = op_sgate != 0;
        if (gate && !sgate_concurrent) {
            fprintf(stderr, "radiance: layer %d: the fused router writes the shared gate ahead of "
                            "its declared writer, which needs `sgate` out of the sharing pool, "
                            "and it is not\n", l);
            return RAD_E_STATE;
        }
        const RadParam fp[] = {
            RAD_RANGE("M", 1, router_fused_rows), RAD_INT("K", g.n_embd),
            RAD_INT("n_expert", c.n_expert_all), RAD_INT("n_local", c.n_expert),
            RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
            RAD_STR("dtype", g.dtype), RAD_INT("expert_base", c.expert_base),
            RAD_STR("protect", prot_list),
        };
        const int n_fp = (int)(sizeof(fp) / sizeof(fp[0])) - (prot_list ? 0 : 1);
        const rad_weight fw[2] = { w_router, w_sgate };
        op_router_fused = rad_decl_op(b, "router_gemm_topk_scatter", fp, n_fp, fw, gate ? 2 : 1);
        if (op_router_fused) {
            const rad_buf rd[1] = { w.h.x };
            const rad_buf wr[7] = { w.logits, w.ids, w.ew, w.sorted, w.eoff, w.ecnt, w.sgate };
            rad_op_reads (b, op_router_fused, rd, 1);
            rad_op_writes(b, op_router_fused, wr, gate ? 7 : 6);
        }
        router_fused_gate = op_router_fused && gate;
    }

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});
    return RAD_OK;
}

/* ONE PASS OF AT MOST `c.rows` TOKENS. Every op here is per row -- the norm and the quantiser group
 * along K within a row, the router and the top-k are a row each, the grouped GEMMs give each sorted
 * row its own accumulation, and the gather closes each token's sum over its own slots -- so a pass
 * computes exactly the rows it names.
 *
 * THE EXPERT BUFFERS ARE ADDRESSED FROM ZERO, not from `r0`. They are one pass tall by declaration,
 * and `sorted_tok` indexes them relative to the pass; the only operands that carry the pass offset
 * are the ones that live at the step's height -- the residual stream and the normed input. */


/* The shared arm's first stage: its one-column gate and its gate_up, both over the normed input.
 * The shared expert is ordinary dense linears over that input, and every buffer it touches is the
 * step's height -- so the pass offset goes through all of them exactly as it does through an
 * MlpFP8 slice. */
inline void MoeFP8::shared_up(RadCtx* c_, int64_t T, int64_t r0, int64_t rows,
                              bool gate_done) const {
    /* `gate_done`: the fused router already wrote this pass's gate. */
    if (!gate_done)
    RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_sgate),
                brow_slice(w.sgate, r0, rows, 1));
    sh_gate_up.step(c_, w.h, w.sgu, T, r0, rows);
}

inline void MoeFP8::pass(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    /* `ne` is THIS RANK'S expert count and `ne_all` the model's: the logits plane and the top-k
     * are the model's width, everything after the scatter is this rank's slice. */
    const int64_t ne = c.n_expert, ne_all = c.n_expert_all, k = c.top_k;
    const bool shared = op_sgate != 0;

    if (!ext_in) nq_h.step(c_, T, r0, rows);

    /* THE SHARED ARM ISSUED BEFORE THE ROUTED ONE. It reads the same normed input the routed
     * arm does and writes nothing the routed arm touches until the gather joins them. Issuing
     * the shared arm before op_gu allows the routed pipeline op_gu -> gq_exp -> op_dn to execute
     * strictly back-to-back without interleaving shared-expert weight streaming GEMMs that evict
     * w.egu and w.eff from L2 cache. Furthermore, shared_up runs concurrently with op_router and
     * op_topk_scatter (via Ctx::haz_mark unordered marking), finishing well before op_gather needs it. */

    /* THE ROUTED ARM. */
    /* THE COUNTS GO STRAIGHT INTO THE CORE'S ROW for this layer when it keeps one, so the
     * report at the end of step() hands them over with no copy -- otherwise a card-local blit
     * a routed layer, a whole dependent dispatch for a kilobyte. */
    int32_t* const counts = rad_route_counts(c_, layer, ne);
    const RadOperand ecnt = counts ? praw(counts, RAD_I32, ne) : brows(w.ecnt, ne);
    /* THE FUSED ROUTER where it resolved and the pass is within its rows: the router, the pair
     * below and -- with `router_fused_gate` -- the shared gate, in one launch with the operands
     * the three are handed. Otherwise the router and the pair, as declared. */
    const bool rfused = op_router_fused && rows <= router_fused_rows;
    if (rfused)
        RAD_ISSUE_N(c_, op_router_fused, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_W(w_router), router_fused_gate ? RAD_W(w_sgate) : RAD_NONE,
                    brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                    router_fused_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE);
    else
    RAD_ISSUE_N(c_, op_router, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_router),
                brow_slice(w.logits, r0, rows, ne_all));
    /* One launch or two. `w.ids` is the fused op's OUTPUT where it was the sort's input, and
     * every other operand is the one the pair is handed. */
    if (rfused) {
        /* the fused router wrote all five */
    } else if (op_topk_scatter && rows <= topk_scatter_rows) {
        RAD_ISSUE_N(c_, op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    } else {
        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
        RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }

    /* The rotated re-quantisation of the same input, when the expert plane carries a
     * rotation and the caller's read did not write it. Without a rotation the GEMM reads
     * `h`'s pair. */
    const bool rot = c.expert_rot != 0;
    if (op_hq)
        RAD_ISSUE_N(c_, op_hq, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.hr.q, r0, rows, g.n_embd),
                    brow_slice(w.hr.s, r0, rows, g.n_embd / RAD_FP8_BLOCK));


    /* The two grouped GEMMs, and the weight tables: the FIRST handle of each run and its length.
     * The runs are contiguous by construction -- see declare -- so `w_gu[q] + i` is expert
     * 2 * i + q, and each parity is its own pair of table operands; at four classes expert
     * 4 * i + q, class q holding ceil((n_reg - q) / 4). */
    const int64_t n4[4] = { (n_reg + 3) / 4, (n_reg + 2) / 4, (n_reg + 1) / 4, n_reg / 4 };
    if (c.expert_bf16)
        RAD_ISSUE_N(c_, op_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_WTAB(w_gu[0], ne), brows(w.sorted, rows * k), brows(w.eoff, ne + 1),
                    brows(w.egu, rows * k));
    else if (ncls == 4)
    RAD_ISSUE_N(c_, op_gu, rows,
                brow_slice(rot ? w.hr.q : w.h.q, r0, rows, g.n_embd),
                brow_slice(rot ? w.hr.s : w.h.s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                RAD_WTAB(w_gu[0], n4[0]), RAD_WTAB(w_gus[0], n4[0]),
                brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                RAD_WTAB(w_gu[1], n4[1]), RAD_WTAB(w_gus[1], n4[1]),
                RAD_WTAB(w_gu[2], n4[2]), RAD_WTAB(w_gus[2], n4[2]),
                RAD_WTAB(w_gu[3], n4[3]), RAD_WTAB(w_gus[3], n4[3]),
                brows(w.egu, rows * k));
    else
    RAD_ISSUE_N(c_, op_gu, rows,
                brow_slice(rot ? w.hr.q : w.h.q, r0, rows, g.n_embd),
                brow_slice(rot ? w.hr.s : w.h.s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                RAD_WTAB(w_gu[0], n_reg / 2), RAD_WTAB(w_gus[0], n_reg / 2),
                brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                RAD_WTAB(w_gu[1], n_reg / 2), RAD_WTAB(w_gus[1], n_reg / 2),
                brows(w.egu, rows * k));
    if (shared) shared_up(c_, T, r0, rows, rfused && router_fused_gate);
    gq_exp.step(c_, w.egu, rows * k, 0, rows * k);
    if (shared) gq_sh.step(c_, w.sgu, T, r0, rows);
    if (c.expert_bf16)
        RAD_ISSUE_N(c_, op_dn, rows, brows(w.eff.x, rows * k), RAD_WTAB(w_dn[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.edn, rows * k));
    else if (ncls == 4)
    RAD_ISSUE_N(c_, op_dn, rows,
                brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                RAD_WTAB(w_dn[0], n4[0]), RAD_WTAB(w_dns[0], n4[0]),
                brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                RAD_WTAB(w_dn[1], n4[1]), RAD_WTAB(w_dns[1], n4[1]),
                RAD_WTAB(w_dn[2], n4[2]), RAD_WTAB(w_dns[2], n4[2]),
                RAD_WTAB(w_dn[3], n4[3]), RAD_WTAB(w_dns[3], n4[3]),
                brows(w.edn, rows * k));
    else
    RAD_ISSUE_N(c_, op_dn, rows,
                brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                RAD_WTAB(w_dn[0], n_reg / 2), RAD_WTAB(w_dns[0], n_reg / 2),
                brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                RAD_WTAB(w_dn[1], n_reg / 2), RAD_WTAB(w_dns[1], n_reg / 2),
                brows(w.edn, rows * k));
    /* THE PROTECTED EXPERTS, over the trailing run of the offsets: their sorted rows are the
     * ones the quantised down GEMM above zeroed as past its live count, and the down GEMM
     * here writes them -- after it, which the shared `edn` orders. */
    if (op_pgu) {
        const int64_t np = (int64_t)prot.size();
        const RadOperand poff = brow_slice(w.eoff, n_reg, np + 1, 1);
        RAD_ISSUE_N(c_, op_pgu, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_WTAB(w_pgu[0], np), brows(w.sorted, rows * k), poff,
                    brows(b_pgu, rows * k));
        gq_prot.step(c_, b_pgu, rows * k, 0, rows * k);
        RAD_ISSUE_N(c_, op_pdn, rows, brows(pff.x, rows * k), RAD_WTAB(w_pdn[0], np),
                    brows(w.sorted, rows * k), poff, brows(w.edn, rows * k));
    }
    if (shared) sh_down.step(c_, w.sff, w.sout, T, r0, rows);
    /* THE CALIBRATION TAP, issued between the two GEMMs' inputs being written and the pass
     * ending. It reads exactly the operands the GEMMs above read and writes nothing they
     * touch, so removing it changes no number the model produces -- which is the property
     * that lets a calibration run be compared against a serving one. */
    if (op_gram_gu && !calib_done) {
        if (calib_rows == 0) {
            /* The accumulators are arena memory and arrive holding whatever was there. */
            rad_memset_async(rad_buf_ptr(c_, b_gram_gu), 0,
                             g.n_embd * g.n_embd * 4, rad_stream(c_));
            rad_memset_async(rad_buf_ptr(c_, b_gram_dn), 0,
                             c.n_ff_exp * c.n_ff_exp * 4, rad_stream(c_));
        }
        if (c.expert_bf16) {
            RAD_ISSUE_N(c_, op_gram_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE,
                        brows(b_gram_gu, g.n_embd));
            RAD_ISSUE_N(c_, op_gram_dn, rows * k, brows(w.eff.x, rows * k), RAD_NONE,
                        brows(b_gram_dn, c.n_ff_exp));
        } else {
        RAD_ISSUE_N(c_, op_gram_gu, rows,
                    brow_slice(rot ? w.hr.q : w.h.q, r0, rows, g.n_embd),
                    brow_slice(rot ? w.hr.s : w.h.s, r0, rows,
                               g.n_embd / RAD_FP8_BLOCK),
                    brows(b_gram_gu, g.n_embd));
        RAD_ISSUE_N(c_, op_gram_dn, rows * k,
                    brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                    brows(b_gram_dn, c.n_ff_exp));
        }
        if (op_gram_dne) calib_experts_pass(c_, rows * k);
        calib_rows += rows;
        if (calib_rows >= calib_target()) {
            const bool a = calib_write(c_, b_gram_gu, g.n_embd, calib_rows, gram_gu_stem,
                                       gram_domain);
            const bool bb = calib_write(c_, b_gram_dn, c.n_ff_exp, calib_rows * k,
                                        gram_dn_stem, gram_domain);
            const bool ce = !op_gram_dne ||
                            (!calib_dne_failed &&
                             calib_dne.write(gram_dn_stem, c.expert_base, gram_domain,
                                             rad_rank(c_)));
            calib_done = true;
            if (layer == 0)
                std::fprintf(stderr, "radiance: calibration wrote layer Hessians after %lld "
                                     "tokens to %s%s\n", calib_rows, calib_dir(),
                             (a && bb && ce) ? "" : " (WITH ERRORS)");
        }
    }

    /* THE GATHER LAST, because with the fold it reads the arm. Its operands are the routed slots
     * plus `sout` and `sgate`, and it writes the block's delta straight into `h.x`. */
    if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))
        RAD_ISSUE_N(c_, op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k),
                    gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.h.x, r0, rows, g.n_embd));

    /* A write that takes the all-reduce does it ONCE over the whole step, so it answers on T and
     * this pass has to ask the same question of the same T, not of its own rows. */
    if (op_ar && !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take))
        RAD_ISSUE_N(c_, op_ar, rows * g.n_embd, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    if (op_add)
        RAD_ISSUE_N(c_, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

/* ONE PASS OF THE PER-EXPERT TAP. The expert offsets the scatter wrote decide how many launches
 * there are and where each reads, so they come back first; then the scratch is zeroed, each
 * routed expert's run of the `slots` sorted rows is summed into its square, and the scratch comes
 * back to be added on the host. Both syncs are on the block's own stream, with nothing issued
 * after the tap, so every op the scratch could share storage with is finished before it is
 * written and none is issued until it has been read. */
inline void MoeFP8::calib_experts_pass(RadCtx* c_, int64_t slots) const {
    if (calib_dne_failed) return;
    const int64_t ne = c.n_expert, K = c.n_ff_exp;
    const RadStream s = rad_stream(c_);
    if (calib_dne.ne == 0) calib_dne.init(ne, K);

    std::vector<int32_t> off((size_t)(ne + 1));
    rad_memcpy_async(off.data(), rad_buf_ptr(c_, w.eoff), (ne + 1) * 4, s);
    rad_stream_sync(s);
    if (off[0] != 0 || off[(size_t)ne] < 0 || off[(size_t)ne] > slots) {
        std::fprintf(stderr, "radiance: layer %d's expert offsets end at %d of %lld sorted rows; "
                             "the per-expert calibration stops\n",
                     layer, off[(size_t)ne], (long long)slots);
        calib_dne_failed = true;
        return;
    }

    const int64_t bytes = ne * K * K * 4;
    float* staging = calib_staging(c_, bytes);
    if (!staging) { calib_dne_failed = true; return; }
    void* scratch = rad_buf_ptr(c_, b_gram_dne);
    rad_memset_async(scratch, 0, bytes, s);
    for (int64_t e = 0; e < ne; ++e) {
        const int64_t r0 = off[(size_t)e], n = off[(size_t)e + 1] - r0;
        if (n <= 0) continue;
        if (c.expert_bf16)
            RAD_ISSUE_N(c_, op_gram_dne, n, brow_slice(w.eff.x, r0, n, K), RAD_NONE,
                        brow_slice(b_gram_dne, e * K, K, K));
        else
            RAD_ISSUE_N(c_, op_gram_dne, n, brow_slice(w.eff.q, r0, n, K),
                        brow_slice(w.eff.s, r0, n, K / RAD_FP8_BLOCK),
                        brow_slice(b_gram_dne, e * K, K, K));
        calib_dne.rows[(size_t)e] += n;
    }
    rad_memcpy_async(staging, scratch, bytes, s);
    rad_stream_sync(s);
    calib_dne.add(staging);
}

inline void MoeFP8::step(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    for (int64_t r0 = 0; r0 < T; r0 += c.rows) {
        const int64_t n = (T - r0 < c.rows) ? (T - r0) : c.rows;
        pass(c_, T, r0, n);
    }

    /* ---- THE HANDOFF TO THE HEAT ENGINE, and it is a handoff and not a computation: the buffers
     * below were written by ops this block already issued. `rad_route_report` copies the histogram
     * back ASYNCHRONOUSLY and the heat engine consumes it on the NEXT step, so nothing here blocks
     * the compute stream (spec §5.5). Without it the expert-tiered placement has no measurement to
     * promote on, and every expert stays wherever the warm start put it.
     *
     * IT REPORTS THE LAST PASS, which on the shipped configuration is the whole step: `c.rows`
     * defaults to the prefill chunk, so there is one pass. A deployment that raises the chunk past
     * it gets the last pass's histogram -- a sample of the step rather than the whole of it, which
     * is what the heat engine wants anyway (it is a popularity estimate consumed a step late).
     *
     * `expert_w` IS LEFT NULL rather than filled: RadRouting declares it `const float*` and this
     * block's routing weights are bf16, which is what the gather multiplies. Nothing in the core
     * reads the field -- the heat engine takes `expert_count` -- and handing it a bf16 plane under
     * a float pointer would be a lie that survives until something does. */
    moe_debug_weights(c_, layer, c.top_k, w.ids, w.ew);
    RadRouting r{};
    r.expert_ids    = (const int32_t*)rad_buf_ptr(c_, w.ids);
    r.expert_w      = nullptr;
    const int32_t* const counts = rad_route_counts(c_, layer, c.n_expert);
    r.expert_count  = counts ? counts : (const int32_t*)rad_buf_ptr(c_, w.ecnt);
    r.sorted_tok    = (const int32_t*)rad_buf_ptr(c_, w.sorted);
    r.expert_offset = (const int32_t*)rad_buf_ptr(c_, w.eoff);
    r.top_k         = c.top_k;
    r.n_expert      = c.n_expert;
    rad_route_report(c_, layer, &r);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MOE_FP8_H */
