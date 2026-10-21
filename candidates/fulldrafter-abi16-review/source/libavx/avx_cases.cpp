/* avx_cases.cpp -- the geometry table both harnesses run.
 *
 * ONE TABLE, TWO BINARIES. A benchmark taken at a different extent from the correctness check is a
 * measurement of something else, and the two tables would drift the first time one of them gained
 * a case. tools/opshapes.h makes this argument for rad-kbench and rad-tune; it holds one level
 * down too.
 *
 * WHAT THE VALUES ARE CHOSEN FOR. Not realism -- a real n_embd would make the checker take minutes
 * and would not find anything a 640-wide row does not. They are chosen to break a hand-vectorised
 * kernel:
 *
 *   A ROW WIDER THAN ANY VECTOR, so the main body runs at every level (16 floats at AVX-512).
 *   A ROW THAT DOES NOT DIVIDE IT, because the tail loop is where a hand-vectorised kernel is
 *     wrong and where a level-3 masked store and a level-0 scalar loop can disagree. 645 and 259
 *     are chosen for exactly this: 645 % 16 == 5, % 8 == 5, and 259 % 16 == 3.
 *   MORE THAN ONE ROW, so a kernel that computed row 0 and broadcast it is caught.
 *   M = 1, 7 and 65 ON EVERY GEMM, which is decode, a speculative verify window, and one row past
 *     any register-block edge a microkernel can have.
 *   BOTH SIDES OF EVERY BANDED OPTION -- `causal` on and off, `window` bounded and unbounded,
 *     `norm` on and off, both `a_order` spellings, both rope `mode`s. A parameter exists because
 *     two implementations can disagree about it; a check that ran one value could never see that.
 *
 * Every op appears in f32, where the tolerance is tight enough that only the summation order may
 * differ, and in bf16, where the narrowing is exercised.
 *
 * `bytes` and `flops` are for the bench's rate columns and are left 0 wherever the cost is not a
 * simple function of the geometry -- a sampler that narrows a candidate set in place, a scatter
 * whose work depends on the draw. An invented denominator is worse than no rate at all: it
 * produces a number that looks comparable and is not.
 */
#include "avx_harness.h"

std::vector<Case> build_cases() {
    std::vector<Case> c;
    auto add = [&](const char* op, std::vector<P> p, double bytes = 0, double flops = 0) {
        c.push_back(Case{ op, std::move(p), bytes, flops });
    };
    /* Bytes an elementwise op of `rows` x `n` moves at `w` bytes an element and `k` streams. */
    auto EB = [](double rows, double n, double w, double k) { return rows * n * w * k; };

    for (const char* dt : { "f32", "bf16", "f16" }) {
        const double w = std::strcmp(dt, "f32") == 0 ? 4.0 : 2.0;
        add("rmsnorm",     { HI("M", 7), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(7, 640, w, 2));
        add("rmsnorm",     { HI("M", 3), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt),
                             HF("wadd", 1) }, EB(3, 645, w, 2));
        add("rmsnorm_add", { HI("M", 5), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(5, 640, w, 4));
        add("layernorm",   { HI("M", 5), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt) },
                           EB(5, 645, w, 2));
        add("add",         { HI("M", 9), HI("n", 645), HS("dtype", dt) }, EB(9, 645, w, 3));
        add("mul",         { HI("M", 9), HI("n", 640), HS("dtype", dt) }, EB(9, 640, w, 3));
        add("silu_mul",    { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 3));
        add("gelu",        { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("silu",        { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        add("sigmoid",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("softmax",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("gather_rows", { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        add("scatter_rows", { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 645), HS("act", "sigmoid"), HS("dtype", dt) },
                           EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 640), HS("act", "none"), HS("dtype", dt) },
                           EB(6, 640, w, 2));
    }
    /* Every dtype pair, because `cast` is where the narrowing conventions live and the tolerance
     * for it is 1e-6 -- which is to say exact. A cast that rounded differently from the ABI's RTNE
     * would make every operand of every other op disagree by up to an ulp. */
    for (const char* from : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
        for (const char* to : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
            add("cast", { HI("M", 5), HI("n", 259), HS("from", from), HS("to", to) });

    /* The wide residual stream. hc = 4 and lowrank = 8 is Qwen4-Exp's gated residual, scaled down.
     * `inject` on is the trunk case and off is the final mixer, and the two differ in the operand
     * LIST -- which is what makes running both worth it rather than a duplicate. */
    for (const char* dt : { "f32", "bf16" }) {
        add("hc_enter",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("hc_write",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 0) });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 0), HI("inject", 0),
                           HI("group", 0) });
        /* The fp8 form and its Hadamard-rotated twin, 32-wide groups: four of them a stream. */
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 32), HI("rotate", 1) });
        add("mtp_enter", { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HI("ngroup", 4) });
        add("ple_gate",  { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HF("gate_eps", 1e-6) });
        add("ple_conv",  { HI("M", 9), HI("n", 128), HI("width", 4), HI("dilation", 2),
                           HS("dtype", dt) });
    }
    add("ngram_ids", { HI("M", 9), HI("heads", 3), HI("ngram", 4), HI("eos", 2) });

    /* The DFlash2 tap. hc 4 at a row that does NOT divide the vector width (259 % 16 == 3), the
     * production hc 4 over a 640-wide row, and an hc that is not a power of two, because `* 1/hc`
     * and `/ hc` would differ in the last ulp for one. `dt` is not looped: the op is bf16 only and
     * refuses anything else, so an f32 case would be a DECLINE rather than a check. */
    add("hc_mean", { HI("M", 5), HI("N", 259), HI("hc", 4), HS("dtype", "bf16") });
    add("hc_mean", { HI("M", 7), HI("N", 640), HI("hc", 4), HS("dtype", "bf16") });
    add("hc_mean", { HI("M", 1), HI("N", 128), HI("hc", 3), HS("dtype", "bf16") });

    /* Quantisation. 128 is every real grid; 64 proves a second block edge works. n = 256 keeps
     * both dividing, which the had_* forms require by contract (a rotation block may not straddle
     * the end of a row). */
    for (int group : { 64, 128 }) {
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "f32") }, EB(5, 256, 4, 1.25));
        add("quant_act_fp8",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("had_quant_act_fp8",  { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("had_quant_act_i8",   { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("dequant",            { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("gated_had_quant_i8", { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu"), HI("mode", 0) });
        add("rmsnorm_had_quant_i8", { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                      HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("rmsnorm_quant_fp8",  { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                    HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("gated_quant_fp8",    { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu") });
    }
    /* The fp8 fusions off the grid: a row the group does not divide (260 = 4 x 64 + 4, so the last
     * scale covers four values), f32 activations -- which are narrowed to bf16 all the same -- and
     * the other gate and gain. */
    add("rmsnorm_quant_fp8", { HI("M", 5), HI("n", 260), HF("eps", 1e-6), HI("group", 64),
                               HS("dtype", "f32"), HF("wadd", 1) });
    add("gated_quant_fp8",   { HI("M", 5), HI("n", 260), HI("group", 64), HS("dtype", "f32"),
                               HS("act", "sigmoid") });
    add("gram_accum", { HI("M", 5), HI("n", 128), HI("group", 128), HS("dtype", "fp8_e4m3") },
        0, 2.0 * 5 * 128 * 128);
    /* ...and the plain bf16 activation, which passes no scale. */
    add("gram_accum", { HI("M", 5), HI("n", 260), HI("group", 128), HS("dtype", "bf16") },
        0, 2.0 * 5 * 260 * 260);

    /* GEMM. K = 259 is deliberately not a multiple of anything. */
    for (const char* dt : { "f32", "bf16", "f16" })
        for (int M : { 1, 7, 65 }) {
            const double fl = 2.0 * M * 96 * 259;
            add("gemm_nt",      { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            for (const char* act : { "none", "gelu", "gelu_tanh" })
                add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt),
                                      HS("act", act) }, 0, fl);
            add("logits_gemm",  { HI("M", M), HI("n_vocab", 96), HI("n_embd", 259),
                                  HS("dtype", dt) }, 0, fl);
        }
    /* fp8a8 and fp8a16 ARE THE ONLY TWO libref CAN DESCRIBE, and that is a real limit rather than
     * a gap in this table: `gemm_nt_q` is one op over six weight formats and the SCALE GRID is
     * what differs between them -- per row, per row-group, per output-row-block, an e8m0 exponent
     * byte, with or without a zero point. Every one of those is a property of the kernel that
     * reads it, so libref declines to guess (ref_shapes.cpp says so at length) and a case at
     * "w4a8" would be SKIPPED, not checked. The packed-nibble path is still exercised -- through
     * `dequant`, whose grid the schema does state. */
    for (const char* dt : { "fp8a8", "fp8a16" })
        for (int M : { 1, 65 }) {
            const double fl = 2.0 * M * 96 * 256;
            add("gemm_nt_q",      { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                    HS("dtype", dt) }, 0, fl);
            add("gemm_nt_q_bias", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                    HS("dtype", dt) }, 0, fl);
        }
    /* The gated form, which libref describes at fp8a8 only. N is the WEIGHT's rows, so the output
     * is N/2 wide: 96 is one scale group a row at group 128 and 256 is two. */
    for (int M : { 1, 65 })
        for (const char* act : { "silu", "sigmoid" })
            add("gemm_nt_q_gated", { HI("M", M), HI("N", 192), HI("K", 256), HI("group", 128),
                                     HS("dtype", "fp8a8"), HS("act", act) }, 0,
                2.0 * M * 192 * 256);
    add("gemm_nt_q_gated", { HI("M", 7), HI("N", 512), HI("K", 256), HI("group", 128),
                             HS("dtype", "fp8a8"), HS("act", "silu") }, 0, 2.0 * 7 * 512 * 256);
    for (const char* dt : { "f32", "bf16" }) {
        add("embed_lookup",   { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("wscale", 0) });
        add("embed_lookup_q", { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("wscale", 0.01) });
    }

    /* Attention. q_len 1 is decode, 17 a ragged verify window. `window` 0 is unbounded and 8 the
     * sliding form, which is a different loop bound and where an off-by-one lives. 64 takes
     * the prefill panel form (q_len >= 32, uniform), so the second kernel has coverage too. */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 17, 64 })
            for (int window : { 0, 8 })
                add("attn_paged", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                    HI("block_size", 16), HI("causal", 1), HI("window", window),
                                    HS("q_dtype", dt), HS("kv_dtype", dt), HF("scale", 0.125),
                                    HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125),
                                HI("max_seqlen", 24) });
    for (const char* dt : { "f32", "bf16" })
        add("kv_store", { HI("M", 12), HI("head_dim", 64), HI("n_head_kv", 2),
                          HI("block_size", 16), HS("kv_dtype", dt), HF("k_scale", 1.0),
                          HF("v_scale", 1.0) });
    for (const char* mode : { "neox", "gptj" }) {
        add("rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", mode),
                      HI("rotary_dim", 64) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HF("scale", 1.0),
                              HS("mode", mode), HI("rotary_dim", 64), HF("wadd", 0) });
    }
    add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 10000.0), HF("scale", 1.0),
                        HS("dtype", "f32") });
    /* The multi-component forms: Qwen3-VL's interleaved M-RoPE over a partial rotary, and a vision
     * tower's 2-D axial rotary over a whole 72-wide head. */
    for (const char* dt : { "f32", "bf16" }) {
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 1e7), HF("scale", 1.0), HS("mode", "imrope"),
                      HI("rotary_dim", 64), HS("sections", "11 11 10"), HS("dtype", dt) });
        add("rope", { HI("M", 12), HI("head_dim", 72), HI("n_head", 4), HI("n_head_kv", 4),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "axial"),
                      HI("rotary_dim", 72), HS("sections", "18 18"), HS("dtype", dt) });
        add("grid_embed", { HI("M", 20), HI("n", 72), HI("side", 6), HS("dtype", dt) });
    }

    /* MoE. `a_order` exists because the two grouped GEMMs of one routed block disagree about it,
     * and a check that ran one spelling could never have seen the defect it was added for. */
    for (const char* dt : { "f32", "bf16" }) {
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 1),
                             HS("dtype", dt) });
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 0),
                             HS("dtype", dt) });
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt) });
        /* The shared-expert fold: `shared`/`shared_gate` are always described, so the
         * harness always presents them -- this case names the sigmoid spelling, the plain one
         * above covers "none". */
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt),
                             HS("act", "sigmoid") });
        add("row_topk",    { HI("M", 5), HI("N", 97), HI("R", 4), HI("vocab_off", 0),
                             HS("dtype", dt) });
        add("row_topk_merge", { HI("M", 5), HI("R", 4), HI("world_size", 2), HS("dtype", dt) });
        /* "sorted" and not "slot": libref refuses any other spelling with RAD_E_UNSUPPORTED, and a
           case at a spelling both plugins decline is a case that proves nothing. */
        for (const char* order : { "token", "sorted" }) {
            add("moe_gemm",   { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HS("a_order", order), HS("dtype", dt) });
            /* `fp8a8`, the block-fp8 grid: libref describes a grouped GEMM's weight scale plane
             * only at a dtype that names its grid. */
            add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HI("group", 128), HS("a_order", order),
                                HS("dtype", "fp8a8") });
        }
        /* The two classes at different widths: a gate/up pair whose odd experts are half as wide
         * (the output's second part begins past the odd class's zeros), and a down projection
         * whose odd experts are half as long. */
        add("moe_gemm_q", { HI("M", 9), HI("N", 256), HI("K", 128), HI("N_odd", 128),
                            HI("parts", 2), HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 256), HI("K_odd", 128),
                            HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "sorted"), HS("dtype", "fp8a8") });
        /* Four classes at three widths -- a four-rank slice of a five-block expert is two blocks
         * of one class and one of each other -- over ten experts, so classes 2 and 3 hold one
         * fewer than 0 and 1: the gate/up pair and the down projection. */
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 128), HI("K", 128), HI("N_1", 256),
                                 HI("N_2", 128), HI("N_3", 128), HI("parts", 2),
                                 HI("n_expert", 10), HI("top_k", 2), HI("group", 128),
                                 HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 64), HI("K", 128), HI("K_1", 128),
                                 HI("K_2", 256), HI("K_3", 128), HI("n_expert", 10),
                                 HI("top_k", 2), HI("group", 128), HS("a_order", "sorted"),
                                 HS("dtype", "fp8a8") });
    }
    add("moe_scatter", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("expert_base", 0) });

    /* Gated delta net. head_k = head_v = 32 and chunk 8 keeps the state matrices small enough to
     * read in a dump while still running every loop the full 128/128 shape does.
     *
     * `n_head_k`, `n_head_v` and `n_seq` ARE NOT PARAMETERS OF THESE OPS and are passed anyway.
     * The gdn family reads its head counts off the OPERANDS, so no op names them and no geometry
     * carries them -- which means a tool sizing the operands cold has nothing to size them FROM,
     * and libref's shape hook declines (RAD_E_UNSUPPORTED) rather than inventing a head ratio and
     * calling the result correctness. tools/opshapes.h calls the same three a `ShapeCtx` and takes
     * them from the container's metadata. Here they are stated in the case, which is the same
     * decision made by the same person for the same reason; without them every gdn case is SKIPPED
     * AND COUNTED and the family is never checked at all.
     *
     * `n_seq` 2 rather than 1 deliberately: every gdn op opens by reading `cu`, and a single
     * sequence makes a wrong `cu` walk indistinguishable from a right one. */
    add("gdn_conv_prep", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                           HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_conv_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                             HI("conv_width", 4), HI("max_query_len", 2),
                             HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_kkt_solve", { HI("M", 16), HI("head_k", 32), HI("chunk", 8),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_chunk_scan", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                            HF("scale", 0.176), HI("state_fp16", 0),
                            HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                  HI("n_head_v", 2), HF("scale", 0.176), HF("eps", 1e-6),
                                  HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                  HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    add("gdn_conv_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                       HI("conv_width", 4), HI("n_head_v", 2), HF("scale", 0.176),
                                       HF("eps", 1e-6), HF("l2_eps", 1e-6),
                                       HF("softplus_thr", 20.0), HS("act", "silu"),
                                       HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    for (const char* act : { "silu", "sigmoid" })
        add("gdn_gated_rmsnorm", { HI("M", 12), HI("channels", 128), HF("eps", 1e-6),
                                   HS("act", act) });

    /* Collectives at world_size 1, which is the only width libavx serves and says so in its row. */
    add("all_reduce", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("exact", 1),
                        HI("min_bytes", 0), HI("hops", 1) });
    add("all_gather", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("row", 0) });

    /* Sampling. n_vocab 517 is above every vector width and divides nothing; n_cand 12 is a real
     * candidate set after a top-k. */
    for (const char* op : { "sample_penalties", "sample_dry", "sample_temp", "sample_mask",
                            "sample_argmax" })
        add(op, { HI("M", 5), HI("n_vocab", 517) });
    add("sample_topk", { HI("M", 5), HI("n_vocab", 517), HI("vocab_off", 0) });
    for (const char* op : { "sample_topp", "sample_minp", "sample_typical", "sample_xtc",
                            "sample_pick", "sample_merge_topk" })
        add(op, { HI("M", 5), HI("n_cand", 12) });

    /* The drafter's walk: four steps over eight candidates, so every step after the first reads its
     * predecessor off the walk. Rank 64 is whole vectors at every level and 37 is not; a stride of
     * 2 reads every other anchor. */
    add("dflash_select", { HI("M", 3), HI("steps", 4), HI("top_k", 8), HI("rank", 64),
                           HI("n_vocab", 300), HI("anchor_stride", 1), HS("dtype", "bf16") });
    add("dflash_select", { HI("M", 5), HI("steps", 3), HI("top_k", 12), HI("rank", 37),
                           HI("n_vocab", 97), HI("anchor_stride", 2), HS("dtype", "bf16") });

    /* ================================================================== speed-only, at model size
     *
     * WHY THESE EXIST SEPARATELY. Everything above is sized to break a kernel, not to represent
     * one -- a 640-wide row exercises every code path a 2048-wide row does and the checker would
     * spend minutes inside libref to learn nothing more. But SPEED is decided by the cache
     * hierarchy, and a shape that fits in L1 answers a different question from the one a
     * deployment asks. rad-avx-check skips every case below; rad-avx-bench runs them.
     *
     * The geometry is a ~1.5B dense model -- n_embd 2048, n_ff 5632, n_vocab 32000, 16 heads of
     * 128 -- which is the size a CPU plugin plausibly serves. M = 1 is decode, 64 a speculative
     * verify window, 512 a prefill chunk.
     *
     * `logits_gemm` AT 32000 x 2048 IS THE ONE THAT IS NOT A GEMM. Its weight is 128 MiB in bf16,
     * which fits in no cache on any part this runs on, so whatever the flop count says it is a memory
     * benchmark -- and reading it against the DRAM triad rather than the FMA ceiling is the only
     * honest way to read it. It is in this list precisely because no other case can show that. */
    auto speed = [&](const char* op, std::vector<P> p, double bytes, double flops) {
        c.push_back(Case{ op, std::move(p), bytes, flops, true });
    };
    for (int M : { 1, 64, 512 }) {
        speed("gemm_nt", { HI("M", M), HI("N", 2048), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 2048 * 2048);
        speed("gemm_nt", { HI("M", M), HI("N", 5632), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 5632 * 2048);
        speed("rmsnorm", { HI("M", M), HI("n", 2048), HF("eps", 1e-6), HS("dtype", "bf16"),
                           HF("wadd", 0) }, (double)M * 2048 * 2 * 2, 0);
        speed("silu_mul", { HI("M", M), HI("n", 5632), HS("dtype", "bf16") },
              (double)M * 5632 * 3 * 2, 0);
        speed("softmax", { HI("M", M), HI("n", 2048), HS("dtype", "f32") },
              (double)M * 2048 * 4 * 2, 0);
        speed("quant_act_fp8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                 HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_fp8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                     HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                    HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
    }
    /* GDN at qwen35 model shape (k128/v128, chunk 64, 16 key heads, 48 value heads): the
     * prefill heart (chunk_scan over a 512 chunk) and the decode heart (recurrent_update at
     * q_len 1). NO RATE: the traffic denominator depends on the state layout, so a time and no
     * rate is the honest column. libref runs these scalar -- keep --iters low. */
    speed("gdn_chunk_scan", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_chunk_scan", { HI("M", 128), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_recurrent_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                                    HI("n_head_v", 48), HF("scale", 0.176), HF("eps", 1e-6),
                                    HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                    HI("state_fp16", 0), HI("n_head_k", 16), HI("n_seq", 2) }, 0, 0);
    speed("gdn_kkt_solve", { HI("M", 512), HI("head_k", 128), HI("chunk", 64),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv prep at model shape: conv_dim 10240 = 2*16*128 + 48*128, width 4, over a 512 chunk. */
    speed("gdn_conv_prep", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                             HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv update at decode shape: one new token a sequence, same widths. */
    speed("gdn_conv_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                               HI("conv_width", 4), HI("max_query_len", 1),
                               HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("logits_gemm", { HI("M", 1), HI("n_vocab", 32000), HI("n_embd", 2048),
                           HS("dtype", "bf16") }, (double)32000 * 2048 * 2, 2.0 * 32000 * 2048);
    speed("sample_argmax", { HI("M", 1), HI("n_vocab", 32000) }, (double)32000 * 4, 0);
    speed("sample_topk", { HI("M", 1), HI("n_vocab", 32000), HI("vocab_off", 0) }, 0, 0);
    /* Decode attention over a real context: 16 query heads in groups of 4, 2048 keys already in
     * the cache. This is the shape a decode step spends its time in. */
    speed("attn_paged", { HI("q_len", 1), HI("head_dim", 128), HI("gqa", 4), HI("block_size", 16),
                          HI("causal", 1), HI("window", 0), HS("q_dtype", "bf16"),
                          HS("kv_dtype", "bf16"), HF("scale", 0.088), HI("max_ctx", 2048),
                          HI("n_head", 16), HI("max_seqs", 1) },
          /* NO RATE. The KV bytes an attention row reads depend on the CONTEXT LENGTH the
             draw put in `seqused`, which this table does not set and cannot see, so any
             denominator here would be invented. A time and no rate is the honest column. */
          0, 0);
    /* Prefill attention: 512 queries over 512 cached keys, same heads. */
    speed("attn_paged", { HI("q_len", 512), HI("head_dim", 128), HI("gqa", 4),
                          HI("block_size", 16), HI("causal", 1), HI("window", 0),
                          HS("q_dtype", "bf16"), HS("kv_dtype", "bf16"), HF("scale", 0.088),
                          HI("max_ctx", 512), HI("n_head", 16), HI("max_seqs", 1) },
          0, 0);
    /* Rope at model shape: 512 tokens, 16+4 heads of 128. Time-only (libref scalar). */
    speed("rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16), HI("n_head_kv", 4),
                    HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "neox"),
                    HI("rotary_dim", 128) }, 0, 0);
    speed("qk_norm_rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16),
                            HI("n_head_kv", 4), HF("theta", 10000.0), HF("eps", 1e-6),
                            HF("scale", 1.0), HS("mode", "neox"), HI("rotary_dim", 128),
                            HF("wadd", 0) }, 0, 0);

    return c;
}
