/* avx_isa.h -- what this machine can execute, and the one list of ops the whole plugin is built
 * around.
 *
 * ============================== WHY A DISPATCH AT ALL ==============================
 *
 * A kernel plugin is a shared object the engine dlopens (spec.md §2). It is shipped, copied,
 * copied and run on machines nobody compiled it on, so `-march=native` is not an option here even
 * though it is the obvious one: a binary built with it on an AVX-512 part SIGILLs on the first
 * vmovdqu64 anywhere older, and the failure is a signal rather than a diagnostic. The
 * alternative -- compile everything baseline -- gives up most of the machine: an AVX-512 rmsnorm
 * moves 64 bytes a cycle where the SSE2 one moves 16.
 *
 * So: the kernels are compiled ONCE PER ISA LEVEL into separate objects, each with its own
 * -m flags, and ONE baseline-compiled translation unit (avx_dispatch.cpp) picks between them at
 * load. Nothing outside those per-level objects may contain an instruction above the baseline,
 * which is why the dispatcher, the registry and the schemas are all in files with no -m flags on
 * them: a single stray intrinsic in the registry would fault before the dispatch ever ran.
 *
 * ============================== THE FOUR LEVELS ==============================
 *
 *   AVX_LEVEL_SCALAR (0)  x86-64 baseline. SSE2 exists on every x86-64 and the compiler uses it
 *                         for scalar float, but nothing here is hand-vectorised. ALWAYS CORRECT,
 *                         always selectable, and the answer every other level is checked against.
 *   AVX_LEVEL_AVX    (1)  AVX. 256-bit FLOAT only -- Sandy/Ivy Bridge have no 256-bit integer
 *                         ALU and no FMA, so the bf16<->f32 conversions in avx_vec.h split into
 *                         two 128-bit halves and every a*b+c is a separate mul and add.
 *   AVX_LEVEL_AVX2   (2)  AVX2 + FMA + F16C. 256-bit integer, vpermd/vpgatherdd, hardware f16.
 *                         This is the level almost every x86 machine made since 2013 runs.
 *   AVX_LEVEL_AVX512 (3)  AVX-512 F/BW/DQ/VL. 512-bit, and -- what actually matters more than the
 *                         width -- 32 architectural vector registers instead of 16, which is what
 *                         lets the GEMM microkernel hold a 12x2 accumulator block in registers,
 *                         and write masks, which remove the scalar tail loop from every op.
 *
 * AVX-512 IS NOT ONE FEATURE and the check has to be the conjunction: a Knights Landing has F but
 * not BW/DQ/VL, and a kernel that tested `avx512f` alone and then used a byte shuffle would fault
 * on it. __builtin_cpu_supports is used rather than raw CPUID because it also consults XGETBV --
 * a CPU may enumerate AVX-512 while the OS has not enabled the ZMM state, and using it then
 * faults. Missing that check is the single most common way a hand-rolled CPUID dispatch is wrong.
 *
 * ============================== THE OP LIST IS A MACRO ==============================
 *
 * AVX_OP_LIST below is the only place the op set is written down. It generates the dispatch table
 * struct, the four extern table declarations, the baseline thunks the registry rows point at, and
 * the harness's op enumeration. Adding an op means adding one line here and one definition per
 * level; forgetting either is a link error rather than a row that silently resolves to null.
 * libref uses the same technique for its shape hooks (`fn ## _shape`, libref/ref_ops.h).
 */
#ifndef RAD_AVX_ISA_H
#define RAD_AVX_ISA_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AVX_LEVEL_SCALAR = 0,
    AVX_LEVEL_AVX    = 1,
    AVX_LEVEL_AVX2   = 2,
    AVX_LEVEL_AVX512 = 3,
    AVX_N_LEVELS     = 4
};

/* The highest level this CPU can execute, computed once. Safe to call before rad_plugin_open. */
int         avx_isa_detect(void);
/* The level currently selected. Defaults to avx_isa_detect(). */
int         avx_isa_level(void);
const char* avx_isa_name(int level);

/* FORCE A LEVEL, FOR THE CHECKER AND FOR NOTHING ELSE.
 *
 * It is an exported C symbol rather than an environment variable on purpose: this project's rule
 * is that an env switch that changes what a kernel computes is exactly the defect that makes "what
 * actually ran" unanswerable, and the engine must never be able to reach this. A symbol the test
 * binary dlsym's cannot be set by accident in a deployment and does not appear in a shell
 * history. Returns the level actually selected, which is
 * min(requested, detected) -- asking for a level the machine cannot execute is clamped rather than
 * honoured, because honouring it means SIGILL. */
int         avx_isa_force(int level);

/* ================================================================== the op list
 * X(name) -- `name` is both the op name in the conventional vocabulary (docs/OPS.md) and the
 * suffix of every implementation symbol: avx_<name>_sc / _v1 / _v2 / _v3. */
#define AVX_OP_LIST(X)                                                                            \
    /* ---- elementwise and norms */                                                              \
    X(rmsnorm)              X(rmsnorm_add)          X(hc_enter)             X(hc_read)            \
    X(hc_write)             X(hc_mean)              X(mtp_enter)            X(layernorm)          \
    X(grid_embed)           X(add)                                                                \
    X(mul)                  X(silu_mul)             X(gelu)                 X(silu)               \
    X(sigmoid)              X(gather_rows)          X(scatter_rows)         X(ngram_ids)          \
    X(ple_gate)             X(ple_conv)             X(cast)                 X(softmax)            \
    X(scale_rows)                                                                                 \
    /* ---- quantisation and rotation */                                                          \
    X(quant_act_i8)         X(had_quant_act_i8)     X(quant_act_fp8)        X(had_quant_act_fp8)  \
    X(gram_accum)           X(rmsnorm_had_quant_i8) X(gated_had_quant_i8)   X(dequant)            \
    X(rmsnorm_quant_fp8)    X(gated_quant_fp8)                                                    \
    /* ---- gemm and the vocabulary edges */                                                      \
    X(gemm_nt)              X(gemm_nt_bias)         X(gemm_nt_q)            X(gemm_nt_q_bias)     \
    X(gemm_nt_q_gated)      X(embed_lookup)         X(embed_lookup_q)       X(logits_gemm)        \
    /* ---- attention */                                                                          \
    X(attn_paged)           X(attn_dense)           X(kv_store)             X(rope)               \
    X(rope_table)           X(qk_norm_rope)                                                       \
    /* ---- mixture of experts */                                                                 \
    X(router_topk)          X(moe_scatter)          X(moe_gemm)             X(moe_gemm_q)         \
    X(moe_gather)           X(row_topk)             X(row_topk_merge)       X(logit_rerank)     \
    X(moe_gemm_q_mod4)                                                                            \
    /* ---- gated delta net */                                                                    \
    X(gdn_conv_prep)        X(gdn_conv_update)      X(gdn_kkt_solve)        X(gdn_chunk_scan)     \
    X(gdn_recurrent_update) X(gdn_conv_recurrent_update)                X(gdn_gated_rmsnorm)  \
    /* ---- collectives */                                                                        \
    X(all_reduce)           X(all_gather)                                                         \
    /* ---- sampling, and the drafter's token walk */                                             \
    X(sample_penalties)     X(sample_dry)           X(sample_temp)          X(sample_topk)        \
    X(sample_topp)          X(sample_minp)          X(sample_typical)       X(sample_xtc)         \
    X(sample_merge_topk)    X(sample_mask)          X(sample_pick)          X(sample_argmax)      \
    X(dflash_select)

/* The table one ISA level fills. One member per op, same order as AVX_OP_LIST. A missing
 * definition is a link error against the level's own table definition in avx_table.cpp. */
typedef struct AvxKernelTable {
#define AVX_DECL_MEMBER(nm) int (*nm)(const RadArgs*, RadStream);
    AVX_OP_LIST(AVX_DECL_MEMBER)
#undef AVX_DECL_MEMBER
} AvxKernelTable;

/* One per level, defined in the level's own object. The dispatcher picks one. */
extern const AvxKernelTable avx_table_sc;
extern const AvxKernelTable avx_table_v1;
extern const AvxKernelTable avx_table_v2;
extern const AvxKernelTable avx_table_v3;

/* The table for a level, or the scalar table if that level was not built. */
const AvxKernelTable* avx_table_for(int level);
/* The live table -- what the thunks call. Repointed by avx_isa_force. */
const AvxKernelTable* avx_table_live(void);

#ifdef __cplusplus
}
#endif
#endif /* RAD_AVX_ISA_H */
