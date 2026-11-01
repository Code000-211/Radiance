/* ced_gemv.h -- RDNA4 Wave32 CED (Conditional / Clustered Expert Distribution)
 * Low-Rank GEMV Projection Kernel interface.
 */
#pragma once
#include <cstdint>

namespace rad {

struct CedGemvArgs {
    const float* h_t;         /* [d] input hidden state */
    const float* v_down;      /* [r x d] down-projection matrix (row-major) */
    const float* u_up;        /* [E x r] up-projection matrix (row-major) */
    const float* bias;        /* [E] bias vector */
    float*       logits;      /* [E] output logits */
    int          d;           /* hidden dimension (e.g. 2560) */
    int          r;           /* bottleneck rank (32, matches Wave32) */
    int          n_experts;   /* total experts (512) */
};

/* Launch CED low-rank GEMV projection kernel on RDNA4 Wave32:
 * 1 workgroup of 128 threads (4 Wave32 waves).
 * LDS: 128 bytes (32 floats for intermediate bottleneck z).
 * VGPR <= 48 (< 96 for 100% occupancy).
 * Compile-time constants, zero waterfall loops.
 * Temporal hint: NT_RT for matrix loads, LU for h_t.
 * Epilogue: MSG_DEALLOC_VGPRS before wait.
 */
int rad_ced_gemv_launch(const CedGemvArgs& args, void* stream);

}  /* namespace rad */
