#ifndef RAD_REF_DLOOP_H
#define RAD_REF_DLOOP_H
#include "dloop_ops.h"
int rad_ref_dloop_ctx_append(const RadArgs*,RadStream);
int rad_ref_dloop_base(const RadArgs*,RadStream);
int rad_ref_dloop_pool(const RadArgs*,RadStream);
int rad_ref_dloop_dartree(const RadArgs*,RadStream);
int rad_ref_dloop_count();
const RadKernelInfo* rad_ref_dloop_at(int);
/* Allocation-free backbone seam. ctxh is left-padded F32[C,H], prefix_embed
 * contains prefix_length BF16[W] rows; scratch has scratch_elements(config). */
void rad_ref_dloop_forward(const RadDLoopConfig&,const uint16_t* weights,
    const float* ctxh,int length,const uint16_t* anchor_embed,
    const uint16_t* const* prefix_embed,int prefix_length,float* y,float* scratch);
#endif
