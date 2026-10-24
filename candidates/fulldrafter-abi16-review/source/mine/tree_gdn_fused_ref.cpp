#include "ref_gdn_tree.h"
#include <cmath>
#include <cstring>

/* This oracle composes the existing unfused host operations. Scratch is supplied
 * by the runtime, so reference placement still performs no step-path allocation. */
int64_t rad_ref_gdn_tree_fused_scratch(const RadArgs* a) {
    if (!a) return RAD_E_INVAL;
    const int64_t hk=rad_param_getdim(a->p,a->n_p,"n_head_k",0);
    const int64_t hv=rad_param_getdim(a->p,a->n_p,"n_head_v",0);
    const int64_t tokens=rad_param_getdim(a->p,a->n_p,"max_tokens",0);
    if (hk<1 || hv<1 || hk>INT32_MAX/384 || hv>INT32_MAX/384 || tokens<1 ||
        tokens>INT64_MAX/(512*(hk+hv))) return RAD_E_SHAPE;
    return tokens*512*(hk+hv); // BF16 Q,K,V,O; every band starts on a 256-byte boundary.
}

namespace {
RadTensor dense_bf16(void* ptr,int64_t t,int64_t channels) {
    RadTensor r{};r.data=ptr;r.dtype=RAD_BF16;r.rank=2;
    r.shape[0]=t;r.shape[1]=channels;r.stride[0]=channels;r.stride[1]=1;
    return r;
}
float bf16(uint16_t x) {
    const uint32_t bits=(uint32_t)x<<16;float v;std::memcpy(&v,&bits,4);return v;
}
/* q8_pack4<true>'s magic-add nearest-even conversion, not lround (tie-away),
 * and not amax/127 followed by 127/amax: both change codes at boundaries. */
uint8_t code(float value,float inv) {
    const float clamped=std::fmin(std::fmax(value*inv,-127.0f),127.0f);
    const float rounded=clamped+12582912.0f;
    uint32_t bits;std::memcpy(&bits,&rounded,4);return (uint8_t)bits;
}
}

int rad_ref_gdn_conv_recurrent_tree_i8g(const RadArgs* a,RadStream stream) {
    RadGdnTreeFusedCall f{};
    int rc=rad_gdn_tree_fused_args(a,&f);if (rc!=RAD_OK) return rc;
    const auto& c=f.geometry;
    for (int s=0;s<c.nseq;++s) if (!rad_gdn_tree_fused_metadata(f,s)) return RAD_E_INVAL;
    const int64_t need=rad_ref_gdn_tree_fused_scratch(a);
    if (need<0) return (int)need;
    if (!a->scratch || a->scratch_bytes<need || (reinterpret_cast<uintptr_t>(a->scratch)&1))
        return RAD_E_SCRATCH;
    auto* base=static_cast<uint16_t*>(a->scratch);
    const int64_t kn=(int64_t)c.tokens*c.hk*128,vn=(int64_t)c.tokens*c.hv*128;
    const RadTensor q=dense_bf16(base,c.tokens,c.hk*128);
    const RadTensor k=dense_bf16(base+kn,c.tokens,c.hk*128);
    const RadTensor v=dense_bf16(base+2*kn,c.tokens,c.hv*128);
    const RadTensor o=dense_bf16(base+2*kn+vn,c.tokens,c.hv*128);
    RadTensor conv[12]{};
    for (int i=0;i<8;++i) conv[i]=a->t[i];
    conv[8]=q;conv[9]=k;conv[10]=v;conv[11]=a->t[16];
    RadArgs ca=*a;ca.t=conv;ca.n_t=12;
    rc=rad_ref_gdn_conv_tree(&ca,stream);if (rc!=RAD_OK) return rc;
    RadTensor recurrent[15]={q,k,v,a->t[8],a->t[9],a->t[10],a->t[11],a->t[12],a->t[13],
                              a->t[5],a->t[6],a->t[14],a->t[15],o,a->t[17]};
    RadArgs ra=*a;ra.t=recurrent;ra.n_t=15;
    rc=rad_ref_gdn_recurrent_tree(&ra,stream);if (rc!=RAD_OK) return rc;
    const auto* values=static_cast<const uint16_t*>(o.data);
    for (int s=0;s<c.nseq;++s) {
        const int bos=rad_gdn_int(f.t[5],s),end=rad_gdn_int(f.t[5],s+1);
        for (int tok=bos;tok<end;++tok) for (int head=0;head<c.hv;++head) {
            float m=0;
            for (int j=0;j<128;++j) m=std::fmax(m,std::fabs(bf16(values[(int64_t)tok*c.hv*128+head*128+j])));
            const float scale=m>0 ? m*(1.0f/127.0f) : 1.0f,inv=1.0f/scale;
            static_cast<float*>(f.t[19].data)[rad_gdn_off(f.t[19],tok,head)]=scale;
            for (int j=0;j<128;++j) static_cast<uint8_t*>(f.t[18].data)[rad_gdn_off(f.t[18],tok,head*128+j)]=
                code(bf16(values[(int64_t)tok*c.hv*128+head*128+j]),inv);
        }
    }
    return RAD_OK;
}
