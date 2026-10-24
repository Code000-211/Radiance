#include "ref_gdn_tree.h"
#include "tree_gdn.h"
#include <cmath>
#include <cstring>

namespace {
float bf(uint16_t x) { uint32_t u=(uint32_t)x<<16;float f;std::memcpy(&f,&u,4);return f; }
uint16_t rb(float x) {
    uint32_t u;std::memcpy(&u,&x,4);
    /* Preserve r4d_gdn_wmma.h::f2bf, including its tie-away rounding. */
    return (uint16_t)((u+0x8000U)>>16);
}
float ld(const RadGdnTreeTensor& t,int a,int b=0,int c=0,int d=0) {
    return static_cast<const float*>(t.data)[rad_gdn_off(t,a,b,c,d)];
}
float lb(const RadGdnTreeTensor& t,int a,int b=0,int c=0) {
    return bf(static_cast<const uint16_t*>(t.data)[rad_gdn_off(t,a,b,c)]);
}
void st(const RadGdnTreeTensor& t,float x,int a,int b=0,int c=0,int d=0) {
    static_cast<float*>(t.data)[rad_gdn_off(t,a,b,c,d)]=x;
}
void sb(const RadGdnTreeTensor& t,float x,int a,int b=0) {
    static_cast<uint16_t*>(t.data)[rad_gdn_off(t,a,b)]=rb(x);
}
int checked(const RadArgs* a,int op,RadGdnTreeCall& c) {
    const int status=rad_gdn_tree_args(a,op,&c);
    if (status!=RAD_OK) return status;
    /* Validate the complete transaction before mutating any output/anchor. */
    for (int s=0;s<c.nseq;++s) if (!rad_gdn_tree_metadata(c,s)) return RAD_E_INVAL;
    return RAD_OK;
}
float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }
float softplus(float x,float thr) {
    if (x>thr) return x;
    return x>0 ? x+std::log(1.0f+std::exp(-x)) : std::log(1.0f+std::exp(x));
}
float dot(const float* x,const float* y) {
    float a=0,b=0;
    for (int i=0;i<64;++i) {
        a=std::fma(x[i],y[i],a);b=std::fma(x[i+64],y[i+64],b);
    }
    return a+b;
}
float rms(const float* x) {
    float total=0;
    for (int wave=0;wave<8;++wave) {
        float v[16];for (int j=0;j<16;++j) v[j]=x[wave*16+j]*x[wave*16+j];
        for (int mask=8;mask;mask>>=1) {
            float old[16];std::memcpy(old,v,sizeof v);
            for (int j=0;j<16;++j) v[j]=old[j]+old[j^mask];
        }
        total+=v[0];
    }
    return total;
}
void replay(float h[128][128],const RadGdnTreeTensor& f,int tok,int head) {
    const float decay=ld(f,tok,head,256);
    for (int row=0;row<128;++row) {
        const float u=ld(f,tok,head,128+row);
        for (int k=0;k<128;++k) h[row][k]=std::fma(u,ld(f,tok,head,k),h[row][k]*decay);
    }
}
} // namespace

int rad_ref_gdn_conv_tree(const RadArgs* a,RadStream) {
    RadGdnTreeCall c{};const int status=checked(a,0,c);if (status!=RAD_OK) return status;
    const int d=128*(2*c.hk+c.hv);
    for (int s=0;s<c.nseq;++s) {
        const int bos=rad_gdn_int(c.t[5],s),end=rad_gdn_int(c.t[5],s+1);
        const int slot=rad_gdn_int(c.t[4],s),ctx=rad_gdn_int(c.t[7],s);
        for (int node=0;node<end-bos;++node) {
            int path[RAD_GDN_TREE_WIDE_CAP],n=0;
            for (int p=node;p>=0;p=rad_gdn_parent(c.t[6],bos,p)) path[n++]=p;
            for (int ch=0;ch<d;++ch) {
                float acc=c.t[2].data ? lb(c.t[2],ch) : 0;
                for (int tap=0;tap<c.width;++tap) {
                    const int lag=c.width-1-tap,pos=ctx+n-1-lag;
                    float x=0;
                    if (lag<n) x=lb(c.t[0],bos+path[lag],ch);
                    else if (pos>=0) x=lb(c.t[3],slot,ch,c.ring ? pos%c.hist_len : c.width-1+n-1-lag);
                    acc=std::fma(lb(c.t[1],ch,tap),x,acc);
                }
                const float value=acc*sigmoid(acc);
                const int which=ch<c.hk*128 ? 8 : ch<2*c.hk*128 ? 9 : 10;
                const int col=which==8 ? ch : which==9 ? ch-c.hk*128 : ch-2*c.hk*128;
                sb(c.t[which],value,bos+node,col);
                /* Preserve the source bits, not a BF16->float->BF16 roundtrip. */
                static_cast<uint16_t*>(c.t[11].data)[rad_gdn_off(c.t[11],bos+node,ch)]=
                    static_cast<const uint16_t*>(c.t[0].data)[rad_gdn_off(c.t[0],bos+node,ch)];
            }
        }
    }
    return RAD_OK;
}

int rad_ref_gdn_recurrent_tree(const RadArgs* a,RadStream) {
    RadGdnTreeCall c{};const int status=checked(a,1,c);if (status!=RAD_OK) return status;
    for (int s=0;s<c.nseq;++s) {
        const int bos=rad_gdn_int(c.t[9],s),end=rad_gdn_int(c.t[9],s+1),slot=rad_gdn_int(c.t[8],s);
        for (int head=0;head<c.hv;++head) {
            float h[128][128];
            for (int node=0;node<end-bos;++node) {
                const int par=rad_gdn_parent(c.t[10],bos,node);
                if (node==0 || par!=node-1) {
                    for (int row=0;row<128;++row) for (int k=0;k<128;++k)
                        h[row][k]=ld(c.t[7],slot,head,row,k);
                    int path[RAD_GDN_TREE_WIDE_CAP],n=0;
                    for (int p=par;p>=0;p=rad_gdn_parent(c.t[10],bos,p)) path[n++]=p;
                    while (n) replay(h,c.t[14],bos+path[--n],head);
                }
                const int tok=bos+node,key_head=head/(c.hv/c.hk);
                float q[128],k[128],o[128];
                for (int j=0;j<128;++j) { q[j]=lb(c.t[0],tok,key_head*128+j);k[j]=lb(c.t[1],tok,key_head*128+j); }
                const float qs=(1.0f/std::sqrt(dot(q,q)+c.l2_eps))*c.scale;
                const float ks=1.0f/std::sqrt(dot(k,k)+c.l2_eps);
                for (int j=0;j<128;++j) { q[j]*=qs;k[j]*=ks;st(c.t[14],k[j],tok,head,j); }
                const float g=-std::exp(ld(c.t[5],head))*softplus(lb(c.t[3],tok,head)+ld(c.t[6],head),c.softplus_thr);
                const float decay=std::exp(g),beta=sigmoid(lb(c.t[4],tok,head));
                st(c.t[14],decay,tok,head,256);
                for (int row=0;row<128;++row) {
                    for (int j=0;j<128;++j) h[row][j]*=decay;
                    const float u=(lb(c.t[2],tok,head*128+row)-dot(h[row],k))*beta;
                    st(c.t[14],u,tok,head,128+row);
                    for (int j=0;j<128;++j) h[row][j]=std::fma(u,k[j],h[row][j]);
                    o[row]=dot(h[row],q);
                }
                const float sc=1.0f/std::sqrt(rms(o)*(1.0f/128)+c.eps);
                for (int row=0;row<128;++row) {
                    const float z=lb(c.t[11],tok,head*128+row);
                    const float gate=c.sigmoid ? sigmoid(z) : z*sigmoid(z);
                    sb(c.t[13],o[row]*(sc*ld(c.t[12],row)*gate),tok,head*128+row);
                }
            }
        }
    }
    return RAD_OK;
}

int rad_ref_gdn_tree_commit(const RadArgs* a,RadStream) {
    RadGdnTreeCall c{};const int status=checked(a,2,c);if (status!=RAD_OK) return status;
    const int d=128*(2*c.hk+c.hv),w=c.width-1;
    for (int s=0;s<c.nseq;++s) {
        const int slot=rad_gdn_int(c.t[1],s),hs=rad_gdn_int(c.t[4],s);
        const int bos=rad_gdn_int(c.t[9],s),len=rad_gdn_int(c.t[7],s),ctx=rad_gdn_int(c.t[8],s);
        for (int head=0;head<c.hv;++head) {
            float h[128][128];
            for (int row=0;row<128;++row) for (int k=0;k<128;++k) h[row][k]=ld(c.t[0],slot,head,row,k);
            for (int j=0;j<len;++j) replay(h,c.t[2],bos+rad_gdn_int(c.t[6],s,j),head);
            for (int row=0;row<128;++row) for (int k=0;k<128;++k) st(c.t[0],h[row][k],slot,head,row,k);
        }
        auto* history=static_cast<uint16_t*>(c.t[3].data);
        const auto* raw=static_cast<const uint16_t*>(c.t[5].data);
        for (int ch=0;ch<d;++ch) {
            if (c.ring) {
                for (int j=0;j<len;++j)
                    history[rad_gdn_off(c.t[3],hs,ch,(ctx+j)%c.hist_len)]=
                        raw[rad_gdn_off(c.t[5],bos+rad_gdn_int(c.t[6],s,j),ch)];
            } else if (w) {
                uint16_t win[16];
                for (int i=0;i<w;++i) win[i]=ctx-w+i<0 ? 0 : history[rad_gdn_off(c.t[3],hs,ch,i)];
                for (int j=0;j<len;++j) {
                    for (int i=0;i<w-1;++i) win[i]=win[i+1];
                    win[w-1]=raw[rad_gdn_off(c.t[5],bos+rad_gdn_int(c.t[6],s,j),ch)];
                }
                for (int i=0;i<w;++i) history[rad_gdn_off(c.t[3],hs,ch,i)]=win[i];
            }
        }
    }
    return RAD_OK;
}
int rad_ref_gdn_tree_count(void) { return RAD_GDN_TREE_OPS; }
const RadKernelInfo* rad_ref_gdn_tree_at(int i) {
    static const RadKernelInfo fused=[] {
        auto r=rad_gdn_tree_row(RAD_GDN_TREE_FUSED_I8G,RAD_DOMAIN_HOST,rad_ref_gdn_conv_recurrent_tree_i8g,true);
        r.scratch=rad_ref_gdn_tree_fused_scratch;return r;
    }();
    static const RadKernelInfo rows[]={
        rad_gdn_tree_row(0,RAD_DOMAIN_HOST,rad_ref_gdn_conv_tree,true),
        rad_gdn_tree_row(1,RAD_DOMAIN_HOST,rad_ref_gdn_recurrent_tree,true),
        rad_gdn_tree_row(2,RAD_DOMAIN_HOST,rad_ref_gdn_tree_commit,true),fused};
    return i>=0 && i<RAD_GDN_TREE_OPS ? rows+i : nullptr;
}
