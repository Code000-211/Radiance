#include "ref_ple_tree.h"
#include <cmath>
#include <cstring>

namespace {
int checked(const RadArgs* a,int op,RadPleTreeCall& c) {
    const int rc=rad_ple_tree_args(a,op,&c);
    if (rc!=RAD_OK) return rc;
    /* Validate the complete transaction before touching any output or history. */
    for (int s=0;s<c.nseq;++s) if (!rad_ple_tree_metadata(c,s)) return RAD_E_INVAL;
    return RAD_OK;
}
uint16_t raw(const RadPleTreeTensor& t,int a,int b=0,int d=0) {
    return static_cast<const uint16_t*>(t.data)[rad_ple_off(t,a,b,d)];
}
float bf(const RadPleTreeTensor& t,int a,int b=0,int d=0) {
    return rad_bf16_to_f32(raw(t,a,b,d));
}
int64_t i64(const RadPleTreeTensor& t,int i) {
    return static_cast<const int64_t*>(t.data)[rad_ple_off(t,i)];
}
void store_raw(const RadPleTreeTensor& t,uint16_t v,int a,int b=0,int d=0) {
    static_cast<uint16_t*>(t.data)[rad_ple_off(t,a,b,d)]=v;
}
void store_int(const RadPleTreeTensor& t,int32_t v,int a,int b=0) {
    static_cast<int32_t*>(t.data)[rad_ple_off(t,a,b)]=v;
}
void prefill_history(const RadPleTreeCall& c,int s) {
    const int bos=rad_ple_int(c.t[4],s),len=rad_ple_int(c.t[4],s+1)-bos;
    const int slot=rad_ple_int(c.t[3],s),ctx=rad_ple_int(c.t[6],s);
    for (int ch=0;ch<c.n;++ch) {
        if (c.ring) {
            const int first=len>c.conv_len ? len-c.conv_len : 0;
            for (int j=first;j<len;++j)
                store_raw(c.t[2],raw(c.t[0],bos+j,ch),slot,ch,(ctx+j)%c.conv_len);
        } else {
            uint16_t win[RAD_PLE_TREE_HIST_MAX];
            for (int i=0;i<c.hist;++i) {
                const int rel=len-c.hist+i;
                win[i]=rel>=0 ? raw(c.t[0],bos+rel,ch) :
                    ctx+rel<0 ? 0 : raw(c.t[2],slot,ch,c.hist+rel);
            }
            for (int i=0;i<c.hist;++i) store_raw(c.t[2],win[i],slot,ch,i);
        }
    }
}
} // namespace

int rad_ref_ngram_ids_tree(const RadArgs* a,RadStream) {
    RadPleTreeCall c{};const int rc=checked(a,RAD_PLE_TREE_IDS,c);
    if (rc!=RAD_OK) return rc;
    for (int h=0;h<c.heads;++h) {
        const int64_t v=i64(c.t[7],h),off=i64(c.t[8],h);
        if (v<=0 || off<0 || off>INT32_MAX || v-1>INT32_MAX-off) return RAD_E_SHAPE;
    }
    const int per=c.heads/(c.ngram-1);
    for (int s=0;s<c.nseq;++s) {
        const int bos=rad_ple_int(c.t[3],s),end=rad_ple_int(c.t[3],s+1);
        const int slot=rad_ple_int(c.t[2],s),ctx=rad_ple_int(c.t[5],s);
        for (int node=0;node<end-bos;++node) {
            int32_t sh[8];sh[0]=rad_ple_int(c.t[0],bos+node);
            for (int k=1;k<c.ngram;++k) {
                const int src=rad_ple_tree_source(c,bos,node,k);
                sh[k]=sh[k-1]==c.eos ? c.eos : src>=0 ? rad_ple_int(c.t[0],bos+src) :
                    ctx<-src ? c.eos : rad_ple_int(c.t[1],slot,c.ngram-1+src);
            }
            uint64_t bits=0;
            for (int k=0;k<c.ngram;++k) {
                bits^=(uint64_t)(int64_t)sh[k]*(uint64_t)i64(c.t[6],k);
                if (!k) continue;
                int64_t mixed;std::memcpy(&mixed,&bits,sizeof mixed);
                for (int h=(k-1)*per;h<k*per;++h) {
                    const int64_t v=i64(c.t[7],h),r=mixed%v;
                    store_int(c.t[9],(int32_t)((r<0 ? r+v : r)+i64(c.t[8],h)),bos+node,h);
                }
            }
        }
    }
    return RAD_OK;
}

int rad_ref_ple_conv_tree(const RadArgs* a,RadStream) {
    RadPleTreeCall c{};const int rc=checked(a,RAD_PLE_TREE_CONV,c);
    if (rc!=RAD_OK) return rc;
    for (int s=0;s<c.nseq;++s) {
        const int bos=rad_ple_int(c.t[4],s),end=rad_ple_int(c.t[4],s+1);
        const int slot=rad_ple_int(c.t[3],s),ctx=rad_ple_int(c.t[6],s);
        for (int node=0;node<end-bos;++node) for (int ch=0;ch<c.n;++ch) {
            float acc=0;
            for (int tap=0;tap<c.width;++tap) {
                const int src=rad_ple_tree_source(c,bos,node,c.hist-tap*c.dilation);
                const float x=src>=0 ? bf(c.t[0],bos+src,ch) : ctx<-src ? 0.0f :
                    bf(c.t[2],slot,ch,c.ring ? (ctx+src)%c.conv_len : c.hist+src);
                const float product=x*bf(c.t[1],ch,tap);
                acc=acc+product;
            }
            const float residual=c.t[7].data ? bf(c.t[7],bos+node,ch) : 0.0f;
            store_raw(c.t[8],rad_f32_to_bf16(residual+acc/(1.0f+std::exp(-acc))),bos+node,ch);
            if (c.t[9].data) store_raw(c.t[9],raw(c.t[0],bos+node,ch),bos+node,ch);
        }
    }
    if (c.prefill) for (int s=0;s<c.nseq;++s) prefill_history(c,s);
    return RAD_OK;
}

int rad_ref_ple_tree_commit(const RadArgs* a,RadStream) {
    RadPleTreeCall c{};const int rc=checked(a,RAD_PLE_TREE_COMMIT,c);
    if (rc!=RAD_OK) return rc;
    for (int s=0;s<c.nseq;++s) {
        const int ts=rad_ple_int(c.t[1],s),cs=rad_ple_int(c.t[3],s);
        const int bos=rad_ple_int(c.t[6],s),ctx=rad_ple_int(c.t[8],s),len=rad_ple_int(c.t[10],s);
        int32_t tok[7];
        for (int i=0;i<c.ngram-1;++i) {
            const int rel=len-(c.ngram-1)+i;
            tok[i]=rel>=0 ? rad_ple_int(c.t[4],bos+rad_ple_int(c.t[9],s,rel)) :
                ctx+rel<0 ? c.eos : rad_ple_int(c.t[0],ts,c.ngram-1+rel);
        }
        for (int i=0;i<c.ngram-1;++i) store_int(c.t[0],tok[i],ts,i);
        for (int ch=0;ch<c.n;++ch) {
            if (c.ring) {
                for (int j=0;j<len;++j)
                    store_raw(c.t[2],raw(c.t[5],bos+rad_ple_int(c.t[9],s,j),ch),cs,ch,(ctx+j)%c.conv_len);
            } else {
                uint16_t win[RAD_PLE_TREE_HIST_MAX];
                for (int i=0;i<c.hist;++i) {
                    const int rel=len-c.hist+i;
                    win[i]=rel>=0 ? raw(c.t[5],bos+rad_ple_int(c.t[9],s,rel),ch) :
                        ctx+rel<0 ? 0 : raw(c.t[2],cs,ch,c.hist+rel);
                }
                for (int i=0;i<c.hist;++i) store_raw(c.t[2],win[i],cs,ch,i);
            }
        }
    }
    return RAD_OK;
}
int rad_ref_ple_tree_count(void) { return RAD_PLE_TREE_OPS; }
const RadKernelInfo* rad_ref_ple_tree_at(int i) {
    static const RadKernelInfo rows[]={
        rad_ple_tree_row(RAD_PLE_TREE_IDS,RAD_DOMAIN_HOST,rad_ref_ngram_ids_tree),
        rad_ple_tree_row(RAD_PLE_TREE_CONV,RAD_DOMAIN_HOST,rad_ref_ple_conv_tree),
        rad_ple_tree_row(RAD_PLE_TREE_COMMIT,RAD_DOMAIN_HOST,rad_ref_ple_tree_commit)};
    return i>=0 && i<RAD_PLE_TREE_OPS ? rows+i : nullptr;
}
