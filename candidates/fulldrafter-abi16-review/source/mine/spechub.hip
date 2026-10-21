#ifdef __FAST_MATH__
#error "coupled sampler requires strict FP64 arithmetic"
#endif
#include <hip/hip_runtime.h>
#include "spechub.h"
#include "../libr4d/r4d_emit.h"
namespace {
__global__ void propose(RadSpechubProposeCall c) {
    int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row<c.M) rad_spechub_propose_row(c,row);
}
__global__ void verify(RadSpechubVerifyCall c) {
    int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row<c.M) c.token[row*c.ld[4]]=rad_spechub_verify_row(c,row);
}
__global__ void couple_propose(RadMtpCoupleProposeCall c) {
    int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row<c.M) rad_mtp_couple_propose_row(c,row);
}
constexpr int CoupleMaxCand=1024, CoupleThreads=128;
// Separate pointer arguments let graph replay rebind each operand.
__global__ void couple_verify(const int32_t* idx,const float* values,const uint32_t* params,
                              const uint32_t* state,int32_t* token,int64_t M,int64_t n_cand,
                              int64_t vocab,int64_t ld0,int64_t ld1,int64_t ld2,
                              int64_t ld3,int64_t ld4) {
    RadMtpCoupleVerifyCall c{idx,values,params,state,token,M,n_cand,vocab,{ld0,ld1,ld2,ld3,ld4}};
    const int64_t row=blockIdx.x;
    if (row>=M) return;
    // The public geometry admits more than 1024 candidates; use its general algorithm.
    if (n_cand>CoupleMaxCand) {
        if (threadIdx.x==0) rad_mtp_couple_verify_row(c,row);
        return;
    }
    __shared__ double weight[CoupleMaxCand],mass[CoupleMaxCand],pq[8],sum;
    __shared__ float top;
    __shared__ int eligible,first[8],ix,dominated;
    const auto& s=*reinterpret_cast<const RadMtpCoupleState*>(c.state+row*c.ld[3]);
    const auto& p=*reinterpret_cast<const RadSampleParams*>(c.params+row*c.ld[2]);
    const int32_t* ids=c.idx+row*c.ld[0];
    const float* val=c.val+row*c.ld[1];
    if (threadIdx.x==0) {
        eligible=0;
        if (s.status==1 && !(p.flags&RAD_SP_GREEDY) && s.draw.seed==p.seed &&
            s.pos==p.pos && rad_mtp_couple_state_valid(s,c.vocab)) {
            float maximum=-INFINITY;
            bool valid=true;
            for (int64_t j=0;j<c.n_cand;++j) {
                if (ids[j]<0) continue;
                if (ids[j]>=c.vocab || val[j]!=val[j] || val[j]==INFINITY) {
                    valid=false;
                    break;
                }
                if (val[j]>maximum) maximum=val[j];
            }
            if (valid && maximum!=-INFINITY) {top=maximum;eligible=1;}
        }
    }
    __syncthreads();
    if (!eligible) return;
    for (int64_t j=threadIdx.x;j<c.n_cand;j+=blockDim.x)
        if (ids[j]>=0 && val[j]!=-INFINITY)
            weight[j]=::exp(static_cast<double>(val[j])-top);
    __syncthreads();
    if (threadIdx.x==0) {
        sum=0;
        for (int k=0;k<s.count;++k) {pq[k]=0;first[k]=-1;}
        for (int64_t j=0;j<c.n_cand;++j) {
            if (ids[j]<0 || val[j]==-INFINITY) continue;
            const double w=weight[j];
            sum+=w;
            for (int k=0;k<s.count;++k)
                if (ids[j]==s.ids[k]) {
                    if (first[k]<0) first[k]=static_cast<int>(j);
                    pq[k]+=w;
                    break;
                }
        }
        if (!(sum>0) || !rad_spechub_finite(sum)) eligible=0;
        if (eligible) {
            ix=-1;
            dominated=true;
            for (int k=0;k<s.count;++k) {
                pq[k]/=sum;
                dominated=dominated && pq[k]>=s.q[k];
                if (s.ids[k]==s.x) ix=k;
            }
            if (ix<0) eligible=0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    for (int64_t j=threadIdx.x;j<c.n_cand;j+=blockDim.x) {
        double m=0;
        if (ids[j]>=0 && val[j]!=-INFINITY) {
            bool in_q=false;
            for (int k=0;k<s.count;++k) if (s.ids[k]==ids[j]) {
                in_q=true;
                if (first[k]==j) m=rad_spechub_positive(pq[k]-s.q[k]);
                break;
            }
            if (!in_q) m=weight[j]/sum;
        }
        mass[j]=m;
    }
    __syncthreads();
    if (threadIdx.x!=0) return;
    double resid=0;
    for (int64_t j=0;j<c.n_cand;++j) resid+=mass[j];
    int32_t out=-1;
    const double u_resid=rad_spechub_uniform(p.seed,p.pos,RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid>0 && rad_spechub_finite(resid))) {
        double cdf=u_resid*sum;
        for (int64_t j=0;j<c.n_cand;++j) if (ids[j]>=0 && val[j]!=-INFINITY) {
            const double w=weight[j];
            if (w>0) {out=ids[j];cdf-=w;if (cdf<0) break;}
        }
    } else if (pq[ix]>=s.q[ix] ||
               rad_spechub_uniform(p.seed,p.pos,RAD_SAMPLE_KEY_ACCEPT)*s.q[ix]<pq[ix]) {
        out=s.x;
    } else {
        double cdf=u_resid*resid;
        for (int64_t j=0;j<c.n_cand;++j) {
            if (ids[j]==s.x) continue;
            const double m=mass[j];
            if (m>0) {out=ids[j];cdf-=m;if (cdf<0) break;}
        }
    }
    c.token[row*c.ld[4]]=out;
}
static_assert(R4dParams<decltype(couple_verify)>::count==13 &&
              R4dParams<decltype(couple_verify)>::bytes()==104);
template <class Call>
int launch(const void* kernel,Call& c,RadStream stream) {
    void* args[]={&c};
    return hipLaunchKernel(kernel,dim3(static_cast<unsigned>((c.M+63)/64)),dim3(64),args,0,
        reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}
static int couple_verify_body(const RadArgs* a,const R4dSink& sink) {
    RadMtpCoupleVerifyCall c{};
    const int rc=rad_mtp_couple_verify_args(a,&c);
    if (rc!=RAD_OK) return rc;
    return R4D_EMIT(sink,couple_verify,dim3(static_cast<unsigned>(c.M)),dim3(CoupleThreads),0,
                    c.idx,c.val,c.params,c.state,c.token,c.M,c.n_cand,c.vocab,
                    c.ld[0],c.ld[1],c.ld[2],c.ld[3],c.ld[4]);
}
static int couple_verify_describe(const RadArgs* a,RadStream stream,RadLaunchDesc* desc,int cap,int* n) {
    if (!n || !desc) return RAD_E_INVAL;
    *n=0;
    R4dSink sink;sink.stream=stream;sink.desc=desc;sink.cap=cap;sink.n=n;
    return couple_verify_body(a,sink);
}
int rad_mine_spechub_propose_launch(const RadArgs* a,RadStream stream) {
    RadSpechubProposeCall c{};int rc=rad_spechub_propose_args(a,&c);if (rc!=RAD_OK) return rc;
    return launch(reinterpret_cast<const void*>(propose),c,stream);
}
int rad_mine_spechub_verify_launch(const RadArgs* a,RadStream stream) {
    RadSpechubVerifyCall c{};int rc=rad_spechub_verify_args(a,&c);if (rc!=RAD_OK) return rc;
    return launch(reinterpret_cast<const void*>(verify),c,stream);
}
int rad_mine_mtp_couple_propose_launch(const RadArgs* a,RadStream stream) {
    RadMtpCoupleProposeCall c{};int rc=rad_mtp_couple_propose_args(a,&c);if (rc!=RAD_OK) return rc;
    return launch(reinterpret_cast<const void*>(couple_propose),c,stream);
}
int rad_mine_mtp_couple_verify_launch(const RadArgs* a,RadStream stream) {
    return couple_verify_body(a,r4d_stream_sink(stream));
}
int rad_mine_spechub_count() {return 4;}
const RadKernelInfo* rad_mine_spechub_at(int i) {
    static const RadKernelInfo rows[]={
        rad_spechub_row(0,rad_mine_spechub_propose_launch),
        rad_spechub_row(1,rad_mine_spechub_verify_launch),
        rad_mtp_couple_row(0,rad_mine_mtp_couple_propose_launch),
        [] {
            RadKernelInfo row=rad_mtp_couple_row(1,rad_mine_mtp_couple_verify_launch);
            row.describe=couple_verify_describe;
            return row;
        }()
    };
    return i>=0 && i<4 ? &rows[i] : nullptr;
}
