/* Included inside tree_gdn.hip's private namespace; shares its FP32 recurrence helpers. */
struct PackedTreeGdnI8g {
    void* data[20];
    int32_t stride[20][2], extra[4];
    int nseq,tokens,hk,hv,capacity,q_len,ring,hist_len,hist_slots,state_slots,sigmoid;
    float scale,eps,l2_eps,softplus_thr;
};
static_assert(sizeof(PackedTreeGdnI8g)==400,"compact fused kernarg layout");

__device__ __forceinline__ RadGdnTreeFusedCall unpack_fused(const PackedTreeGdnI8g& p) {
    RadGdnTreeFusedCall f{};
    for (int i=0;i<20;++i) {
        f.t[i].data=p.data[i];f.t[i].s[0]=p.stride[i][0];f.t[i].s[1]=p.stride[i][1];
    }
    f.t[3].s[2]=p.extra[0];f.t[12].s[2]=p.extra[1];f.t[12].s[3]=p.extra[2];f.t[17].s[2]=p.extra[3];
    auto& c=f.geometry;
    c.nseq=p.nseq;c.tokens=p.tokens;c.hk=p.hk;c.hv=p.hv;c.width=4;c.capacity=p.capacity;
    c.q_len=p.q_len;c.ring=p.ring;c.hist_len=p.hist_len;c.hist_slots=p.hist_slots;c.state_slots=p.state_slots;
    c.sigmoid=p.sigmoid;c.scale=p.scale;c.eps=p.eps;c.l2_eps=p.l2_eps;c.softplus_thr=p.softplus_thr;
    return f;
}

__device__ __forceinline__ uint16_t fused_conv(const RadGdnTreeFusedCall& f,int bos,int ch,
                                           int slot,int ctx,const signed char* path,int depth) {
    const auto& c=f.geometry;
    float acc=f.t[2].data ? bf2f(static_cast<const uint16_t*>(f.t[2].data)[rad_gdn_off(f.t[2],ch)]) : 0;
#pragma unroll
    for (int tap=0;tap<4;++tap) {
        const int lag=3-tap,pos=ctx+depth-1-lag;
        float x=0;
        if (lag<depth) x=bf2f(static_cast<const uint16_t*>(f.t[0].data)[rad_gdn_off(f.t[0],bos+path[lag],ch)]);
        else if (pos>=0) x=bf2f(static_cast<const uint16_t*>(f.t[3].data)[
            rad_gdn_off(f.t[3],slot,ch,c.ring ? pos%c.hist_len : 3+depth-1-lag)]);
        const float w=bf2f(static_cast<const uint16_t*>(f.t[1].data)[rad_gdn_off(f.t[1],ch,tap)]);
        acc=fmaf(w,x,acc);
    }
    /* Preserve gdn_conv_tree's BF16 store; its consumers perform the BF16 load. */
    return f2bf(cp_silu(acc));
}

/* Parallel=true assigns each target node an independent block. Its FP32 path state
 * lives only in registers: no global per-node matrices or cross-block dependencies.
 * This replaces Bole's reordered affine solve, not its rounding with a tolerance.
 * The block keeps only its target's ancestor path (up to RAD_GDN_TREE_WIDE_CAP
 * nodes) and stages K/V in RAD_GDN_TREE_CAP-step tiles indexed by tile-local
 * ordinal, so LDS does not grow with capacity. A path of at most RAD_GDN_TREE_CAP
 * nodes is one tile with the original single staging barrier.
 * Parallel=false is the serial row: its launch refuses capacity above
 * RAD_GDN_TREE_CAP, so its one tile's ordinal is the node ID, as before.
 * ponytail: ancestors are recomputed per descendant; measure the extra work against
 * the additional node-level parallelism before enabling this row in production. */
template <bool Parallel>
__global__ __launch_bounds__(512) void fused_i8g_kernel(PackedTreeGdnI8g packed) {
    const RadGdnTreeFusedCall f=unpack_fused(packed);
    const auto& c=f.geometry;
    const int seq=blockIdx.x,head=blockIdx.y,tid=threadIdx.x,row=tid>>2,qtr=tid&3;
    /* Serial: path[n] for every node n. Parallel: path[0] is the target's path only. */
    constexpr int Nodes=Parallel ? RAD_GDN_TREE_WIDE_CAP : RAD_GDN_TREE_CAP;
    constexpr int Paths=Parallel ? 1 : RAD_GDN_TREE_CAP;
    __shared__ char ok;
    __shared__ int par[Nodes];
    __shared__ signed char path[Paths][Nodes],depth[Paths];
    __shared__ __align__(16) float skey[Parallel ? 1 : 16][128],su[16][128],sg[16];
    __shared__ uint16_t svalue[16][128];
    __shared__ float wave_sum[16],quant[128];
    if (!tid) {
        ok=1;
        for (int s=0;s<c.nseq;++s) if (!rad_gdn_tree_fused_metadata(f,s)) { ok=0;break; }
    }
    lds_barrier();
    if (!ok) return; // No output, raw-history copy, or factor write before the full verdict.
    const int bos=rad_gdn_int(f.t[5],seq),nnode=rad_gdn_int(f.t[5],seq+1)-bos;
    const int target=Parallel ? int(blockIdx.z) : 0;
    if (Parallel && target>=nnode) return;
    const int hslot=rad_gdn_int(f.t[4],seq),ctx=rad_gdn_int(f.t[7],seq),slot=rad_gdn_int(f.t[13],seq);
    const int kh=head/(c.hv/c.hk),d=128*(2*c.hk+c.hv);
    if (tid<nnode) par[tid]=rad_gdn_parent(f.t[6],bos,tid);
    lds_barrier();
    if (Parallel ? !tid : tid<nnode) {
        const int at=Parallel ? 0 : tid;
        int n=0;
        for (int p=Parallel ? target : tid;p>=0;p=par[p]) path[at][n++]=(signed char)p;
        depth[at]=(signed char)n;
    }
    lds_barrier();
    /* Each raw channel has exactly one writer, even when Q/K convolution is duplicated. */
    for (int node=Parallel ? target : 0;node<(Parallel ? target+1 : nnode);++node)
        for (int ch=head*512+tid;ch<d;ch+=c.hv*512)
            static_cast<uint16_t*>(f.t[16].data)[rad_gdn_off(f.t[16],bos+node,ch)]=
                static_cast<const uint16_t*>(f.t[0].data)[rad_gdn_off(f.t[0],bos+node,ch)];
    const int steps=Parallel ? int(depth[0]) : nnode;
    const float alog=__expf(ld(f.t[10],head)),dtb=ld(f.t[11],head),nw=ld(f.t[15],row);
    float anchor[32],h[32];
    if (Parallel) ld32(h,static_cast<const float*>(f.t[12].data)+rad_gdn_off(f.t[12],slot,head,row,qtr*32),f.t[12].s[3]);
    else ld32(anchor,static_cast<const float*>(f.t[12].data)+rad_gdn_off(f.t[12],slot,head,row,qtr*32),f.t[12].s[3]);
    for (int step=0;step<steps;++step) {
        const int node=Parallel ? int(path[0][steps-1-step]) : step,local=step%RAD_GDN_TREE_CAP;
        if (!local) {
            /* Serial replay reuses Q/K bands for K/U. Independent paths need K/V for
             * ancestors but Q only for the target: keep just that one Q row in LDS.
             * A later tile first waits until every thread has left the previous one. */
            if (step) lds_barrier();
            const int count=steps-step<RAD_GDN_TREE_CAP ? steps-step : (int)RAD_GDN_TREE_CAP;
            for (int i=tid;i<count*384;i+=512) {
                const int st=step+i/384,channel=i%384,col=channel&127;
                const int n=Parallel ? int(path[0][steps-1-st]) : st;
                if (Parallel && n!=target && channel<128) continue;
                const int ch=channel<128 ? kh*128+col :
                             channel<256 ? (c.hk+kh)*128+col : (2*c.hk+head)*128+col;
                const uint16_t value=fused_conv(f,bos,ch,hslot,ctx,Parallel ? &path[0][steps-1-st] : path[n],
                                                Parallel ? st+1 : int(depth[n]));
                if (channel<128) skey[Parallel ? 0 : n][col]=bf2f(value);
                else if (channel<256) su[i/384][col]=bf2f(value);
                else svalue[i/384][col]=value;
            }
            lds_barrier();
        }
        const int tok=bos+node;
        if (!Parallel && !node) cp32(h,anchor);
        else if (!Parallel && par[node]!=node-1) {
            cp32(h,anchor);
            for (int i=depth[node]-1;i>=1;--i) {
                const int a=path[node][i];replay_lds(h,skey[a],su[a][row],sg[a],qtr);
            }
        }
        /* Q/K already hold the exact BF16 convolution store/load values. */
        const bool output=!Parallel || node==target;
        float q[32],k[32],sq=0,ks=0;
#pragma unroll
        for (int j=0;j<32;++j) {
            if (output) { q[j]=skey[Parallel ? 0 : node][qtr*32+j];sq=fmaf(q[j],q[j],sq); }
            k[j]=su[local][qtr*32+j];ks=fmaf(k[j],k[j],ks);
        }
        float qs=0;
        if (output) {
            sq+=r4d_xor_lane<1>(sq);sq+=r4d_xor_lane<2>(sq);
            qs=__frsqrt_rn(sq+c.l2_eps)*c.scale;
        }
        ks+=r4d_xor_lane<1>(ks);ks+=r4d_xor_lane<2>(ks);
        const float kn=__frsqrt_rn(ks+c.l2_eps);
#pragma unroll
        for (int j=0;j<32;++j) { if (output) q[j]*=qs;k[j]*=kn; }
        const float av=bf2f(static_cast<const uint16_t*>(f.t[8].data)[rad_gdn_off(f.t[8],tok,head)]);
        const float bv=bf2f(static_cast<const uint16_t*>(f.t[9].data)[rad_gdn_off(f.t[9],tok,head)]);
        const float decay=__expf(-alog*softplus(av+dtb,c.softplus_thr)),beta=1.0f/(1.0f+__expf(-bv));
        float dot=0;
#pragma unroll
        for (int j=0;j<32;++j) { h[j]*=decay;dot=fmaf(h[j],k[j],dot); }
        dot+=r4d_xor_lane<1>(dot);dot+=r4d_xor_lane<2>(dot);
        const float v=bf2f(svalue[local][row]);
        const float u=(v-dot)*beta;
        if (!Parallel) lds_barrier(); // Publish serial replay factors only after Q/K readers finish.
        if (!Parallel || node==target) {
            float* const factors=static_cast<float*>(f.t[17].data)+rad_gdn_off(f.t[17],tok,head);
            const int64_t fi=f.t[17].s[2];
            if (!row) {
#pragma unroll
                for (int j=0;j<32;++j) {
                    factors[(qtr*32+j)*fi]=k[j];
                    if (!Parallel) skey[node][qtr*32+j]=k[j];
                }
            }
            if (!qtr) { factors[(128+row)*fi]=u;if (!Parallel) su[node][row]=u; }
            if (!tid) { factors[256*fi]=decay;if (!Parallel) sg[node]=decay; }
        }
        if (Parallel && node!=target) {
#pragma unroll
            for (int j=0;j<32;++j) h[j]=fmaf(u,k[j],h[j]);
            continue; // No intermediate output/normalization is needed along this private path.
        }
        float on=0;
#pragma unroll
        for (int j=0;j<32;++j) { h[j]=fmaf(u,k[j],h[j]);on=fmaf(h[j],q[j],on); }
        on+=r4d_xor_lane<1>(on);on+=r4d_xor_lane<2>(on);
        float rms=!qtr ? on*on : 0;
        rms+=r4d_xor_lane<4>(rms);rms+=r4d_xor_lane<8>(rms);rms+=r4d_xor_lane<16>(rms);
        if (!(tid&31)) wave_sum[tid>>5]=rms;
        lds_barrier();
        float total=0;
#pragma unroll
        for (int i=0;i<16;++i) total+=wave_sum[i];
        const float sc=__frsqrt_rn(total*(1.0f/128)+c.eps);
        const float z=bf2f(static_cast<const uint16_t*>(f.t[14].data)[rad_gdn_off(f.t[14],tok,head*128+row)]);
        const float sig=1.0f/(1.0f+__expf(-z));
        on*=sc*nw*(c.sigmoid ? sig : z*sig);
        if (!qtr) quant[row]=bf2f(f2bf(on)); // Exact unfused BF16 output -> quantizer input boundary.
        lds_barrier();
        if (tid<32) {
            const int col=tid*4;
            const float a=quant[col],b=quant[col+1],v2=quant[col+2],v3=quant[col+3];
            float m=fmaxf(fmaxf(fabsf(a),fabsf(b)),fmaxf(fabsf(v2),fabsf(v3)));
            m=r4d_wave_max(m);
            const float scale=m>0 ? m*(1.0f/127.0f) : 1.0f;
            if (!tid) static_cast<float*>(f.t[19].data)[rad_gdn_off(f.t[19],tok,head)]=scale;
            const uint32_t codes=q8_pack4<true>(a,b,v2,v3,1.0f/scale);
            auto* out=static_cast<uint8_t*>(f.t[18].data);
#pragma unroll
            for (int j=0;j<4;++j) out[rad_gdn_off(f.t[18],tok,head*128+col+j)]=(uint8_t)(codes>>(8*j));
        }
        lds_barrier(); // quant[] and wave_sum[] are reused by the next node.
    }
}

template <bool Parallel>
int fused_i8g_launch(const RadArgs* a,RadStream stream) {
    RadGdnTreeFusedCall f{};const int rc=rad_gdn_tree_fused_args(a,&f);if (rc!=RAD_OK) return rc;
    /* Only the exact row streams wide paths; the serial row's LDS tables hold RAD_GDN_TREE_CAP. */
    if (!Parallel && f.geometry.capacity>RAD_GDN_TREE_CAP) return RAD_E_UNSUPPORTED;
    PackedTreeGdnI8g p{};
    for (int i=0;i<20;++i) {
        p.data[i]=f.t[i].data;p.stride[i][0]=(int32_t)f.t[i].s[0];p.stride[i][1]=(int32_t)f.t[i].s[1];
    }
    p.extra[0]=(int32_t)f.t[3].s[2];p.extra[1]=(int32_t)f.t[12].s[2];
    p.extra[2]=(int32_t)f.t[12].s[3];p.extra[3]=(int32_t)f.t[17].s[2];
    const auto& c=f.geometry;
    p.nseq=c.nseq;p.tokens=c.tokens;p.hk=c.hk;p.hv=c.hv;p.capacity=c.capacity;p.q_len=c.q_len;
    p.ring=c.ring;p.hist_len=c.hist_len;p.hist_slots=c.hist_slots;p.state_slots=c.state_slots;
    p.sigmoid=c.sigmoid;p.scale=c.scale;p.eps=c.eps;p.l2_eps=c.l2_eps;p.softplus_thr=c.softplus_thr;
    void* args[]={&p};
    return hipLaunchKernel((const void*)fused_i8g_kernel<Parallel>,
                           dim3(c.nseq,c.hv,Parallel ? c.q_len : 1),dim3(512),args,0,
                           (hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
