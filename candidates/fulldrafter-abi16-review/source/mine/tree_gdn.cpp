#include "tree_gdn.h"
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
#define PO(k) {k,RAD_P_INT,RAD_OPTIONAL,0}
#define PF(k) {k,RAD_P_F64,RAD_OPTIONAL,0}
static const RadParamSpec params[] = {
    PI("q_len"), PI("head_k"), PI("head_v"), PI("n_head_k"), PI("n_head_v"),
    PI("conv_width"), PI("node_capacity"), PI("history_ring"), PO("n_seq"),
    PF("scale"), PF("eps"), PF("l2_eps"), PF("softplus_thr"),
    {"act",RAD_P_STR,RAD_OPTIONAL,0},
    PO("state_group"), PO("conv_group"), PO("layer"), PO("raw_buffer"), PO("factor_buffer"),
    PI("max_tokens")
};
#undef PI
#undef PO
#undef PF
#define IN(n) {n,RAD_OPD_IN,0}
#define OPT(n) {n,RAD_OPD_IN,1}
#define OUT(n) {n,RAD_OPD_OUT,0}
#define IO(n) {n,RAD_OPD_INOUT,0}
#define W(n) {n,RAD_OPD_WEIGHT,0}
static const RadOperandSpec conv[] = {
    IN("x"),W("weights"),{"bias",RAD_OPD_WEIGHT,1},IN("history"),IN("history_idx"),
    IN("cu"),OPT("parent"),IN("ctx_before"),OUT("q"),OUT("k"),OUT("v"),OUT("saved_raw_x")
};
static const RadOperandSpec recurrent[] = {
    IN("q"),IN("k"),IN("v"),IN("a"),IN("b"),W("A_log"),W("dt_bias"),IN("state"),
    IN("state_idx"),IN("cu"),OPT("parent"),IN("z"),W("norm_w"),OUT("o"),OUT("factors")
};
static const RadOperandSpec commit[] = {
    IO("state"),IN("state_idx"),IN("factors"),IO("history"),IN("history_idx"),
    IN("saved_raw_x"),IN("path_nodes"),IN("path_len"),IN("ctx_before"),IN("cu"),OPT("parent")
};
static const RadOperandSpec fused[] = {
    IN("x"),W("weights"),{"bias",RAD_OPD_WEIGHT,1},IN("history"),IN("history_idx"),
    IN("cu"),OPT("parent"),IN("ctx_before"),IN("a"),IN("b"),W("A_log"),W("dt_bias"),
    IN("state"),IN("state_idx"),IN("z"),W("norm_w"),OUT("saved_raw_x"),OUT("factors"),
    OUT("o_q"),OUT("o_s")
};
#undef IN
#undef OPT
#undef OUT
#undef IO
#undef W
static const RadOpSchema schemas[] = {
    {"gdn_conv_tree",params,19,conv,12,"Parent-aware BF16 causal convolution; immutable committed history; saves raw x."},
    {"gdn_recurrent_tree",params,19,recurrent,15,"Parent-aware K128/V128 F32 recurrence and gated BF16 output; compact delta factors."},
    {"gdn_tree_commit",params,19,commit,11,"Root-inclusive winner replay and raw convolution history commit; no bonus."},
    {"gdn_conv_recurrent_tree_i8g",params,20,fused,20,
     "W4 parent-aware convolution, K128/V128 F32 recurrence, BF16 rounding and existing group128 INT8 quantization; raw x and F32 factors for separate commit."}
};
/* [0]: legacy serial device rows; [1]: wide rows (exact fused, commit, host references). */
#define TREE_LIMITS(cap) RAD_CGE("q_len",1),RAD_CLE("q_len",cap),\
    RAD_CGE("node_capacity",1),RAD_CLE("node_capacity",cap)
static const RadConstraint constraints[2][10] = {
    {RAD_CEQ("head_k",128),RAD_CEQ("head_v",128),TREE_LIMITS(RAD_GDN_TREE_CAP),
     RAD_CGE("conv_width",1),RAD_CLE("conv_width",17),RAD_CGE("history_ring",0),RAD_CLE("history_ring",1)},
    {RAD_CEQ("head_k",128),RAD_CEQ("head_v",128),TREE_LIMITS(RAD_GDN_TREE_WIDE_CAP),
     RAD_CGE("conv_width",1),RAD_CLE("conv_width",17),RAD_CGE("history_ring",0),RAD_CLE("history_ring",1)}
};
static const RadConstraint fused_constraints[2][9] = {
    {RAD_CEQ("head_k",128),RAD_CEQ("head_v",128),RAD_CEQ("conv_width",4),TREE_LIMITS(RAD_GDN_TREE_CAP),
     RAD_CGE("history_ring",0),RAD_CLE("history_ring",1)},
    {RAD_CEQ("head_k",128),RAD_CEQ("head_v",128),RAD_CEQ("conv_width",4),TREE_LIMITS(RAD_GDN_TREE_WIDE_CAP),
     RAD_CGE("history_ring",0),RAD_CLE("history_ring",1)}
};
#undef TREE_LIMITS

bool geometry(const RadParam* p,int np,RadGdnTreeCall& c) {
    auto get=[&](const char* key,int64_t def=0) { return rad_param_getdim(p,np,key,def); };
    const int64_t hk=get("n_head_k"),hv=get("n_head_v"),w=get("conv_width"),cap=get("node_capacity");
    const int64_t ring=get("history_ring",-1),q=get("q_len");
    if (get("head_k")!=128 || get("head_v")!=128 || hk<1 || hv<1 || hv%hk ||
        hk>INT32_MAX/384 || hv>INT32_MAX/384 || 128*(2*hk+hv)>INT32_MAX ||
        w<1 || w>17 || cap<1 || cap>RAD_GDN_TREE_WIDE_CAP || q<1 || q>cap || ring<0 || ring>1) return false;
    c.hk=(int)hk; c.hv=(int)hv; c.width=(int)w; c.capacity=(int)cap; c.ring=(int)ring;
    c.q_len=(int)q;
    c.scale=(float)rad_param_getf(p,np,"scale",1.0/std::sqrt(128.0));
    c.eps=(float)rad_param_getf(p,np,"eps",1e-6);
    c.l2_eps=(float)rad_param_getf(p,np,"l2_eps",1e-6);
    c.softplus_thr=(float)rad_param_getf(p,np,"softplus_thr",20.0);
    const char* act=rad_param_gets(p,np,"act","silu");
    if (std::strcmp(act,"silu") && std::strcmp(act,"sigmoid")) return false;
    c.sigmoid=!std::strcmp(act,"sigmoid");
    return std::isfinite(c.scale) && std::isfinite(c.eps) && c.eps>0 &&
        std::isfinite(c.l2_eps) && c.l2_eps>0 && std::isfinite(c.softplus_thr) && c.softplus_thr>0;
}

/* Reject overlapping/negative strides and overflowing offset arithmetic, while
 * preserving padded state slots and column slices of the fused projection. */
bool tensor(const RadTensor& t,uint32_t dt,std::initializer_list<int64_t> shape,
            RadGdnTreeTensor& v) {
    if (!t.data || t.dtype!=dt || t.rank!=shape.size() || t.rank>4) return false;
    int i=0;
    for (int64_t d:shape) if (t.shape[i++]!=d) return false;
    int64_t span=1;
    for (int j=(int)t.rank-1;j>=0;--j) {
        if (t.shape[j]<1 || t.stride[j]<span ||
            t.shape[j]>INT64_MAX/t.stride[j]) return false;
        span=t.shape[j]*t.stride[j];
    }
    if ((uint64_t)span>SIZE_MAX/(dt==RAD_BF16 ? 2u : 4u)) return false;
    v={}; v.data=t.data;
    for (unsigned j=0;j<t.rank;++j) v.s[j]=t.stride[j];
    return true;
}
bool index(const RadTensor& t,int64_t n,RadGdnTreeTensor& v) {
    if (t.rank==1) return tensor(t,RAD_I32,{n},v);
    return t.rank==2 && t.shape[1]>0 && tensor(t,RAD_I32,{n,t.shape[1]},v);
}
bool rows(const RadTensor& t,int64_t n,int64_t channels,RadGdnTreeTensor& v) {
    if (t.rank==2) return tensor(t,RAD_BF16,{n,channels},v);
    if (t.rank!=3 || channels%128 || !tensor(t,RAD_BF16,{n,channels/128,128},v) ||
        t.stride[1]!=128*t.stride[2]) return false;
    v.s[1]=t.stride[2]; v.s[2]=0;
    return true;
}

int shape(int op,const RadParam* p,int np,int operand,RadOpdDesc* out) {
    if (!out || op<0 || op>=3 || operand<0 || operand>=schemas[op].n_operands) return RAD_E_INVAL;
    RadGdnTreeCall c{};
    if (!geometry(p,np,c)) return RAD_E_UNSUPPORTED;
    const int64_t s=rad_param_getdim(p,np,"n_seq",1),q=rad_param_getdim(p,np,"q_len",1);
    if (s<1 || s>INT32_MAX/q) return RAD_E_SHAPE;
    const int64_t t=s*q,d=128*(2*c.hk+c.hv),r=c.width>1 ? c.width-1 : 1;
    RadOpdDesc desc[15]{};
    int at=0;
    auto add=[&](uint32_t dt,std::initializer_list<int64_t> dims) -> RadOpdDesc& {
        auto& v=desc[at++]; v.dtype=dt; v.idx_const=-1;
        for (auto n:dims) v.shape[v.rank++]=n;
        return v;
    };
    auto idx=[&](int64_t count,int64_t max,int64_t fixed,uint32_t flags=0) -> RadOpdDesc& {
        auto& v=add(RAD_I32,{count});v.fill=RAD_FILL_INDEX;v.idx_max=max;v.idx_const=fixed;v.flags=flags;
        return v;
    };
    auto cu=[&]() { idx(s+1,t,-1,RAD_OPD_F_IDX_CU); };
    /* The public fill ABI cannot express root=-1 followed by local parents.
     * Null parent has explicitly linear semantics, not random invalid indices. */
    auto parent=[&]() { add(RAD_I32,{t}).flags=RAD_OPD_F_ABSENT; };
    auto state=[&]() { add(RAD_F32,{s,c.hv,128,128}); };
    auto history=[&]() { add(RAD_BF16,{s,d,r}); };
    auto slot=[&]() { idx(s,s,-1,RAD_OPD_F_IDX_UNIQUE); };
    if (op==RAD_GDN_TREE_CONV) {
        add(RAD_BF16,{t,d});add(RAD_BF16,{d,c.width});add(RAD_BF16,{d});history();slot();cu();parent();
        idx(s,0,r);add(RAD_BF16,{t,c.hk*128});add(RAD_BF16,{t,c.hk*128});
        add(RAD_BF16,{t,c.hv*128});add(RAD_BF16,{t,d});
    } else if (op==RAD_GDN_TREE_RECURRENT) {
        add(RAD_BF16,{t,c.hk*128});add(RAD_BF16,{t,c.hk*128});add(RAD_BF16,{t,c.hv*128});
        add(RAD_BF16,{t,c.hv});add(RAD_BF16,{t,c.hv});add(RAD_F32,{c.hv});add(RAD_F32,{c.hv});
        state();slot();cu();parent();add(RAD_BF16,{t,c.hv*128});add(RAD_F32,{128});
        add(RAD_BF16,{t,c.hv*128});add(RAD_F32,{t,c.hv,257});
    } else {
        state();slot();add(RAD_F32,{t,c.hv,257}).fill=RAD_FILL_SIGMOID;
        history();slot();add(RAD_BF16,{t,d});
        auto& path=add(RAD_I32,{s,c.capacity>RAD_GDN_TREE_CAP ? c.capacity : (int)RAD_GDN_TREE_CAP});
        path.fill=RAD_FILL_INDEX;path.idx_const=0;
        idx(s,0,1);idx(s,0,r);cu();parent();
    }
    if (at!=schemas[op].n_operands) return RAD_E_SHAPE;
    *out=desc[operand];return RAD_OK;
}
} // namespace

int rad_gdn_tree_args(const RadArgs* a,int op,RadGdnTreeCall* out) {
    if (!a || !a->t || !out || op<0 || op>=3 || a->n_t!=schemas[op].n_operands) return RAD_E_INVAL;
    RadGdnTreeCall c{};c.op=op;
    if (!geometry(a->p,a->n_p,c)) return RAD_E_UNSUPPORTED;
    const int ci=op==RAD_GDN_TREE_CONV ? 5 : 9,pi=op==RAD_GDN_TREE_CONV ? 6 : 10;
    const RadTensor& cu=a->t[ci];
    if (cu.rank!=1 || cu.shape[0]<2 || cu.shape[0]>INT32_MAX) return RAD_E_SHAPE;
    c.nseq=(int)cu.shape[0]-1;
    const int token_arg=op==RAD_GDN_TREE_COMMIT ? 5 : 0;
    if (a->t[token_arg].rank<2 || a->t[token_arg].shape[0]<1 || a->t[token_arg].shape[0]>INT32_MAX)
        return RAD_E_SHAPE;
    c.tokens=(int)a->t[token_arg].shape[0];
    if (!tensor(cu,RAD_I32,{c.nseq+1},c.t[ci])) return RAD_E_SHAPE;
    if (a->t[pi].data && !tensor(a->t[pi],RAD_I32,{c.tokens},c.t[pi])) return RAD_E_SHAPE;
    const int d=128*(2*c.hk+c.hv);
    auto row=[&](int i,int channels) { return rows(a->t[i],c.tokens,channels,c.t[i]); };
    auto vec=[&](int i,uint32_t dt,int n) { return tensor(a->t[i],dt,{n},c.t[i]); };
    auto ix=[&](int i) { return index(a->t[i],c.nseq,c.t[i]); };
    auto history=[&]() {
        const auto& h=a->t[3];
        if (h.rank!=3 || h.shape[0]>INT32_MAX || h.shape[2]>INT32_MAX ||
            h.shape[2]<c.width-1 || h.shape[2]<1) return false;
        c.hist_slots=(int)h.shape[0];c.hist_len=(int)h.shape[2];
        return tensor(h,RAD_BF16,{c.hist_slots,d,c.hist_len},c.t[3]);
    };
    auto state=[&](int i) {
        const auto& st=a->t[i];
        if (st.rank!=4 || st.shape[0]>INT32_MAX) return false;
        c.state_slots=(int)st.shape[0];
        return tensor(st,RAD_F32,{c.state_slots,c.hv,128,128},c.t[i]);
    };
    auto factors=[&](int i) { return tensor(a->t[i],RAD_F32,{c.tokens,c.hv,257},c.t[i]); };
    bool ok;
    if (op==RAD_GDN_TREE_CONV) {
        ok=row(0,d) && tensor(a->t[1],RAD_BF16,{d,c.width},c.t[1]) &&
            (!a->t[2].data || vec(2,RAD_BF16,d)) && history() && ix(4) && ix(7) &&
            row(8,c.hk*128) && row(9,c.hk*128) && row(10,c.hv*128) && row(11,d);
    } else if (op==RAD_GDN_TREE_RECURRENT) {
        ok=row(0,c.hk*128) && row(1,c.hk*128) && row(2,c.hv*128) &&
            row(3,c.hv) && row(4,c.hv) && vec(5,RAD_F32,c.hv) && vec(6,RAD_F32,c.hv) &&
            state(7) && ix(8) && row(11,c.hv*128) && vec(12,RAD_F32,128) &&
            row(13,c.hv*128) && factors(14);
    } else {
        /* Any row pitch covering the logical capacity; trailing columns are ignored. */
        const auto& path=a->t[6];
        ok=state(0) && ix(1) && factors(2) && history() && ix(4) && row(5,d) &&
            path.rank==2 && path.shape[1]>=c.capacity &&
            tensor(path,RAD_I32,{c.nseq,path.shape[1]},c.t[6]) && ix(7) && ix(8);
    }
    if (!ok) return RAD_E_SHAPE;
    /* A caller must not turn immutable anchors into output aliases. */
    if (op!=RAD_GDN_TREE_COMMIT) {
        const int first=op==RAD_GDN_TREE_CONV ? 8 : 13;
        const int anchor=op==RAD_GDN_TREE_CONV ? 3 : 7;
        for (int i=first;i<a->n_t;++i) if (a->t[i].data==a->t[anchor].data) return RAD_E_INVAL;
    }
    *out=c;return RAD_OK;
}
int rad_gdn_tree_schema_count(void) { return RAD_GDN_TREE_OPS; }
const RadOpSchema* rad_gdn_tree_schema_at(int i) { return i>=0 && i<RAD_GDN_TREE_OPS ? schemas+i : nullptr; }
int rad_gdn_tree_conv_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(0,p,n,i,out); }
int rad_gdn_tree_recurrent_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(1,p,n,i,out); }
int rad_gdn_tree_commit_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(2,p,n,i,out); }
RadKernelInfo rad_gdn_tree_row(int op,int domain,RadLaunchFn launch,bool wide) {
    static const char* host[]={"ref_gdn_conv_tree","ref_gdn_recurrent_tree","ref_gdn_tree_commit",
                               "ref_gdn_conv_recurrent_tree_i8g"};
    static const char* device[]={"mine_gdn_conv_tree","mine_gdn_recurrent_tree","mine_gdn_tree_commit",
                                 "mine_gdn_conv_recurrent_tree_i8g"};
    static const RadShapeFn shapes[]={rad_gdn_tree_conv_shape,rad_gdn_tree_recurrent_shape,
                                      rad_gdn_tree_commit_shape,rad_gdn_tree_fused_shape};
    RadKernelInfo r{};
    if (op<0 || op>=RAD_GDN_TREE_OPS) return r;
    r.name=domain==RAD_DOMAIN_HOST ? host[op] : device[op];r.op=schemas[op].op;r.family="gdn";
    r.computes=schemas[op].doc;
    r.shape=wide ? "K=V=128, 1<=nodes<=65, 1<=width<=17" : "K=V=128, 1<=nodes<=16, 1<=width<=17";
    r.dtypes="bf16 activations/history; f32 state/factors/accumulators";
    r.domain=domain;r.priority=domain==RAD_DOMAIN_DEVICE ? 100 : 0;
    r.constraints=constraints[wide];r.n_constraints=sizeof(constraints[0])/sizeof(constraints[0][0]);
    if (op==RAD_GDN_TREE_FUSED_I8G) {
        r.constraints=fused_constraints[wide];
        r.n_constraints=sizeof(fused_constraints[0])/sizeof(fused_constraints[0][0]);
        r.shape=wide ? "K=V=group=128, width=4, 1<=nodes<=65" : "K=V=group=128, width=4, 1<=nodes<=16";
        r.dtypes="bf16 inputs/history/local rounding; f32 state/factors/scales; i8 codes";
    }
    r.launch=launch;r.opd_shape=shapes[op];return r;
}

int rad_gdn_tree_fused_args(const RadArgs* a,RadGdnTreeFusedCall* out) {
    if (!a || !a->t || !out || a->n_t!=20) return RAD_E_INVAL;
    RadGdnTreeFusedCall f{};auto& c=f.geometry;
    if (!geometry(a->p,a->n_p,c) || c.width!=4 ||
        rad_param_getdim(a->p,a->n_p,"group",128)!=128) return RAD_E_UNSUPPORTED;
    const auto& cu=a->t[5];
    if (cu.rank!=1 || cu.shape[0]<2 || cu.shape[0]>INT32_MAX ||
        a->t[0].rank<2 || a->t[0].shape[0]<1 || a->t[0].shape[0]>INT32_MAX)
        return RAD_E_SHAPE;
    c.nseq=(int)cu.shape[0]-1;c.tokens=(int)a->t[0].shape[0];
    const int64_t max_tokens=rad_param_getdim(a->p,a->n_p,"max_tokens",0);
    if (max_tokens<1 || max_tokens>INT32_MAX || c.tokens>max_tokens) return RAD_E_SHAPE;
    const auto& h=a->t[3];const auto& st=a->t[12];
    if (h.rank!=3 || h.shape[0]<1 || h.shape[0]>INT32_MAX || h.shape[2]<3 ||
        h.shape[2]>INT32_MAX || st.rank!=4 || st.shape[0]<1 || st.shape[0]>INT32_MAX)
        return RAD_E_SHAPE;
    c.hist_slots=(int)h.shape[0];c.hist_len=(int)h.shape[2];c.state_slots=(int)st.shape[0];
    const int d=128*(2*c.hk+c.hv);
    auto row=[&](int i,int n) { return rows(a->t[i],c.tokens,n,f.t[i]); };
    auto vec=[&](int i,uint32_t dt,int n) { return tensor(a->t[i],dt,{n},f.t[i]); };
    auto ix=[&](int i) { return index(a->t[i],c.nseq,f.t[i]); };
    if (!(row(0,d) && tensor(a->t[1],RAD_BF16,{d,4},f.t[1]) &&
        (!a->t[2].data || vec(2,RAD_BF16,d)) &&
        tensor(h,RAD_BF16,{c.hist_slots,d,c.hist_len},f.t[3]) && ix(4) &&
        tensor(cu,RAD_I32,{c.nseq+1},f.t[5]) &&
        (!a->t[6].data || vec(6,RAD_I32,c.tokens)) && ix(7) &&
        row(8,c.hv) && row(9,c.hv) && vec(10,RAD_F32,c.hv) && vec(11,RAD_F32,c.hv) &&
        tensor(st,RAD_F32,{c.state_slots,c.hv,128,128},f.t[12]) && ix(13) &&
        row(14,c.hv*128) && vec(15,RAD_F32,128) && row(16,d) &&
        tensor(a->t[17],RAD_F32,{c.tokens,c.hv,257},f.t[17]) &&
        tensor(a->t[18],RAD_I8,{c.tokens,c.hv*128},f.t[18]) &&
        tensor(a->t[19],RAD_F32,{c.tokens,c.hv},f.t[19]))) return RAD_E_SHAPE;
    /* Packing uses signed 32-bit strides, but offset products stay 64-bit. */
    for (int i=0;i<20;++i) for (int j=0;j<4;++j)
        if (f.t[i].s[j]>INT32_MAX) return RAD_E_STRIDE;
    /* Outputs may not overwrite any immutable input or one another. */
    for (int i=16;i<20;++i) for (int j=0;j<i;++j)
        if (a->t[i].data==a->t[j].data) return RAD_E_INVAL;
    *out=f;return RAD_OK;
}

int rad_gdn_tree_fused_shape(const RadParam* p,int np,int operand,RadOpdDesc* out) {
    if (!out || operand<0 || operand>=20) return RAD_E_INVAL;
    RadGdnTreeCall c{};
    if (!geometry(p,np,c) || c.width!=4 ||
        rad_param_getdim(p,np,"group",128)!=128) return RAD_E_UNSUPPORTED;
    const int64_t s=rad_param_getdim(p,np,"n_seq",1);
    if (s<1 || s>INT32_MAX/c.q_len) return RAD_E_SHAPE;
    const int64_t t=s*c.q_len,d=128*(2*c.hk+c.hv);
    const int64_t max_tokens=rad_param_getdim(p,np,"max_tokens",0);
    if (max_tokens<1 || max_tokens>INT32_MAX || t>max_tokens) return RAD_E_SHAPE;
    RadOpdDesc v[20]{};
    auto set=[&](int i,uint32_t dt,std::initializer_list<int64_t> dims) {
        v[i].dtype=dt;v[i].idx_const=-1;for (auto n:dims) v[i].shape[v[i].rank++]=n;
    };
    set(0,RAD_BF16,{t,d});set(1,RAD_BF16,{d,4});set(2,RAD_BF16,{d});
    set(3,RAD_BF16,{s,d,3});set(4,RAD_I32,{s});set(5,RAD_I32,{s+1});
    set(6,RAD_I32,{t});v[6].flags=RAD_OPD_F_ABSENT;
    set(7,RAD_I32,{s});set(8,RAD_BF16,{t,c.hv});set(9,RAD_BF16,{t,c.hv});
    set(10,RAD_F32,{c.hv});set(11,RAD_F32,{c.hv});
    set(12,RAD_F32,{s,c.hv,128,128});set(13,RAD_I32,{s});
    set(14,RAD_BF16,{t,c.hv*128});set(15,RAD_F32,{128});set(16,RAD_BF16,{t,d});
    set(17,RAD_F32,{t,c.hv,257});set(18,RAD_I8,{t,c.hv*128});set(19,RAD_F32,{t,c.hv});
    for (int i:{4,13}) {
        v[i].fill=RAD_FILL_INDEX;v[i].idx_max=s;v[i].flags=RAD_OPD_F_IDX_UNIQUE;
    }
    v[5].fill=RAD_FILL_INDEX;v[5].idx_max=t;v[5].flags=RAD_OPD_F_IDX_CU;
    v[7].fill=RAD_FILL_INDEX;v[7].idx_const=3;
    *out=v[operand];return RAD_OK;
}
