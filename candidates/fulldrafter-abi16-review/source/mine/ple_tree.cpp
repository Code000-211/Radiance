#include "ple_tree.h"
#include <cstring>
#include <initializer_list>

namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
#define PO(k) {k,RAD_P_INT,RAD_OPTIONAL,0}
static const RadParamSpec params[] = {
    PI("q_len"),PI("n"),PI("heads"),PI("ngram"),PI("eos"),PI("width"),PI("dilation"),
    PI("node_capacity"),PI("history_ring"),PO("n_seq"),
    {"mode",RAD_P_STR,RAD_OPTIONAL,0},PO("tok_group"),PO("conv_group"),PO("layer"),PO("raw_buffer")
};
#undef PI
#undef PO
#define IN(n) {n,RAD_OPD_IN,0}
#define OPT(n) {n,RAD_OPD_IN,1}
#define OUT(n) {n,RAD_OPD_OUT,0}
#define IO(n) {n,RAD_OPD_INOUT,0}
#define W(n) {n,RAD_OPD_WEIGHT,0}
static const RadOperandSpec ids[] = {
    IN("tokens"),IN("tok_history"),IN("tok_idx"),IN("cu"),OPT("parent"),IN("ctx_before"),
    W("mult"),W("vocab_sizes"),W("offsets"),OUT("ids")
};
static const RadOperandSpec conv[] = {
    IN("x"),W("weights"),IO("conv_history"),IN("conv_idx"),IN("cu"),OPT("parent"),
    IN("ctx_before"),OPT("residual"),OUT("y"),{"saved_raw",RAD_OPD_OUT,1}
};
static const RadOperandSpec commit[] = {
    IO("tok_history"),IN("tok_idx"),IO("conv_history"),IN("conv_idx"),IN("tokens"),
    IN("saved_raw"),IN("cu"),OPT("parent"),IN("ctx_before"),IN("path_nodes"),IN("path_len")
};
#undef IN
#undef OPT
#undef OUT
#undef IO
#undef W
static const RadOpSchema schemas[] = {
    {"ngram_ids_tree",params,15,ids,10,"Parent-aware EOS-cut ngram hashes over immutable canonical token history."},
    {"ple_conv_tree",params,15,conv,10,"Dilated BF16 PLE convolution: immutable tree verification or linear committing prefill."},
    {"ple_tree_commit",params,15,commit,11,"Commit only the accepted root-inclusive token/raw-input ancestry path."}
};
static const RadConstraint constraints[] = {
    RAD_CGE("q_len",1),RAD_CGE("n",1),RAD_CGE("heads",1),RAD_CGE("ngram",2),RAD_CLE("ngram",8),
    RAD_CGE("eos",0),RAD_CGE("width",2),RAD_CLE("width",8),RAD_CGE("dilation",1),
    RAD_CGE("node_capacity",16),RAD_CLE("node_capacity",65),RAD_CGE("history_ring",0),RAD_CLE("history_ring",1)
};

bool geometry(const RadParam* p,int np,int op,RadPleTreeCall& c) {
    for (int i=0;i<9;++i) {
        const auto* v=rad_param_find(p,np,params[i].key);
        if (!v || (v->kind!=RAD_P_INT && !(i==0 && v->kind==RAD_P_RANGE))) return false;
        if (v->kind==RAD_P_RANGE && (v->ival<1 || v->ival>v->ihi)) return false;
    }
    for (int i=9;i<15;++i) {
        const auto* v=rad_param_find(p,np,params[i].key);
        if (v && v->kind!=params[i].type && !(i==9 && v->kind==RAD_P_RANGE)) return false;
        if (v && ((v->kind==RAD_P_STR && !v->sval) ||
            (v->kind==RAD_P_RANGE && (v->ival<1 || v->ival>v->ihi)))) return false;
    }
    auto get=[&](const char* key) { return rad_param_getdim(p,np,key,0); };
    const int64_t n=get("n"),heads=get("heads"),ng=get("ngram"),eos=get("eos");
    const int64_t width=get("width"),dil=get("dilation"),q=get("q_len"),ring=get("history_ring");
    const int64_t capacity=get("node_capacity");
    const char* mode=rad_param_gets(p,np,"mode","verify");
    if (std::strcmp(mode,"verify") && std::strcmp(mode,"prefill")) return false;
    c.prefill=!std::strcmp(mode,"prefill");
    if ((c.prefill && op!=RAD_PLE_TREE_CONV) || n<1 || n>INT32_MAX || heads<1 || heads>INT32_MAX ||
        ng<2 || ng>8 || heads%(ng-1) || eos<0 || eos>INT32_MAX || width<2 || width>8 ||
        dil<1 || dil>RAD_PLE_TREE_HIST_MAX/(width-1) ||
        (capacity!=RAD_PLE_TREE_CAP && capacity!=RAD_PLE_TREE_WIDE_CAP) ||
        q<1 || q>INT32_MAX || (!c.prefill && q>capacity) || ring<0 || ring>1) return false;
    c.op=op;c.n=(int)n;c.heads=(int)heads;c.ngram=(int)ng;c.eos=(int)eos;
    c.width=(int)width;c.dilation=(int)dil;c.capacity=(int)capacity;c.q_len=(int)q;c.ring=(int)ring;
    c.hist=(int)((width-1)*dil);
    return true;
}

/* Positive non-overlapping strides permit pitched projections and state slots,
 * while bounding every element and byte offset before either backend launches. */
bool tensor(const RadTensor& t,uint32_t dt,std::initializer_list<int64_t> shape,
            RadPleTreeTensor& v) {
    if (!t.data || t.dtype!=dt || t.rank!=shape.size() || t.rank>3) return false;
    int i=0;
    for (int64_t d:shape) if (t.shape[i++]!=d) return false;
    int64_t span=1;
    for (int j=(int)t.rank-1;j>=0;--j) {
        if (t.shape[j]<1 || t.stride[j]<span || t.shape[j]>INT64_MAX/t.stride[j]) return false;
        span=t.shape[j]*t.stride[j];
    }
    const unsigned bytes=dt==RAD_BF16 ? 2u : dt==RAD_I64 ? 8u : 4u;
    if ((uint64_t)span>SIZE_MAX/bytes || (uintptr_t)t.data>UINTPTR_MAX-(uint64_t)span*bytes) return false;
    v={};v.data=t.data;
    for (unsigned j=0;j<t.rank;++j) v.s[j]=t.stride[j];
    return true;
}
bool index(const RadTensor& t,int n,RadPleTreeTensor& v) {
    return t.rank==1 ? tensor(t,RAD_I32,{n},v) :
        t.rank==2 && t.shape[1]>0 && tensor(t,RAD_I32,{n,t.shape[1]},v);
}
bool token_history(const RadTensor& t,RadPleTreeCall& c,int arg) {
    if ((t.rank!=2 && t.rank!=3) || t.shape[0]<1 || t.shape[0]>INT32_MAX) return false;
    const int64_t len=t.shape[t.rank-1];
    if (len<c.ngram-1 || len>INT32_MAX) return false;
    c.tok_slots=(int)t.shape[0];c.tok_len=(int)len;
    if (t.rank==2) return tensor(t,RAD_I32,{c.tok_slots,len},c.t[arg]);
    if (!tensor(t,RAD_I32,{c.tok_slots,1,len},c.t[arg])) return false;
    c.t[arg].s[1]=c.t[arg].s[2];c.t[arg].s[2]=0;
    return true;
}
bool conv_history(const RadTensor& t,RadPleTreeCall& c) {
    if (t.rank!=3 || t.shape[0]<1 || t.shape[0]>INT32_MAX ||
        t.shape[2]<c.hist || t.shape[2]>INT32_MAX) return false;
    c.conv_slots=(int)t.shape[0];c.conv_len=(int)t.shape[2];
    return tensor(t,RAD_BF16,{c.conv_slots,c.n,c.conv_len},c.t[2]);
}
uint64_t bytes(const RadTensor& t) {
    if (!t.data) return 0;
    uint64_t count=1;
    for (unsigned j=0;j<t.rank;++j) count+=(uint64_t)(t.shape[j]-1)*t.stride[j];
    return count*(t.dtype==RAD_BF16 ? 2u : t.dtype==RAD_I64 ? 8u : 4u);
}
bool overlaps(const RadTensor& a,const RadTensor& b) {
    if (!a.data || !b.data) return false;
    const uintptr_t x=(uintptr_t)a.data,y=(uintptr_t)b.data;
    return x<y+bytes(b) && y<x+bytes(a);
}

int shape(int op,const RadParam* p,int np,int operand,RadOpdDesc* out) {
    if (!out || operand<0 || operand>=schemas[op].n_operands) return RAD_E_INVAL;
    RadPleTreeCall c{};
    if (!geometry(p,np,op,c)) return RAD_E_UNSUPPORTED;
    const int64_t s=rad_param_getdim(p,np,"n_seq",1);
    if (s<1 || s>INT32_MAX/c.q_len) return RAD_E_SHAPE;
    const int64_t t=s*c.q_len;
    RadOpdDesc desc[11]{};int at=0;
    auto add=[&](uint32_t dt,std::initializer_list<int64_t> dims) -> RadOpdDesc& {
        auto& v=desc[at++];v.dtype=dt;v.idx_const=-1;
        for (int64_t d:dims) v.shape[v.rank++]=d;
        return v;
    };
    auto idx=[&](int64_t count,int64_t max,int64_t fixed,uint32_t flags=0) -> RadOpdDesc& {
        auto& v=add(RAD_I32,{count});v.fill=RAD_FILL_INDEX;v.idx_max=max;v.idx_const=fixed;v.flags=flags;
        return v;
    };
    auto tokhist=[&]() { add(RAD_I32,{s,1,c.ngram-1}); };
    auto convhist=[&]() { add(RAD_BF16,{s,c.n,c.hist}); };
    auto slot=[&]() { idx(s,s,-1,RAD_OPD_F_IDX_UNIQUE); };
    auto cu=[&]() { idx(s+1,t,-1,RAD_OPD_F_IDX_CU); };
    auto parent=[&]() { add(RAD_I32,{t}).flags=RAD_OPD_F_ABSENT; };
    auto ctx=[&]() { idx(s,0,c.hist); };
    if (op==RAD_PLE_TREE_IDS) {
        idx(t,1024,-1);tokhist();slot();cu();parent();ctx();
        auto& m=add(RAD_I64,{c.ngram});m.fill=RAD_FILL_INDEX;m.idx_const=17;
        auto& v=add(RAD_I64,{c.heads});v.fill=RAD_FILL_INDEX;v.idx_const=997;
        auto& o=add(RAD_I64,{c.heads});o.fill=RAD_FILL_INDEX;o.idx_const=0;
        add(RAD_I32,{t,c.heads});
    } else if (op==RAD_PLE_TREE_CONV) {
        add(RAD_BF16,{t,c.n});add(RAD_BF16,{c.n,c.width});convhist();slot();cu();parent();ctx();
        add(RAD_BF16,{t,c.n});add(RAD_BF16,{t,c.n});add(RAD_BF16,{t,c.n});
    } else {
        tokhist();slot();convhist();slot();idx(t,1024,-1);add(RAD_BF16,{t,c.n});cu();parent();ctx();
        auto& path=add(RAD_I32,{s,c.capacity});path.fill=RAD_FILL_INDEX;path.idx_const=0;
        idx(s,0,1);
    }
    *out=desc[operand];return RAD_OK;
}
} // namespace

int rad_ple_tree_args(const RadArgs* a,int op,RadPleTreeCall* out) {
    if (!a || !a->t || !out || op<0 || op>=RAD_PLE_TREE_OPS || a->n_t!=schemas[op].n_operands)
        return RAD_E_INVAL;
    RadPleTreeCall c{};
    if (!geometry(a->p,a->n_p,op,c)) return RAD_E_UNSUPPORTED;
    const int ci=rad_ple_tree_cu(c),pi=rad_ple_tree_parent_index(c),cx=rad_ple_tree_ctx_index(c);
    const auto& cu=a->t[ci];
    const auto& x=a->t[op==RAD_PLE_TREE_COMMIT ? 4 : 0];
    if (cu.rank!=1 || cu.shape[0]<2 || cu.shape[0]>INT32_MAX ||
        x.rank!=(op==RAD_PLE_TREE_CONV ? 2u : 1u) || x.shape[0]<1 || x.shape[0]>INT32_MAX)
        return RAD_E_SHAPE;
    c.nseq=(int)cu.shape[0]-1;c.tokens=(int)x.shape[0];
    if (!tensor(cu,RAD_I32,{c.nseq+1},c.t[ci]) ||
        !tensor(a->t[cx],RAD_I32,{c.nseq},c.t[cx]) ||
        (a->t[pi].data && !tensor(a->t[pi],RAD_I32,{c.tokens},c.t[pi]))) return RAD_E_SHAPE;
    if (c.prefill && a->t[pi].data) return RAD_E_INVAL;
    auto row=[&](int i) { return tensor(a->t[i],RAD_BF16,{c.tokens,c.n},c.t[i]); };
    auto ix=[&](int i) { return index(a->t[i],c.nseq,c.t[i]); };
    bool ok;
    if (op==RAD_PLE_TREE_IDS) {
        ok=tensor(x,RAD_I32,{c.tokens},c.t[0]) && token_history(a->t[1],c,1) && ix(2) &&
            tensor(a->t[6],RAD_I64,{c.ngram},c.t[6]) && tensor(a->t[7],RAD_I64,{c.heads},c.t[7]) &&
            tensor(a->t[8],RAD_I64,{c.heads},c.t[8]) && tensor(a->t[9],RAD_I32,{c.tokens,c.heads},c.t[9]);
    } else if (op==RAD_PLE_TREE_CONV) {
        ok=row(0) && tensor(a->t[1],RAD_BF16,{c.n,c.width},c.t[1]) && conv_history(a->t[2],c) &&
            ix(3) && (!a->t[7].data || row(7)) && row(8) &&
            (a->t[9].data ? row(9) : c.prefill!=0);
    } else {
        const int64_t pitch=a->t[9].rank==2 ? a->t[9].shape[1] : 0;
        ok=token_history(a->t[0],c,0) && ix(1) && conv_history(a->t[2],c) && ix(3) &&
            tensor(x,RAD_I32,{c.tokens},c.t[4]) && row(5) &&
            (pitch==RAD_PLE_TREE_CAP || pitch==RAD_PLE_TREE_WIDE_CAP) &&
            tensor(a->t[9],RAD_I32,{c.nseq,pitch},c.t[9]) && tensor(a->t[10],RAD_I32,{c.nseq},c.t[10]);
        /* Runtime path storage is independent of the declared verification band. */
        if (pitch<c.capacity) c.capacity=(int)pitch;
    }
    if (!ok) return RAD_E_SHAPE;
    /* No output may destroy another operand's immutable input or saved bits.
     * Check intervals, not pointer equality: narrowed/pitched views can overlap. */
    for (int i=0;i<a->n_t;++i) {
        const bool write=op==RAD_PLE_TREE_IDS ? i==9 : op==RAD_PLE_TREE_CONV ?
            (i==8 || i==9 || (i==2 && c.prefill)) : (i==0 || i==2);
        if (write) for (int j=0;j<a->n_t;++j)
            if (j!=i && overlaps(a->t[i],a->t[j])) return RAD_E_INVAL;
    }
    *out=c;return RAD_OK;
}
int rad_ple_tree_schema_count(void) { return RAD_PLE_TREE_OPS; }
const RadOpSchema* rad_ple_tree_schema_at(int i) {
    return i>=0 && i<RAD_PLE_TREE_OPS ? schemas+i : nullptr;
}
int rad_ple_tree_ids_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(0,p,n,i,out); }
int rad_ple_tree_conv_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(1,p,n,i,out); }
int rad_ple_tree_commit_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(2,p,n,i,out); }
RadKernelInfo rad_ple_tree_row(int op,int domain,RadLaunchFn launch) {
    static const char* host[]={"ref_ngram_ids_tree","ref_ple_conv_tree","ref_ple_tree_commit"};
    static const char* device[]={"mine_ngram_ids_tree","mine_ple_conv_tree","mine_ple_tree_commit"};
    static const RadShapeFn shapes[]={rad_ple_tree_ids_shape,rad_ple_tree_conv_shape,rad_ple_tree_commit_shape};
    RadKernelInfo r{};
    if (op<0 || op>=RAD_PLE_TREE_OPS) return r;
    r.name=domain==RAD_DOMAIN_HOST ? host[op] : device[op];r.op=schemas[op].op;r.family="ple";
    r.computes=schemas[op].doc;r.shape="ngram=2..8, width=2..8, hist<=64, nodes<=65 (capacity 16 or 65) or linear prefill";
    r.dtypes="i32 tokens/ids; i64 hash weights; bf16 activations/history; f32 accumulation";
    r.domain=domain;r.priority=domain==RAD_DOMAIN_DEVICE ? 100 : 0;
    r.constraints=constraints;r.n_constraints=sizeof(constraints)/sizeof(constraints[0]);
    r.launch=launch;r.opd_shape=shapes[op];return r;
}
