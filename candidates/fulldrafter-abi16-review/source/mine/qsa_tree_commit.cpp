#include "qsa_tree_commit.h"

#include <climits>
#include <cmath>
#include <cstring>

namespace {
#define PI(k) {k, RAD_P_INT, RAD_REQUIRED, 0}
#define PF(k) {k, RAD_P_F64, RAD_OPTIONAL, 0}
const RadParamSpec params[] = {PI("M"), PI("n"), PI("ratio"), PI("ring"), PI("rotary_dim"),
    PI("block_size"), {"max_tokens", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY},
    PF("eps"), PF("wadd"), PF("theta"), PF("scale")};
#undef PI
#undef PF
const RadOperandSpec operands[] = {
    {"k", RAD_OPD_IN, 0}, {"pos", RAD_OPD_IN, 0}, {"paths", RAD_OPD_IN, 0},
    {"lengths", RAD_OPD_IN, 0}, {"starts", RAD_OPD_IN, 0}, {"tail", RAD_OPD_INOUT, 0},
    {"sidx", RAD_OPD_IN, 0}, {"bt", RAD_OPD_IN, 0}, {"weight", RAD_OPD_WEIGHT, 0},
    {"block_keys", RAD_OPD_INOUT, 0}};
const RadOpSchema schema = {"qsa_tree_commit", params, 11, operands, 10,
    "Replay each noncontiguous accepted path in order: complete block from old tail and current "
    "raw key, BF16 pool/RMSNorm/NeoX RoPE, then tail store. One owner per sequence; invalid "
    "metadata or duplicate live state/page destinations leave the whole batch unchanged."};

int tensor(const RadTensor& t, uint32_t dtype, int rank, uint64_t& bytes) {
    if (!t.data) return RAD_E_INVAL;
    if (t.dtype!=dtype) return RAD_E_DTYPE;
    if (t.rank!=uint32_t(rank)) return RAD_E_SHAPE;
    const uint64_t element=dtype==RAD_BF16 ? 2 : 4;
    if (reinterpret_cast<uintptr_t>(t.data)%element) return RAD_E_INVAL;
    uint64_t span=1;
    for (int d=rank-1; d>=0; --d) {
        if (t.shape[d]<1 || t.shape[d]>INT32_MAX) return RAD_E_SHAPE;
        if (t.stride[d]<1 || (d<rank-1 && uint64_t(t.stride[d])<span)) return RAD_E_STRIDE;
        const uint64_t extent=uint64_t(t.shape[d]-1);
        if (extent && uint64_t(t.stride[d])>(INT64_MAX/element-span)/extent) return RAD_E_SHAPE;
        span+=extent*uint64_t(t.stride[d]);
    }
    bytes=span*element;
    if (reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes) return RAD_E_INVAL;
    return RAD_OK;
}
bool overlap(const RadTensor& a, uint64_t an, const RadTensor& b, uint64_t bn) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a.data), y=reinterpret_cast<uintptr_t>(b.data);
    return x<y+bn && y<x+an;
}
}

int rad_qsa_tree_commit_schema_count(void) { return 1; }
const RadOpSchema* rad_qsa_tree_commit_schema_at(int i) { return i==0 ? &schema : nullptr; }

int rad_qsa_tree_commit_args(const RadArgs* a, RadQsaTreeCommitCall* out) {
    if (!a || !out || !a->t || a->n_t!=10 || !a->p) return RAD_E_INVAL;
    const int64_t n=rad_param_geti(a->p,a->n_p,"n",0);
    const int64_t ratio=rad_param_geti(a->p,a->n_p,"ratio",0);
    const int64_t ring=rad_param_geti(a->p,a->n_p,"ring",0);
    const int64_t rot=rad_param_geti(a->p,a->n_p,"rotary_dim",0);
    const int64_t max_seq=rad_param_geti(a->p,a->n_p,"M",0);
    const int64_t max_tokens=rad_param_geti(a->p,a->n_p,"max_tokens",0);
    if (n<1 || n>256 || ratio<1 || ratio>32 || ring<ratio || ring>32 || rot<0 || rot>n ||
        (rot&1) || max_seq<1 || max_seq>256 || max_tokens<1 || max_tokens>INT32_MAX ||
        rad_param_geti(a->p,a->n_p,"block_size",0)!=ratio) return RAD_E_UNSUPPORTED;
    const auto& k=a->t[0]; const auto& pos=a->t[1]; const auto& path=a->t[2];
    const auto& length=a->t[3]; const auto& start=a->t[4]; const auto& tail=a->t[5];
    const auto& sidx=a->t[6]; const auto& bt=a->t[7]; const auto& weight=a->t[8];
    const auto& bk=a->t[9];
    const int ranks[]={2,1,2,1,1,4,2,2,1,int(bk.rank)};
    const uint32_t types[]={RAD_BF16,RAD_I32,RAD_I32,RAD_I32,RAD_I32,RAD_BF16,
                            RAD_I32,RAD_I32,RAD_F32,RAD_BF16};
    if (bk.rank!=2 && bk.rank!=4) return RAD_E_SHAPE;
    uint64_t bytes[10]{};
    for (int i=0; i<10; ++i) {
        const int rc=tensor(a->t[i],types[i],ranks[i],bytes[i]);
        if (rc!=RAD_OK) return rc;
    }
    const int64_t seqs=path.shape[0], tokens=k.shape[0];
    if (seqs>max_seq || tokens>max_tokens || k.shape[1]!=n || pos.shape[0]<tokens ||
        (path.shape[1]!=16 && path.shape[1]!=65) || length.shape[0]!=seqs || start.shape[0]!=seqs ||
        sidx.shape[0]!=seqs || bt.shape[0]!=seqs || tail.shape[1]!=1 ||
        tail.shape[2]!=ring || tail.shape[3]!=n || weight.shape[0]!=n) return RAD_E_SHAPE;
    if (k.stride[1]!=1 || tail.stride[3]!=1 || weight.stride[0]!=1 || bt.stride[1]!=1)
        return RAD_E_STRIDE;
    int64_t width=1;
    for (int d=int(bk.rank)-1; d>0; --d) {
        if (bk.stride[d]!=width) return RAD_E_STRIDE;
        if (bk.shape[d]>n/width) return RAD_E_SHAPE;
        width*=bk.shape[d];
    }
    if (width!=n) return RAD_E_SHAPE;
    for (int i=0; i<10; ++i) for (int j=i+1; j<10; ++j)
        if ((i==5 || i==9 || j==5 || j==9) && overlap(a->t[i],bytes[i],a->t[j],bytes[j]))
            return RAD_E_INVAL;
    const float eps=float(rad_param_getf(a->p,a->n_p,"eps",1e-6));
    const float wadd=float(rad_param_getf(a->p,a->n_p,"wadd",1.0));
    const float theta=float(rad_param_getf(a->p,a->n_p,"theta",10000.0));
    const float scale=float(rad_param_getf(a->p,a->n_p,"scale",1.0));
    if (!std::isfinite(eps) || eps<0 || !std::isfinite(wadd) || !std::isfinite(theta) ||
        theta<=0 || !std::isfinite(scale)) return RAD_E_INVAL;
    *out={static_cast<const uint16_t*>(k.data), static_cast<const int32_t*>(pos.data),
        static_cast<const int32_t*>(path.data), static_cast<const int32_t*>(length.data),
        static_cast<const int32_t*>(start.data), static_cast<uint16_t*>(tail.data),
        static_cast<const int32_t*>(sidx.data), static_cast<const int32_t*>(bt.data),
        static_cast<const float*>(weight.data), static_cast<uint16_t*>(bk.data),
        k.stride[0],pos.stride[0],path.stride[0],path.stride[1],length.stride[0],start.stride[0],
        tail.stride[0],tail.stride[2],sidx.stride[0],bt.stride[0],bk.stride[0],
        int(tokens),int(seqs),int(n),int(ratio),int(ring),int(rot),int(tail.shape[0]),
        int(bk.shape[0]),int(bt.shape[1]),int(path.shape[1]),
        eps,wadd,theta,scale};
    return RAD_OK;
}

RadKernelInfo rad_qsa_tree_commit_row(RadLaunchFn launch) {
    RadKernelInfo row{};
    row.name="mine_qsa_tree_commit"; row.op=schema.op; row.family="qsa";
    row.computes=schema.doc;
    row.shape="n<=256, ratio<=ring<=32, nodes<=65, path width 16 or 65, single-component NeoX";
    row.dtypes="bf16,f32,i32"; row.domain=RAD_DOMAIN_DEVICE; row.priority=100; row.launch=launch;
    return row;
}
