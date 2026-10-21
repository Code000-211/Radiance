#ifndef RAD_MINE_TREE_GDN_H
#define RAD_MINE_TREE_GDN_H

#include "rad_plugin.h"
#include <stdint.h>

/* Additive ops; positional operands (all strides are in elements):
 * CONV: x BF16[T,D], w BF16[D,W], bias? BF16[D], history BF16[slots,D,R],
 *       history_idx I32[S] (or [S,pitch], column 0), cu I32[S+1], parent? I32[T],
 *       ctx_before I32[S], q/k BF16[T,Hk*128], v BF16[T,Hv*128], raw BF16[T,D].
 * RECURRENT: q,k,v, a/b BF16[T,Hv], A_log/dt_bias F32[Hv],
 *       state F32[slots,Hv,128,128], state_idx I32[S] (or [S,pitch], column 0),
 *       cu, parent?, z BF16[T,Hv*128], norm_w F32[128], o BF16[T,Hv*128],
 *       factors F32[T,Hv,257]. Factors = [normalized k, parent-dependent u, exp(g)].
 * COMMIT: state,state_idx,factors,history,history_idx,raw,path_nodes I32[S,P>=capacity],
 *       path_len I32[S],ctx_before,cu,parent?. Paths include root, never bonus;
 *       columns past path_len are ignored.
 * q/k/v/z/o may also be rank3 [T,heads,128]. All token tensors use ABSOLUTE cu
 * offsets; only sequence metadata moves when issuing a mixed-step subrange.
 * Non-null parent is sequence-local, root=-1, parent[i]<i. Null is linear.
 * Verification is read-only on committed state/history; candidate data must
 * persist through sampling. Commit requires disjoint slots across sequences.
 * history_ring=0 keeps the W-1 oldest-to-newest taps at offsets [0,W-1);
 * history_ring=1 indexes logical positions modulo R. See tree_gdn_plan.txt.
 * Capacity: geometry accepts 1<=q_len<=node_capacity<=RAD_GDN_TREE_WIDE_CAP (64
 * proposals + root). Device rows advertising it: commit and the exact fused row.
 * Legacy serial device rows (conv, recurrent, serial fused) stay <=RAD_GDN_TREE_CAP
 * and refuse wider geometry at launch. Host references support the wide range.
 */
enum { RAD_GDN_TREE_CONV, RAD_GDN_TREE_RECURRENT, RAD_GDN_TREE_COMMIT,
       RAD_GDN_TREE_FUSED_I8G, RAD_GDN_TREE_OPS, RAD_GDN_TREE_CAP = 16,
       RAD_GDN_TREE_WIDE_CAP = 65, RAD_GDN_TREE_DIM = 128, RAD_GDN_TREE_FACTOR = 257 };

struct RadGdnTreeTensor {
    void* data;
    int64_t s[4];
};
struct RadGdnTreeCall {
    RadGdnTreeTensor t[15];
    int op, nseq, tokens, hk, hv, width, capacity, q_len, ring, hist_len, hist_slots, state_slots;
    float scale, eps, l2_eps, softplus_thr;
    int sigmoid;
};

/* Fused I8g operands: x,w,bias?,history,history_idx,cu,parent?,ctx_before,
 * a,b,A_log,dt_bias,state,state_idx,z,norm_w,raw,factors,o_q,o_s.
 * No Q/K/V or BF16 output operand: both BF16 rounding boundaries stay local.
 * Geometry is W=4, K=V=128, quantization group=128. Commit remains separate. */
struct RadGdnTreeFusedCall {
    RadGdnTreeTensor t[20];
    RadGdnTreeCall geometry;
};
int rad_gdn_tree_fused_args(const RadArgs*, RadGdnTreeFusedCall*);
int rad_gdn_tree_fused_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_ref_gdn_conv_recurrent_tree_i8g(const RadArgs*, RadStream);

int rad_gdn_tree_args(const RadArgs*, int op, RadGdnTreeCall*);
int rad_gdn_tree_schema_count(void);
const RadOpSchema* rad_gdn_tree_schema_at(int);
int rad_gdn_tree_conv_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_gdn_tree_recurrent_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_gdn_tree_commit_shape(const RadParam*, int, int, RadOpdDesc*);
/* wide: advertise RAD_GDN_TREE_WIDE_CAP instead of RAD_GDN_TREE_CAP nodes. */
RadKernelInfo rad_gdn_tree_row(int op, int domain, RadLaunchFn, bool wide);
int rad_mine_gdn_tree_count(void);
const RadKernelInfo* rad_mine_gdn_tree_at(int);

#if defined(__HIPCC__)
#define RAD_GDN_HD __host__ __device__
#else
#define RAD_GDN_HD
#endif

RAD_GDN_HD inline int64_t rad_gdn_off(const RadGdnTreeTensor& t, int64_t a,
                                      int64_t b = 0, int64_t c = 0, int64_t d = 0) {
    return a*t.s[0] + b*t.s[1] + c*t.s[2] + d*t.s[3];
}
RAD_GDN_HD inline int rad_gdn_int(const RadGdnTreeTensor& t, int a, int b = 0) {
    return static_cast<const int32_t*>(t.data)[rad_gdn_off(t,a,b)];
}
RAD_GDN_HD inline int rad_gdn_parent(const RadGdnTreeTensor& p, int bos, int node) {
    return p.data ? rad_gdn_int(p,bos+node) : node-1;
}
/* Device launchers cannot read device indices on the host. These uniform checks
 * bound every access, including malformed producer data, before any write.
 * Runtime must validate topology before submission to report malformed batches;
 * the host reference also returns an error for the same metadata. */
RAD_GDN_HD inline bool rad_gdn_tree_metadata(const RadGdnTreeCall& c, int seq) {
    const int ci = c.op == RAD_GDN_TREE_CONV ? 5 : 9;
    const int pi = c.op == RAD_GDN_TREE_CONV ? 6 : 10;
    const int bos = rad_gdn_int(c.t[ci],seq), end = rad_gdn_int(c.t[ci],seq+1);
    if (bos < 0 || end < bos || end > c.tokens || end-bos > c.q_len) return false;
    if (seq && bos < rad_gdn_int(c.t[ci],seq-1)) return false;
    for (int i=0; i<end-bos; ++i) {
        const int p=rad_gdn_parent(c.t[pi],bos,i);
        if ((i==0 && p!=-1) || (i>0 && (p<0 || p>=i))) return false;
    }
    if (c.op != RAD_GDN_TREE_RECURRENT) {
        const int idx = rad_gdn_int(c.t[4],seq);
        const int ctx = rad_gdn_int(c.t[c.op == RAD_GDN_TREE_CONV ? 7 : 8],seq);
        if (idx<0 || idx>=c.hist_slots || ctx<0 || ctx>INT32_MAX-c.capacity) return false;
    }
    if (c.op != RAD_GDN_TREE_CONV) {
        const int slot=rad_gdn_int(c.t[c.op == RAD_GDN_TREE_COMMIT ? 1 : 8],seq);
        if (slot<0 || slot>=c.state_slots) return false;
    }
    if (c.op == RAD_GDN_TREE_COMMIT) {
        const int len=rad_gdn_int(c.t[7],seq);
        if (len<1 || len>end-bos || len>c.capacity) return false;
        int prev=-1;
        for (int i=0;i<len;++i) {
            const int node=rad_gdn_int(c.t[6],seq,i);
            if (node<0 || node>=end-bos || rad_gdn_parent(c.t[10],bos,node)!=prev) return false;
            prev=node;
        }
        for (int s=0;s<c.nseq;++s) if (s!=seq &&
            (rad_gdn_int(c.t[1],s)==rad_gdn_int(c.t[1],seq) ||
             rad_gdn_int(c.t[4],s)==rad_gdn_int(c.t[4],seq))) return false;
    }
    return true;
}

RAD_GDN_HD inline bool rad_gdn_tree_fused_metadata(const RadGdnTreeFusedCall& f, int seq) {
    RadGdnTreeCall c=f.geometry;
    c.op=RAD_GDN_TREE_CONV;
    c.t[3]=f.t[3];c.t[4]=f.t[4];c.t[5]=f.t[5];c.t[6]=f.t[6];c.t[7]=f.t[7];
    if (!rad_gdn_tree_metadata(c,seq)) return false;
    c.op=RAD_GDN_TREE_RECURRENT;
    c.t[7]=f.t[12];c.t[8]=f.t[13];c.t[9]=f.t[5];c.t[10]=f.t[6];
    return rad_gdn_tree_metadata(c,seq);
}
#undef RAD_GDN_HD
#endif
