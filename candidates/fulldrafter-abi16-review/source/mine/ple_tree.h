#ifndef RAD_MINE_PLE_TREE_H
#define RAD_MINE_PLE_TREE_H

#include "rad_plugin.h"
#include <stdint.h>

/* Absolute token rows, sequence-local parents, and root-inclusive commit paths.
 * Token history uses the first ngram-1 cells, oldest to newest. Convolution
 * history uses the first hist cells in canonical mode, or position % conv_len
 * in ring mode. Physical history tails/padding are not canonical window cells.
 * Verification supports 16- or 65-node capacities without per-node scratch.
 * Commit accepts a physical path pitch of 16 or 65 independently of capacity.
 * Verification never writes committed history; convolution prefill commits all
 * its linear inputs after reading the immutable history. No launch allocates. */
enum { RAD_PLE_TREE_IDS = 0, RAD_PLE_TREE_CONV = 1, RAD_PLE_TREE_COMMIT = 2,
       RAD_PLE_TREE_OPS = 3, RAD_PLE_TREE_CAP = 16, RAD_PLE_TREE_WIDE_CAP = 65,
       RAD_PLE_TREE_HIST_MAX = 64 };

struct RadPleTreeTensor {
    void* data;
    int64_t s[3];
};
struct RadPleTreeCall {
    RadPleTreeTensor t[11];
    int op, nseq, tokens, n, heads, ngram, eos, width, dilation, capacity, q_len;
    int ring, prefill, hist, tok_len, conv_len, tok_slots, conv_slots;
};

/* Shape/stride/alias checks only: device metadata is never dereferenced here. */
int rad_ple_tree_args(const RadArgs*, int op, RadPleTreeCall*);
int rad_ple_tree_schema_count(void);
const RadOpSchema* rad_ple_tree_schema_at(int);
int rad_ple_tree_ids_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_ple_tree_conv_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_ple_tree_commit_shape(const RadParam*, int, int, RadOpdDesc*);
RadKernelInfo rad_ple_tree_row(int op, int domain, RadLaunchFn);
int rad_mine_ple_tree_count(void);
const RadKernelInfo* rad_mine_ple_tree_at(int);

#if defined(__HIPCC__)
#define RAD_PLE_HD __host__ __device__
#else
#define RAD_PLE_HD
#endif
RAD_PLE_HD inline int64_t rad_ple_off(const RadPleTreeTensor& t, int64_t a,
                                     int64_t b = 0, int64_t c = 0) {
    return a*t.s[0] + b*t.s[1] + c*t.s[2];
}
RAD_PLE_HD inline int rad_ple_int(const RadPleTreeTensor& t, int a, int b = 0) {
    return static_cast<const int32_t*>(t.data)[rad_ple_off(t,a,b)];
}
RAD_PLE_HD inline int rad_ple_parent(const RadPleTreeTensor& t, int bos, int node) {
    return t.data ? rad_ple_int(t,bos+node) : node-1;
}
RAD_PLE_HD inline int rad_ple_tree_cu(const RadPleTreeCall& c) {
    return c.op == RAD_PLE_TREE_IDS ? 3 : c.op == RAD_PLE_TREE_CONV ? 4 : 6;
}
RAD_PLE_HD inline int rad_ple_tree_parent_index(const RadPleTreeCall& c) {
    return rad_ple_tree_cu(c)+1;
}
RAD_PLE_HD inline int rad_ple_tree_ctx_index(const RadPleTreeCall& c) {
    return rad_ple_tree_cu(c)+2;
}
/* A nonnegative result is a local candidate node. -k denotes the k-th
 * committed predecessor, k>=1. Linear prefill remains constant-time per tap. */
RAD_PLE_HD inline int rad_ple_tree_source(const RadPleTreeCall& c, int bos,
                                        int node, int lag) {
    const auto& p=c.t[rad_ple_tree_parent_index(c)];
    if (!p.data) return node-lag;
    while (lag && node>=0) { node=rad_ple_parent(p,bos,node); --lag; }
    return node>=0 ? node : -lag-1;
}
/* Uniform bounded device guards. Runtime producers must validate topology to
 * report errors; the host reference validates every sequence before any write. */
RAD_PLE_HD inline bool rad_ple_tree_metadata(const RadPleTreeCall& c, int seq) {
    const int ci=rad_ple_tree_cu(c), pi=rad_ple_tree_parent_index(c);
    const int bos=rad_ple_int(c.t[ci],seq), end=rad_ple_int(c.t[ci],seq+1);
    if (bos<0 || end<=bos || end>c.tokens || end-bos>c.q_len) return false;
    const int ctx=rad_ple_int(c.t[rad_ple_tree_ctx_index(c)],seq);
    if (ctx<0 || ctx>INT32_MAX-(end-bos)) return false;
    if (c.t[pi].data) for (int i=0;i<end-bos;++i) {
        const int p=rad_ple_parent(c.t[pi],bos,i);
        if ((i==0 && p!=-1) || (i>0 && (p<0 || p>=i))) return false;
    }
    if (c.op!=RAD_PLE_TREE_CONV) {
        const int slot=rad_ple_int(c.t[c.op==RAD_PLE_TREE_IDS ? 2 : 1],seq);
        if (slot<0 || slot>=c.tok_slots) return false;
    }
    if (c.op!=RAD_PLE_TREE_IDS) {
        const int slot=rad_ple_int(c.t[3],seq);
        if (slot<0 || slot>=c.conv_slots) return false;
    }
    if (c.op==RAD_PLE_TREE_COMMIT) {
        const int len=rad_ple_int(c.t[10],seq);
        if (len<1 || len>end-bos || len>c.capacity) return false;
        int prev=-1;
        for (int j=0;j<len;++j) {
            const int node=rad_ple_int(c.t[9],seq,j);
            if (node<0 || node>=end-bos || rad_ple_parent(c.t[pi],bos,node)!=prev) return false;
            prev=node;
        }
    }
    if (c.op==RAD_PLE_TREE_COMMIT || c.prefill) {
        for (int s=0;s<c.nseq;++s) if (s!=seq &&
            (rad_ple_int(c.t[3],s)==rad_ple_int(c.t[3],seq) ||
             (c.op==RAD_PLE_TREE_COMMIT && rad_ple_int(c.t[1],s)==rad_ple_int(c.t[1],seq))))
            return false;
    }
    return true;
}
#undef RAD_PLE_HD
#endif
