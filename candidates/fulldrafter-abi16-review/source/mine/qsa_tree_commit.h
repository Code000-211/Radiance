#ifndef RAD_MINE_QSA_TREE_COMMIT_H
#define RAD_MINE_QSA_TREE_COMMIT_H

#include "rad_plugin.h"
#include <stdint.h>

/* k,pos,paths,lengths,starts,tail(INOUT),sidx,bt,weight,block_keys(INOUT).
 * paths are sequence-relative nodes, not a contiguous prefix of the projection.
 * Logical path width (16 or 65) bounds counts and node IDs independently of
 * the physical row stride; node IDs must also fit their sequence's token span.
 * Every accepted token completes its block BEFORE replacing its tail row.
 * Metadata rejection is batch-wide and precedes every state write. Device
 * metadata refusal leaves all state untouched; it cannot synchronously return
 * a content error. Independent live sequences must own distinct state/pages.
 */
struct RadQsaTreeCommitCall {
    const uint16_t* k;
    const int32_t *pos, *paths, *lengths, *starts;
    uint16_t* tail;
    const int32_t *sidx, *bt;
    const float* weight;
    uint16_t* out;
    int64_t k_ld, pos_stride, path_ld, path_stride, length_stride, start_stride;
    int64_t tail_slot, tail_row, sidx_ld, bt_ld, out_ld;
    int tokens, seqs, n, ratio, ring, rot, slots, pages, blocks, capacity;
    float eps, wadd, theta, scale;
};

#if defined(__HIPCC__)
#define RAD_QSA_HD __host__ __device__
#else
#define RAD_QSA_HD
#endif

RAD_QSA_HD inline int rad_qsa_tree_length(const RadQsaTreeCommitCall& c, int s) {
    return c.lengths[int64_t(s)*c.length_stride];
}
RAD_QSA_HD inline int rad_qsa_tree_token(const RadQsaTreeCommitCall& c, int s, int j) {
    return c.starts[int64_t(s)*c.start_stride] + c.paths[int64_t(s)*c.path_ld+int64_t(j)*c.path_stride];
}
RAD_QSA_HD inline int rad_qsa_tree_position(const RadQsaTreeCommitCall& c, int s, int j) {
    return c.pos[int64_t(rad_qsa_tree_token(c,s,j))*c.pos_stride];
}
RAD_QSA_HD inline int rad_qsa_tree_page(const RadQsaTreeCommitCall& c, int s, int p) {
    return c.bt[int64_t(s)*c.bt_ld+p/c.ratio];
}
RAD_QSA_HD inline bool rad_qsa_tree_metadata(const RadQsaTreeCommitCall& c) {
    for (int s=0; s<c.seqs; ++s) {
        const int count=rad_qsa_tree_length(c,s), start=c.starts[int64_t(s)*c.start_stride];
        const int end=s+1<c.seqs ? c.starts[int64_t(s+1)*c.start_stride] : c.tokens;
        const int slot=c.sidx[int64_t(s)*c.sidx_ld];
        if (count<0 || count>c.capacity ||
            start<0 || start>c.tokens || end<start || end>c.tokens ||
            slot < -1 || slot>=c.slots) return false;
        int previous=-1;
        for (int j=0; j<count; ++j) {
            const int node=c.paths[int64_t(s)*c.path_ld+int64_t(j)*c.path_stride];
            if (node<0 || node>=c.capacity || node>=end-start) return false;
            const int p=rad_qsa_tree_position(c,s,j);
            if (p<0 || p==INT32_MAX || (j && p!=previous+1) || p/c.ratio>=c.blocks) return false;
            previous=p;
            if (slot>=0 && (p+1)%c.ratio==0) {
                const int page=rad_qsa_tree_page(c,s,p);
                if (page < -1 || page>=c.pages) return false;
            }
        }
    }
    /* All node and position reads above are bounded before this alias scan. */
    for (int s=0; s<c.seqs; ++s) {
        const int ns=rad_qsa_tree_length(c,s), ss=c.sidx[int64_t(s)*c.sidx_ld];
        if (!ns || ss<0) continue;
        for (int t=s+1; t<c.seqs; ++t) {
            const int nt=rad_qsa_tree_length(c,t), st=c.sidx[int64_t(t)*c.sidx_ld];
            if (!nt || st<0) continue;
            if (ss==st) return false;
            for (int j=0; j<ns; ++j) {
                const int p=rad_qsa_tree_position(c,s,j);
                if ((p+1)%c.ratio) continue;
                const int page=rad_qsa_tree_page(c,s,p);
                if (page<0) continue;
                for (int k=0; k<nt; ++k) {
                    const int q=rad_qsa_tree_position(c,t,k);
                    if ((q+1)%c.ratio==0 && page==rad_qsa_tree_page(c,t,q)) return false;
                }
            }
        }
    }
    return true;
}
#undef RAD_QSA_HD

int rad_qsa_tree_commit_schema_count(void);
const RadOpSchema* rad_qsa_tree_commit_schema_at(int);
int rad_qsa_tree_commit_args(const RadArgs*, RadQsaTreeCommitCall*);
RadKernelInfo rad_qsa_tree_commit_row(RadLaunchFn);
int rad_mine_qsa_tree_commit_launch(const RadArgs*, RadStream);
int rad_ref_qsa_tree_commit(const RadArgs*, RadStream);
int64_t rad_ref_qsa_tree_commit_scratch(const RadArgs*);
int rad_mine_qsa_tree_commit_count(void);
const RadKernelInfo* rad_mine_qsa_tree_commit_at(int);
int rad_ref_qsa_tree_commit_count(void);
const RadKernelInfo* rad_ref_qsa_tree_commit_at(int);

#endif
