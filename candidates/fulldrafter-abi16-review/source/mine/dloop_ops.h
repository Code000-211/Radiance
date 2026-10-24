#ifndef RAD_MINE_DLOOP_OPS_H
#define RAD_MINE_DLOOP_OPS_H
#include "rad_plugin.h"
#include <stdint.h>

/* Dense row-major operands; all weights/teacher rows BF16 (RNE), all computed
 * activations/accumulators F32. Dimensions are config fields below. S=n_seq,
 * Q=max_seqs, T=tap_rows, C=context_window, H=hidden_size, F=context_features_dim,
 * W=target_width, M=proposal_length, K=pool_size, R=selector_rank, V=vocab_size,
 * Vs=vocab_size_shard, G=world_size. All ops share the required parameter list.
 *
 * ctx_append: taps[T,F], path I32[S,16] (absolute tap rows, -1 terminated),
 * slots I32[S] (unique), reset I32[S] (0/1), ring inout[Q,C,F],
 * head/length inout I32[Q]. head is NEXT insertion offset, length is valid count.
 * Reset clears head/length, not unused ring bytes. Append follows path order.
 *
 * base: ring[Q,C,F], head/length I32[Q], slots I32[S], anchor_embed[S,W],
 * weights[weight_elements], head_shard[Vs,W] -> ctxh F32[S,C,H] (left padded),
 * y_base F32[S,M,W], local_ids I32[S,M,K], local_logits F32[S,M,K].
 * Missing local candidates = -1/-infinity. Only IDs < active_vocab compete.
 *
 * pool: gathered_ids I32[G,S,M,K], gathered_logits F32[G,S,M,K], anchor I32[S],
 * head_shard[Vs,W], embed_shard[Vs,W], pred_shard/succ_shard[Vs,R] ->
 * pool_ids I32[S,M,K], unary F32[S,M,K], pool_head/pool_embed[S,M,K,W],
 * pool_pred/pool_succ[S,M,K,R], anchor_pred[S,R]. BF16 outputs contain zero
 * outside this shard; caller all-reduce SUMs the five BF16 row outputs.
 *
 * dartree: ctxh F32[S,C,H], length I32[Q], slots I32[S], anchor_embed[S,W],
 * weights[weight_elements], y_base F32[S,M,W], pool_ids I32[S,M,K],
 * unary F32[S,M,K], pool_head/pool_embed[S,M,K,W],
 * pool_pred/pool_succ[S,M,K,R], anchor_pred[S,R] -> tree_tok/tree_parent
 * I32[S,16], n_nodes I32[S]. Root is IMPLICIT: first draft parent=-1.
 * Outputs unused columns=-1. n_nodes excludes anchor and is <=budget<=15.
 * Optional trailing anchor I32[S], packed_tree I32[S,33] must both be present
 * or absent. Packed output INCLUDES root: count=n_nodes+1, token[0]=anchor,
 * parent[0]=-1, draft parents=split_parent+1. This is drafter ABI cap16.
 * ctx_append has optional trailing tap1/tap2 BF16[T,W]. When both present,
 * taps is BF16[T,W], F=3W, and append concatenates [taps,tap1,tap2].
 *
 * Pool selection: descending score, lower global ID at ties; final pool sorted
 * by token ID. This matches tree/dloop.py::_candidate_pool, which repairs
 * torch.topk's unspecified cutoff tie order. Tree ties use parent-major pool
 * flatten order (stable argsort); prune ties use depth then original order.
 * Scratch is required for base/dartree; hook includes one arena per sequence.
 */
enum { RAD_DLOOP_CTX_APPEND, RAD_DLOOP_BASE, RAD_DLOOP_POOL,
       RAD_DLOOP_DARTREE, RAD_DLOOP_OPS, RAD_DLOOP_CAP=16 };
struct RadDLoopConfig {
    int n_seq,max_seqs,tap_rows,world_size;
    int hidden_size,intermediate_size,num_layers,num_q_heads,num_kv_heads,head_dim;
    int context_features_dim,target_width,proposal_length,context_window;
    int active_vocab,vocab_size,selector_rank,conv_kernel_size,conv_group_size;
    int vocab_lo,vocab_size_shard,pool_size,budget,width,pruned;
    float eps,rope_theta,depth_bonus;
};
/* Blob order, dense with NO padding, in BF16 ELEMENTS (not bytes): context_proj,
 * context_norm, token_proj, mask_token; each layer in the enum order; final_norm,
 * output_proj, selector.hidden_proj. Codebooks are separate shard operands.
 * Names match sidecar names without dloop. prefix. */
enum RadDLoopWeight {
    DL_CONTEXT_PROJ,DL_CONTEXT_NORM,DL_TOKEN_PROJ,DL_MASK_TOKEN,
    DL_INPUT_NORM,DL_ATTN_BASE,DL_ATTN_KERNEL,DL_Q_PROJ,DL_K_PROJ,DL_V_PROJ,DL_O_PROJ,
    DL_POST_NORM,DL_MLP_BASE,DL_MLP_KERNEL,DL_GATE_PROJ,DL_UP_PROJ,DL_DOWN_PROJ,
    DL_FINAL_NORM,DL_OUTPUT_PROJ,DL_SELECTOR_PROJ
};
inline int64_t rad_dloop_weight_size(const RadDLoopConfig& c,int w) {
    const int64_t H=c.hidden_size,D=c.num_kv_heads*c.head_dim,I=c.intermediate_size;
    const int64_t kernel=c.conv_kernel_size,groups=H/c.conv_group_size;
    switch(w) {
    case DL_CONTEXT_PROJ:return H*c.context_features_dim;
    case DL_CONTEXT_NORM:case DL_MASK_TOKEN:case DL_INPUT_NORM:case DL_POST_NORM:case DL_FINAL_NORM:return H;
    case DL_TOKEN_PROJ:case DL_OUTPUT_PROJ:return H*c.target_width;
    case DL_ATTN_BASE:case DL_MLP_BASE:return 2*kernel*H;
    case DL_ATTN_KERNEL:case DL_MLP_KERNEL:return 2*kernel*groups*H;
    case DL_Q_PROJ:case DL_O_PROJ:return H*H;
    case DL_K_PROJ:case DL_V_PROJ:return D*H;
    case DL_GATE_PROJ:case DL_UP_PROJ:case DL_DOWN_PROJ:return H*I;
    case DL_SELECTOR_PROJ:return (int64_t)c.selector_rank*c.target_width;
    default:return 0;
    }
}
inline int64_t rad_dloop_weight_offset(const RadDLoopConfig& c,int w,int layer=0) {
    int64_t off=0;
    for(int i=DL_CONTEXT_PROJ;i<=DL_MASK_TOKEN;++i){if(i==w)return off;off+=rad_dloop_weight_size(c,i);}
    for(int l=0;l<c.num_layers;++l)for(int i=DL_INPUT_NORM;i<=DL_DOWN_PROJ;++i){if(i==w&&l==layer)return off;off+=rad_dloop_weight_size(c,i);}
    for(int i=DL_FINAL_NORM;i<=DL_SELECTOR_PROJ;++i){if(i==w)return off;off+=rad_dloop_weight_size(c,i);}
    return off;
}
inline int64_t rad_dloop_weight_elements(const RadDLoopConfig& c){return rad_dloop_weight_offset(c,-1);}
inline const char* rad_dloop_weight_name(int w) {
    static const char* names[]={"context_proj.weight","context_norm.weight","token_proj.weight","mask_token",
        "input_norm.weight","attention_conv.base_kernel","attention_conv.kernel_projection.weight","attention.q_proj.weight",
        "attention.k_proj.weight","attention.v_proj.weight","attention.o_proj.weight","post_attention_norm.weight",
        "mlp_conv.base_kernel","mlp_conv.kernel_projection.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight",
        "final_norm.weight","output_proj.weight","selector.hidden_proj.weight"};
    return w>=0&&w<20?names[w]:nullptr;
}
int rad_dloop_config(const RadParam*,int,RadDLoopConfig*);
int rad_dloop_args(const RadArgs*,int,RadDLoopConfig*);
int rad_dloop_schema_count(void);
const RadOpSchema* rad_dloop_schema_at(int);
int rad_dloop_ctx_append_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_dloop_base_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_dloop_pool_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_dloop_dartree_shape(const RadParam*,int,int,RadOpdDesc*);
int64_t rad_dloop_scratch_elements(const RadDLoopConfig&);
int64_t rad_dloop_scratch(const RadArgs*);
RadKernelInfo rad_dloop_row(int,int,RadLaunchFn);
#endif
