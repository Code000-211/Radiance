#include "dloop_ops.h"

extern "C" {
int rad_mine_dloop_ctx_append_c32_f7680_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_base_h768_m8_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_pool_h768_k64_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_dartree_h768_m8_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_ctx_append_c32_f48_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_base_h64_m8_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_pool_h64_k64_tp2(const RadArgs*,RadStream);
int rad_mine_dloop_dartree_h64_m8_tp2(const RadArgs*,RadStream);
int64_t rad_mine_dloop_base_scratch(const RadArgs*);
int64_t rad_mine_dloop_tree_scratch(const RadArgs*);
}
namespace {
#define COMMON RAD_CEQ("world_size",2),RAD_CEQ("num_layers",2), \
    RAD_CEQ("proposal_length",8),RAD_CEQ("context_window",32), \
    RAD_CEQ("conv_kernel_size",2),RAD_CEQ("conv_group_size",16), \
    RAD_CEQ("pool_size",64),RAD_CLE("budget",15),RAD_CLE("width",12),RAD_CIN("dtype","bf16")
const RadConstraint production[]={COMMON,
    RAD_CEQ("hidden_size",768),RAD_CEQ("intermediate_size",2304),
    RAD_CEQ("num_q_heads",12),RAD_CEQ("num_kv_heads",4),RAD_CEQ("head_dim",64),
    RAD_CEQ("context_features_dim",7680),RAD_CEQ("target_width",2560),
    RAD_CEQ("selector_rank",256),RAD_CEQ("vocab_size",248320),
    RAD_CEQ("active_vocab",248077),RAD_CEQ("vocab_size_shard",124160)};
const RadConstraint tiny[]={COMMON,
    RAD_CEQ("hidden_size",64),RAD_CEQ("intermediate_size",128),
    RAD_CEQ("num_q_heads",4),RAD_CEQ("num_kv_heads",2),RAD_CEQ("head_dim",16),
    RAD_CEQ("context_features_dim",48),RAD_CEQ("target_width",32),
    RAD_CEQ("selector_rank",8),RAD_CEQ("vocab_size",512),
    RAD_CEQ("active_vocab",509),RAD_CEQ("vocab_size_shard",256)};
#undef COMMON
struct Rows {
    RadKernelInfo row[8];
    Rows() {
        const RadLaunchFn launch[]={
            rad_mine_dloop_ctx_append_c32_f7680_tp2,rad_mine_dloop_base_h768_m8_tp2,
            rad_mine_dloop_pool_h768_k64_tp2,rad_mine_dloop_dartree_h768_m8_tp2,
            rad_mine_dloop_ctx_append_c32_f48_tp2,rad_mine_dloop_base_h64_m8_tp2,
            rad_mine_dloop_pool_h64_k64_tp2,rad_mine_dloop_dartree_h64_m8_tp2};
        const char* names[]={
            "mine_dloop_ctx_append_c32_f7680_tp2","mine_dloop_base_h768_m8_tp2",
            "mine_dloop_pool_h768_k64_tp2","mine_dloop_dartree_h768_m8_tp2",
            "mine_dloop_ctx_append_c32_f48_tp2","mine_dloop_base_h64_m8_tp2",
            "mine_dloop_pool_h64_k64_tp2","mine_dloop_dartree_h64_m8_tp2"};
        for(int i=0;i<8;++i) {
            row[i]=rad_dloop_row(i%4,RAD_DOMAIN_DEVICE,launch[i]);
            row[i].name=names[i];row[i].constraints=i<4?production:tiny;
            row[i].n_constraints=int(sizeof(production)/sizeof(production[0]));
            row[i].shape=i<4?"H768 FF2304 L2 Q12 KV4 Dh64 F7680 W2560 M8 C32 R256 K64 TP2":
                "fixture H64 FF128 L2 Q4 KV2 Dh16 F48 W32 M8 C32 R8 K64 TP2";
            row[i].priority=100;
            if(i%4==RAD_DLOOP_BASE) {
                row[i].computes="Batched FP32 DLoop backbone; cached context KV and single-stream exact shard head/top64";
                row[i].scratch=rad_mine_dloop_base_scratch;
            }
            if(i%4==RAD_DLOOP_DARTREE) {
                row[i].computes="Depth-batched full-prefix DARTree; persistent weight tiles, shared context KV, stable fixed/pruned selection";
                row[i].scratch=rad_mine_dloop_tree_scratch;
            }
        }
    }
};
}
extern "C" int rad_mine_dloop_count(){return 8;}
extern "C" const RadKernelInfo* rad_mine_dloop_at(int i) {
    static const Rows rows;
    return i>=0&&i<8?rows.row+i:nullptr;
}
