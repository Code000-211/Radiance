#include "tree_gdn.h"
#include "dloop_full.h"
#include "dloop_ops.h"
#include "ple_tree.h"
#include "hc_mean.h"
#include "dflash_conv_add.h"
#include "dflash_k_norm_rope_store.h"
#include "dflash_gated_bf16.h"
#include "dflash_tree_select.h"
#include "mtp_tree_accum.h"
#include "mtp_scored_tree.h"
#include "qsa_tree_commit.h"
#include "mtp_pruned_tree.h"
#include "mtp_overlay.h"
#include "spechub.h"
#include "attn_paged_tree.h"

const RadKernelInfo* rad_mine_dloop_full_at(int);
extern "C" int rad_mine_dloop_count();
extern "C" const RadKernelInfo* rad_mine_dloop_at(int);
int rad_mine_hc_up_count();
const RadKernelInfo* rad_mine_hc_up_at(int);
int rad_mine_shared_i8a8_count();
const RadKernelInfo* rad_mine_shared_i8a8_at(int);

static const RadPluginInfo info = {
    RAD_PLUGIN_KERNEL, "mine", "0.1.0",
    "Model-owned GDN/PLE tree verification and DLoop drafter", "gfx1201",
};
extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() { return &info; }
/* The DFlash2 experimental rows are appended AFTER the existing families so every previously
 * published index keeps its meaning. They are experimental: nothing in the production graph
 * selects them, and they stay unselected until the dedicated comparator
 * (tests/dflash2_fusion_gpu_test.cpp) proves byte parity against libr4d. */
extern "C" int rad_kernel_schema_count() {
    return rad_gdn_tree_schema_count() + rad_dloop_schema_count() + rad_dloop_full_schema_count() +
           rad_ple_tree_schema_count() + rad_hc_mean_schema_count() +
           rad_dflash_conv_add_schema_count() + rad_dflash_k_store_schema_count() +
           rad_dflash_gated_bf16_schema_count() + rad_dflash_tree_select_schema_count() +
           rad_mtp_tree_accum_schema_count() + rad_qsa_tree_commit_schema_count() +
           rad_mtp_scored_tree_schema_count() + rad_mtp_pruned_tree_schema_count() +
           rad_mtp_overlay_schema_count() + rad_spechub_schema_count() +
           rad_attn_paged_tree_schema_count();
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    const int n = rad_gdn_tree_schema_count();
    if (i < n) return rad_gdn_tree_schema_at(i);
    i -= n;
    const int d = rad_dloop_schema_count();
    if (i < d) return rad_dloop_schema_at(i);
    i -= d;
    const int f = rad_dloop_full_schema_count();
    if (i < f) return rad_dloop_full_schema_at(i);
    i -= f;
    const int p = rad_ple_tree_schema_count();
    if (i < p) return rad_ple_tree_schema_at(i);
    i -= p;
    const int h = rad_hc_mean_schema_count();
    if (i < h) return rad_hc_mean_schema_at(i);
    i -= h;
    const int c = rad_dflash_conv_add_schema_count();
    if (i < c) return rad_dflash_conv_add_schema_at(i);
    i -= c;
    const int k = rad_dflash_k_store_schema_count();
    if (i < k) return rad_dflash_k_store_schema_at(i);
    i -= k;
    const int g = rad_dflash_gated_bf16_schema_count();
    if (i < g) return rad_dflash_gated_bf16_schema_at(i);
    i -= g;
    const int t = rad_dflash_tree_select_schema_count();
    if (i < t) return rad_dflash_tree_select_schema_at(i);
    i -= t;
    const int m = rad_mtp_tree_accum_schema_count();
    if (i < m) return rad_mtp_tree_accum_schema_at(i);
    i -= m;
    const int q = rad_qsa_tree_commit_schema_count();
    if (i < q) return rad_qsa_tree_commit_schema_at(i);
    i -= q;
    const int s = rad_mtp_scored_tree_schema_count();
    if (i < s) return rad_mtp_scored_tree_schema_at(i);
    i -= s;
    const int pruned = rad_mtp_pruned_tree_schema_count();
    if (i < pruned) return rad_mtp_pruned_tree_schema_at(i);
    i -= pruned;
    const int overlay = rad_mtp_overlay_schema_count();
    if (i < overlay) return rad_mtp_overlay_schema_at(i);
    i -= overlay;
    const int spechub = rad_spechub_schema_count();
    return i < spechub ? rad_spechub_schema_at(i) : rad_attn_paged_tree_schema_at(i - spechub);
}
extern "C" int rad_kernel_count() {
    return rad_mine_gdn_tree_count() + rad_mine_dloop_count() + rad_mine_dloop_full_count() +
           rad_mine_ple_tree_count() + rad_mine_hc_mean_count() +
           rad_mine_dflash_conv_add_count() + rad_mine_dflash_k_store_count() +
           rad_mine_dflash_gated_bf16_count() + rad_mine_dflash_tree_select_count() +
           rad_mine_mtp_tree_accum_count() + rad_mine_qsa_tree_commit_count() +
           rad_mine_mtp_scored_tree_count() + rad_mine_mtp_pruned_tree_count() +
           rad_mine_mtp_overlay_count() + rad_mine_spechub_count() +
           rad_mine_attn_paged_tree_count() + rad_mine_hc_up_count() +
           rad_mine_shared_i8a8_count();
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    const int n = rad_mine_gdn_tree_count();
    if (i < n) return rad_mine_gdn_tree_at(i);
    i -= n;
    const int d = rad_mine_dloop_count();
    if (i < d) return rad_mine_dloop_at(i);
    i -= d;
    const int f = rad_mine_dloop_full_count();
    if (i < f) return rad_mine_dloop_full_at(i);
    i -= f;
    const int p = rad_mine_ple_tree_count();
    if (i < p) return rad_mine_ple_tree_at(i);
    i -= p;
    const int h = rad_mine_hc_mean_count();
    if (i < h) return rad_mine_hc_mean_at(i);
    i -= h;
    const int c = rad_mine_dflash_conv_add_count();
    if (i < c) return rad_mine_dflash_conv_add_at(i);
    i -= c;
    const int k = rad_mine_dflash_k_store_count();
    if (i < k) return rad_mine_dflash_k_store_at(i);
    i -= k;
    const int g = rad_mine_dflash_gated_bf16_count();
    if (i < g) return rad_mine_dflash_gated_bf16_at(i);
    i -= g;
    const int t = rad_mine_dflash_tree_select_count();
    if (i < t) return rad_mine_dflash_tree_select_at(i);
    i -= t;
    const int m = rad_mine_mtp_tree_accum_count();
    if (i < m) return rad_mine_mtp_tree_accum_at(i);
    i -= m;
    const int q = rad_mine_qsa_tree_commit_count();
    if (i < q) return rad_mine_qsa_tree_commit_at(i);
    i -= q;
    const int s = rad_mine_mtp_scored_tree_count();
    if (i < s) return rad_mine_mtp_scored_tree_at(i);
    i -= s;
    const int pruned = rad_mine_mtp_pruned_tree_count();
    if (i < pruned) return rad_mine_mtp_pruned_tree_at(i);
    i -= pruned;
    const int overlay = rad_mine_mtp_overlay_count();
    if (i < overlay) return rad_mine_mtp_overlay_at(i);
    i -= overlay;
    const int spechub = rad_mine_spechub_count();
    if (i < spechub) return rad_mine_spechub_at(i);
    i -= spechub;
    const int attn = rad_mine_attn_paged_tree_count();
    if (i < attn) return rad_mine_attn_paged_tree_at(i);
    i -= attn;
    const int hc = rad_mine_hc_up_count();
    return i < hc ? rad_mine_hc_up_at(i) : rad_mine_shared_i8a8_at(i - hc);
}
