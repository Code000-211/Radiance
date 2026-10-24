#include "ref_dloop.h"
#include "ref_dloop_full.h"
#include "ref_gdn_tree.h"
#include "ref_ple_tree.h"
#include "ref_dflash_tree.h"
#include "dflash_tree_select.h"
#include "mtp_tree_accum.h"
#include "mtp_scored_tree.h"
#include "qsa_tree_commit.h"
#include "mtp_pruned_tree.h"
#include "mtp_overlay.h"
#include "spechub.h"
#include "attn_paged_tree.h"

static const RadPluginInfo info = {
    RAD_PLUGIN_KERNEL, "mine_ref", "0.1.0",
    "Host oracles for model-owned DLoop drafting and GDN/PLE tree verification", "host",
};
extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() { return &info; }
extern "C" int rad_plugin_reference() { return 1; }
/* The local DFlash2 arithmetic fusions are device-only. The tree selector and MTP
 * accumulator publish independent host oracles alongside the GDN/PLE references. */
extern "C" int rad_kernel_schema_count() {
    return rad_gdn_tree_schema_count() + rad_dloop_schema_count() + rad_dloop_full_schema_count() +
           rad_ple_tree_schema_count() + rad_dflash_tree_select_schema_count() +
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
    return rad_ref_gdn_tree_count() + rad_ref_dloop_count() + rad_ref_dloop_full_count() +
           rad_ref_ple_tree_count() + rad_ref_dflash_tree_select_count() +
           rad_ref_mtp_tree_accum_count() + rad_ref_qsa_tree_commit_count() +
           rad_ref_mtp_scored_tree_count() + rad_ref_mtp_pruned_tree_count() +
           rad_ref_mtp_overlay_count() + rad_ref_spechub_count() +
           rad_ref_attn_paged_tree_count();
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    const int n = rad_ref_gdn_tree_count();
    if (i < n) return rad_ref_gdn_tree_at(i);
    i -= n;
    const int d = rad_ref_dloop_count();
    if (i < d) return rad_ref_dloop_at(i);
    i -= d;
    const int f = rad_ref_dloop_full_count();
    if (i < f) return rad_ref_dloop_full_at(i);
    i -= f;
    const int p = rad_ref_ple_tree_count();
    if (i < p) return rad_ref_ple_tree_at(i);
    i -= p;
    const int t = rad_ref_dflash_tree_select_count();
    if (i < t) return rad_ref_dflash_tree_select_at(i);
    i -= t;
    const int m = rad_ref_mtp_tree_accum_count();
    if (i < m) return rad_ref_mtp_tree_accum_at(i);
    i -= m;
    const int q = rad_ref_qsa_tree_commit_count();
    if (i < q) return rad_ref_qsa_tree_commit_at(i);
    i -= q;
    const int s = rad_ref_mtp_scored_tree_count();
    if (i < s) return rad_ref_mtp_scored_tree_at(i);
    i -= s;
    const int pruned = rad_ref_mtp_pruned_tree_count();
    if (i < pruned) return rad_ref_mtp_pruned_tree_at(i);
    i -= pruned;
    const int overlay = rad_ref_mtp_overlay_count();
    if (i < overlay) return rad_ref_mtp_overlay_at(i);
    i -= overlay;
    const int spechub = rad_ref_spechub_count();
    return i < spechub ? rad_ref_spechub_at(i) : rad_ref_attn_paged_tree_at(i - spechub);
}
