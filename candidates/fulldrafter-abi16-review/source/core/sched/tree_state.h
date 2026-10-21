#pragma once
#include "rad_core.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace rad {
inline int64_t tree_param(const OpInfo& op, const char* key, int64_t fallback = -1) {
    long long value;
    return op.base.get_i(key, &value) ? value : fallback;
}
inline bool tree_state_enabled(const Program& p) {
    for (const auto& op : p.ops) if (op.op == "gdn_tree_commit") return true;
    return false;
}
/* Tree producers declare a root-inclusive capacity and a required packed device buffer.
 * Capability checks use the capacity; readback uses the separately validated buffer handle. */
inline bool drafter_tree_declared(const Program& p) {
    return p.drafter_tree_capacity > 0;
}

/* WHETHER THE DECLARED DRAFTER FILLS A PERSISTENT FULL-CONTEXT CACHE. Asked of the op the
 * full-context block is alone in declaring, the same way tree_state_enabled asks about
 * `gdn_tree_commit`: a name in the program is a fact about what will run, and it needs no flag to
 * keep in step with it. The step loop uses this to decide whether to stage a context span at all --
 * the ring backend has no cache for one to describe. */
inline bool dloop_full_enabled(const Program& p) {
    for (const auto& op : p.ops) if (op.op == "dloop_full_append") return true;
    return false;
}
/* Shared public bound for scheduler admission and host drafter publication.
 * Derive it from the selected vocabulary's declared block quota, not metadata
 * budget: a budget not divisible by ratio still selects only floor(budget/ratio)
 * complete blocks. No QSA declaration means there is no sparse-route bound. */
inline int64_t tree_qsa_exact_to(const Program& p) {
    int64_t bound = std::numeric_limits<int64_t>::max();
    for (const auto& op : p.ops) if (op.op == "qsa_select") {
        const int64_t k = tree_param(op, "topk"), r = tree_param(op, "ratio");
        if (k > 0 && r > 0) bound = std::min(bound, k * r + r - 1);
    }
    return bound;
}
/* Explicit group/layer bindings prevent an unrelated tree op from lifting a
 * stateful model's veto. QSA's persistent projection is its winner-replay binding. */
inline bool tree_state_bound(const Program& p, size_t group) {
    const auto& g = p.kv_groups[group];
    if (g.decl.kind != RAD_KV_LINEAR && g.decl.kind != RAD_KV_CONV) return true;
    if (g.layers.empty()) return false;
    const bool ple_tok = g.name == "kv_ple_tok", ple_conv = g.name == "kv_ple_conv";
    if ((ple_tok || ple_conv) && g.decl.kind != RAD_KV_CONV) return false;
    for (int layer : g.layers) {
        if (g.name == "kv_qsa_tail") {
            char name[96];
            std::snprintf(name, sizeof(name), "blk.%d.qsa_tree_qk", layer);
            bool found = false;
            for (const auto& b : p.buffers)
                if (b.name == name && b.decl.kind == RAD_BUF_PERSIST) found = true;
            if (!found) return false;
            continue;
        }
        unsigned mask = 0;
        for (const auto& op : p.ops) {
            if (tree_param(op, "layer") != layer) continue;
            const char* key = ple_tok ? "tok_group" :
                              g.decl.kind == RAD_KV_LINEAR ? "state_group" : "conv_group";
            if (tree_param(op, key) != static_cast<int64_t>(group)) continue;
            const char* mode = op.base.get_s("mode");
            if ((ple_tok || ple_conv) && mode && !std::strcmp(mode, "prefill")) continue;
            bool served = !op.bands.empty();
            for (const auto& band : op.bands) {
                bool any = false;
                for (const auto& dom : band.dom)
                    if (dom.row && dom.row->info && dom.row->info->launch) any = true;
                served = served && any;
            }
            if (!served) continue;
            if (ple_tok || ple_conv) {
                if (op.op == "ngram_ids_tree") mask |= 1;
                if (op.op == "ple_conv_tree") mask |= 2;
                if (op.op == "ple_tree_commit") mask |= 4;
            } else {
                if (op.op == "gdn_conv_tree") mask |= 1;
                if (op.op == "gdn_recurrent_tree") mask |= 2;
                if (op.op == "gdn_conv_recurrent_tree_i8g") mask |= 3;
                if (op.op == "gdn_tree_commit") mask |= 4;
            }
        }
        const unsigned required = ple_tok ? 5 : ple_conv ? 6 : 7;
        if ((mask & required) != required) return false;
    }
    return true;
}
/* A LINEAR GROUP WHOSE DECODE NEVER REACHES THE ANCHOR KERNEL'S SCRATCH SLOT: every layer has a
 * served tree verify and winner commit, which read and commit copy 0 only. Asked of a RESOLVED
 * program -- served means a launchable row in every band, not a declaration. The QSA tail is bound
 * by its persistent projection instead and keeps the KV manager's own one-slot rule. */
inline bool tree_linear_bound(const Program& p, size_t group) {
    const auto& g = p.kv_groups[group];
    return g.decl.kind == RAD_KV_LINEAR && g.name != "kv_qsa_tail" && tree_state_bound(p, group);
}
} // namespace rad
