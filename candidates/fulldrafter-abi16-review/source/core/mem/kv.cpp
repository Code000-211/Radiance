/* kv.cpp -- see kv.h. */
#include "mem/kv.h"

#include <algorithm>
#include <cstring>

namespace rad {

const std::vector<int32_t> KVManager::kNoBlocks;

/* ------------------------------------------------------------------ group sizing */

static int64_t ceil_div(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

BundleComparison compare_bundling(const std::vector<int64_t>& bundles) {
    BundleComparison c;
    c.bundles = bundles;
    if (bundles.empty()) return c;

    auto groups = [&](int64_t g) {
        int64_t n = 0; for (int64_t b : bundles) n += ceil_div(b, g); return n;
    };
    auto slots = [&](int64_t g) {
        int64_t n = 0; for (int64_t b : bundles) n += ceil_div(b, g) * g; return n;
    };

    const int64_t lo = *std::min_element(bundles.begin(), bundles.end());
    const int64_t hi = *std::max_element(bundles.begin(), bundles.end());

    /* vLLM's rule, transcribed: the smallest bundle, escaping to the largest when they are within
     * 1.5x (gpt-oss-20b + eagle is 12 sw + 13 full, and padding to (13,13) beats (12,24)). */
    c.stock_group_size = (hi * 2 < lo * 3) ? hi : lo;
    c.stock_groups = groups(c.stock_group_size);
    c.stock_slots  = slots(c.stock_group_size);

    /* The least-waste answer under the same scheme: least slots, tie-broken on fewer groups, under
     * a hard cap of the stock group count -- without the cap, size 1 always wins on padding and
     * drowns the scheduler in per-group bookkeeping. */
    const int64_t cap = c.stock_groups;
    int64_t best = c.stock_group_size;
    for (int64_t g = 1; g <= hi; ++g) {
        if (groups(g) > cap) continue;
        if (std::pair<int64_t,int64_t>{slots(g), groups(g)} <
            std::pair<int64_t,int64_t>{slots(best), groups(best)}) best = g;
    }
    c.patched_group_size = best;
    c.patched_groups = groups(best);
    c.patched_slots  = slots(best);

    /* Ours. One group per state kind, each with its own pool, so there is nothing to pad up to. */
    c.our_groups = (int64_t)bundles.size();
    c.our_slots  = 0; for (int64_t b : bundles) c.our_slots += b;
    return c;
}

std::string BundleComparison::report() const {
    std::string b;
    for (size_t i = 0; i < bundles.size(); ++i)
        b += fmt("%s%lld", i ? "/" : "", (long long)bundles[i]);
    return fmt("KV layer bundles %s (%lld real layers)\n"
               "    vLLM stock    size %2lld -> %2lld groups, %3lld slots (%lld padding)\n"
               "    least-waste   size %2lld -> %2lld groups, %3lld slots (%lld padding)\n"
               "    radiance      per state kind, own pool -> %2lld groups, %3lld slots (0 padding)\n",
               b.c_str(), (long long)our_slots,
               (long long)stock_group_size, (long long)stock_groups, (long long)stock_slots,
               (long long)(stock_slots - our_slots),
               (long long)patched_group_size, (long long)patched_groups, (long long)patched_slots,
               (long long)(patched_slots - our_slots),
               (long long)our_groups, (long long)our_slots);
}

/* ------------------------------------------------------------------ configure */

static const char* kv_kind_name(int k) {
    switch (k) {
        case RAD_KV_FULL:   return "full";
        case RAD_KV_WINDOW: return "window";
        case RAD_KV_LINEAR: return "linear";
        case RAD_KV_CONV:   return "conv";
        default:            return "?";
    }
}

/* THE PLAN AND THE CARVE ARE SEPARATE BECAUSE THE BUDGET IS DECIDED FROM THE PLAN.
 *
 * Every per-group size here -- the page, the per-sequence state, the fixed floor -- is a function
 * of the declaration and the run configuration alone, and NONE of it depends on how many bytes the
 * pool ends up with. That matters because core/mem/vram_budget.cpp has to know what a full KV
 * cache would cost BEFORE the pools are allocated: a share larger than the contexts the cache can
 * address is VRAM that nothing will ever touch, left out of the weight plane for nothing. So the
 * budget resolver plans the groups, asks ceiling() what they could use, and caps the share there.
 * Splitting the function is what makes that one code path instead of two that have to agree.
 *
 * plan_groups() fills plans_ and fixed_; carve() spends a pool against them; configure() is both,
 * which is what the engine calls. */
int KVManager::configure(const std::vector<KVGroupInfo>& groups, const Config& cfg,
                         Pool& kv_pool) {
    RAD_TRY(plan_groups(groups, cfg));
    return carve(kv_pool);
}

int KVManager::plan_groups(const std::vector<KVGroupInfo>& groups, const Config& cfg) {
    /* The snapshot count is the operator's, through --checkpoint-slots. Nothing here derives it. */
    const int64_t budget_ckpts = cfg.checkpoint_slots > 0 ? cfg.checkpoint_slots : 0;

    forget();
    max_seqs_ = cfg.max_seqs > 0 ? cfg.max_seqs : 1;
    max_ctx_  = cfg.max_ctx > 0 ? cfg.max_ctx : 0;
    max_tok_  = cfg.max_tok > 0 ? cfg.max_tok : 0;

    if (groups.empty()) {
        RAD_ERR("KV manager: no KV groups were declared; the architecture plugin declares at "
                "least one (spec §7.2)");
        return RAD_E_INVAL;
    }

    std::vector<int64_t> bundles;
    for (size_t i = 0; i < groups.size(); ++i) {
        /* HANDLE 0 IS THE NULL GROUP. The builder reserves slot 0 of every handle table so that a
         * zero handle means "none" (rad_builder.cpp), and every loop there starts at 1. A loop
         * here that starts at 0 without this case refuses the sentinel by name -- "no layers are
         * bound to it" -- on every real model.
         *
         * The placeholder is PUSHED rather than skipped, because plan() indexes plans_ by the
         * handle. A default-constructed plan is inert everywhere it is walked: 0 blocks and 0
         * states give empty free lists, and a sequence holds nothing in it. */
        if (i == 0) {
            KVGroupPlan null_plan;
            null_plan.index = 0;
            null_plan.name  = "(none)";
            /* NOT RAD_KV_FULL, which is what a zero-initialised kind would be: -1 makes both
             * paged() and stateful() false, so every loop that switches on the kind skips it
             * instead of taking the paged branch and dividing by a zero block size. */
            null_plan.kind  = -1;
            plans_.push_back(std::move(null_plan));
            /* NOT in `bundles`: that list feeds the layer-bundling comparison report, and a zero
             * entry would report a group with no layers as the smallest real one. */
            continue;
        }
        const KVGroupInfo& g = groups[i];
        KVGroupPlan p;
        p.index    = (int32_t)i;
        p.name     = g.name.empty() ? fmt("kv%zu", i) : g.name;
        p.kind     = g.decl.kind;
        p.n_layers = (int64_t)g.layers.size();
        p.n_head_kv = g.decl.n_head_kv > 0 ? g.decl.n_head_kv : 1;

        if (p.n_layers <= 0) {
            RAD_ERR("KV group '%s': no layers are bound to it. rad_bind_layer_kv() is what tells "
                    "the block manager how many state instances a sequence owns; without it the "
                    "group's page size is a guess and this component does not guess.",
                    p.name.c_str());
            return RAD_E_INVAL;
        }
        const int64_t elt = rad_dtype_bytes(g.decl.dtype, 1);
        if (elt <= 0) {
            RAD_ERR("KV group '%s': dtype %u has no core-known size", p.name.c_str(), g.decl.dtype);
            return RAD_E_DTYPE;
        }
        bundles.push_back(p.n_layers);

        if (p.kind == RAD_KV_FULL || p.kind == RAD_KV_WINDOW) {
            p.block_size = g.block_size;
            if (p.block_size <= 0) {
                RAD_ERR("KV group '%s': block size is 0. It comes off the resolved attention "
                        "kernel's constraints at declare, never from a core constant (spec §7.2) "
                        "-- so a 0 here means the attention op for this group did not resolve.",
                        p.name.c_str());
                return RAD_E_NOKERNEL;
            }
            /* One page covers every layer bound to the group. That is where the zero-padding
             * property lives: there is no uniform page size across groups to round up to. */
            p.bytes_per_block = g.bytes_per_block > 0
                ? g.bytes_per_block
                : p.block_size * g.decl.n_head_kv * g.decl.head_dim * elt * 2 * p.n_layers;
            if (p.bytes_per_block <= 0) {
                RAD_ERR("KV group '%s': page size computes to 0 (n_head_kv=%lld head_dim=%lld)",
                        p.name.c_str(), (long long)g.decl.n_head_kv, (long long)g.decl.head_dim);
                return RAD_E_INVAL;
            }
            if (p.kind == RAD_KV_WINDOW) {
                if (g.decl.window <= 0) {
                    RAD_ERR("KV group '%s': RAD_KV_WINDOW with window 0", p.name.c_str());
                    return RAD_E_INVAL;
                }
                /* THE CAP IS THE ATTENTION WINDOW PLUS THE STEP'S FORWARD REACH. The window alone
                 * is not enough: a verify reserves 1 + n_spec rows ahead of the committed end and
                 * the NEXT step reserves another 1 + depth before this one commits, so a cap sized
                 * from the window alone recycles the oldest positions the next accepted query
                 * still reads. See window_blocks_for (rad_core.h) for the exact bound and its
                 * derivation from the attention mask. */
                p.window_reserve_tokens = speculative_window_reserve(cfg.n_spec);
                p.window_blocks = window_blocks_for(g.decl.window, p.block_size,
                                                    p.window_reserve_tokens);
            }
        } else if (p.kind == RAD_KV_LINEAR || p.kind == RAD_KV_CONV) {
            if (p.kind == RAD_KV_CONV) {
                /* A rolling window of conv_width-1 + n_spec entries, so a speculative rejection is
                 * a change of read offset rather than a recompute (spec §10). The engine's
                 * contract is that the kernel pair supports that; the manager's job is to make the
                 * slot big enough for it. */
                p.conv_slots = conv_window_entries(g.decl, cfg.n_spec);
                if (p.conv_slots <= 0) {
                    RAD_ERR("KV group '%s': conv_width %lld with n_spec %d gives 0 slots",
                            p.name.c_str(), (long long)g.decl.conv_width, cfg.n_spec);
                    return RAD_E_INVAL;
                }
                p.bytes_per_state = g.bytes_per_state > 0
                    ? g.bytes_per_state
                    : p.conv_slots * g.decl.n_head_kv * g.decl.head_dim * elt * p.n_layers;
            } else {
                p.bytes_per_state = g.bytes_per_state > 0
                    ? g.bytes_per_state
                    : g.decl.n_head_kv * g.decl.state_dim[0] * g.decl.state_dim[1] * elt
                      * p.n_layers;
                /* SPECULATION HAS NOWHERE TO PUT WHAT IT CANNOT COMMIT YET. A verify step runs
                 * 1 + n_spec tokens through the recurrence before anyone knows which survive, so
                 * on a rejection the state has folded in tokens the model never emitted -- fluent
                 * output that drifts, which nothing would report.
                 *
                 * The obvious fix is one state per candidate, which is what the plain reading of
                 * gdn_recurrent_update's state_index asks for. It is also gigabytes a SEQUENCE on a
                 * real model, tens of gigabytes at a serving concurrency, which no engine can pay.
                 * The one libr4d implements instead keeps ONE committed state and, beside it, the
                 * handful of numbers each token's rank-1 update needs -- its normalised k, its v,
                 * its decay and its beta. That block is RU_TMAX * (4*head_k + head_v + 8) floats a
                 * head against head_v * head_k for the state, so it fits inside a second slot; the
                 * next step replays the accepted prefix onto the anchor out of it and starts from a
                 * state that carries exactly the tokens that survived.
                 *
                 * Two slots a sequence, and the second is scratch the kernel owns. The engine's
                 * side of the bargain is that both are ZEROED when the slot is handed out (a zero
                 * factor replays as the identity, which is what the first step after a prefill
                 * needs) and that state_index is wide enough to name them.
                 *
                 * NOT UNDER A BOUND TREE. A group whose every layer verifies on the tree kernels
                 * and commits the winner (KVGroupInfo::tree_bound) reads and folds into copy 0 and
                 * keeps its factors in arena buffers, so a second bank would be a state's worth of
                 * pool a sequence that nothing reads. */
                /* QSA tail only ever uses copy 0 (rollback is via its enlarged ring). */
                const bool is_qsa_tail = (g.name.find("tail") != std::string::npos) ||
                                         (g.decl.n_head_kv == 1 && g.decl.state_dim[0] > 0 &&
                                          g.name.find("qsa") != std::string::npos);
                if (is_qsa_tail) {
                    p.state_copies = 1;
                } else {
                    p.state_copies = cfg.n_spec > 0 && !g.tree_bound ? 2 : 1;
                }
            }
            if (p.bytes_per_state <= 0) {
                RAD_ERR("KV group '%s': per-sequence state computes to 0", p.name.c_str());
                return RAD_E_INVAL;
            }
            /* THE SLOT THE KERNELS INDEX, which is the slot this manager has to hand out. The
             * kernels lay slots out from the state tensor ([n_states, channels, conv_slots] or
             * [n_states, n_head_kv, state_dim]); this manager lays them out from bytes_per_state,
             * and clears, saves and restores slots in that layout. Two different sizes make every
             * one of those touch a neighbouring sequence's state. */
            const int64_t indexed =
                (p.kind == RAD_KV_CONV
                     ? p.conv_slots * g.decl.n_head_kv * g.decl.head_dim
                     : g.decl.n_head_kv * g.decl.state_dim[0] * g.decl.state_dim[1])
                * elt * p.n_layers;
            if (p.bytes_per_state != indexed) {
                RAD_ERR("KV group '%s': %lld bytes a sequence are allocated, and the state the "
                        "kernels index is %lld bytes over its %lld layers -- the slot this manager "
                        "clears, checkpoints and restores would not be the slot the kernels read",
                        p.name.c_str(), (long long)p.bytes_per_state, (long long)indexed,
                        (long long)p.n_layers);
                return RAD_E_SHAPE;
            }
            /* Exactly one slot per concurrent sequence, times whatever the kernel needs beside
             * it. Not a heuristic: a sequence owns exactly one recurrent state, so max_seqs is
             * the count of LOGICAL states, full stop. */
            p.n_states = max_seqs_ * p.state_copies;
        } else {
            RAD_ERR("KV group '%s': unknown kind %d", p.name.c_str(), p.kind);
            return RAD_E_INVAL;
        }
        plans_.push_back(std::move(p));
    }
    bundling_ = compare_bundling(bundles);

    /* ---- fixed costs first: they are not negotiable and they are not per-token ---- */
    checkpoint_bytes_ = 0;
    for (const auto& p : plans_) if (p.stateful()) checkpoint_bytes_ += p.bytes_per_state;

    /* EVERY RESERVATION IS PADDED TO RAD_ALIGN_UNIT AND THE BUDGET HAS TO SAY SO. Summing the
     * fixed costs as raw bytes leaves the paged split free to spend the difference, and the plan
     * then reports a fit that the last reservation does not get.
     *
     * It takes an awkward size to show: a state or checkpoint whose byte count sits just past an
     * alignment boundary pushes most of a unit of padding into every following reservation, so the
     * shortfall grows with the slot count while each individual number still looks right.
     *
     * Padded per RESERVATION and not once at the end, because that is how the pool spends it: one
     * align_up per reserve() call, whatever the bytes. */
    int64_t fixed = 0;
    for (const auto& p : plans_)
        if (p.stateful()) fixed += align_up(p.bytes_per_state * p.n_states, RAD_ALIGN_UNIT);
    /* A SLOT IS ELASTIC ONLY WHEN A SNAPSHOT IS WORTH A GRANULE, and the stride follows from
     * that. Committing and releasing a slot on its own needs the stride to be a whole number of
     * granules: a commit rounds outward and a release rounds inward, so a straddling stride
     * would leave every slot's last partial granule backed for the life of the process and let
     * one slot's commit back part of its neighbour.
     *
     * That rounding is free on a linear-attention model, where a snapshot is a sizeable fraction
     * of a gigabyte. It is not free on a small one: a 16 KiB snapshot rounded to a 2 MiB granule
     * is 128x the address space to manage a charge that was never worth managing. Below a
     * granule the slots stay in the eagerly committed region, where they cost what they say
     * they cost and nothing reclaims them because there is nothing to reclaim. */
    const int64_t granule = Pool::granule();
    ck_elastic_ = granule > 0 && checkpoint_bytes_ >= granule;
    ck_stride_  = align_up(checkpoint_bytes_, ck_align());
    /* AND THE PAD IN FRONT OF THE REGION IS PART OF ITS COST. The slots are reserved LAST, after
     * every group, and on a ck_align() boundary -- so the region starts up to one alignment unit
     * past wherever the groups happened to end. Where that is cannot be known here, because the
     * paged split below is what decides it, so the whole unit is reserved rather than the exact
     * pad: it is at most one commit granule against a pool measured in gigabytes, and the
     * alternative is a budget the LAST slot falls off the end of -- a refusal at load on some
     * models and not others, depending on where the groups happened to end. */
    const int64_t ck_pad   = budget_ckpts > 0 ? ck_align() : 0;
    const int64_t ck_total = budget_ckpts * ck_stride_ + ck_pad;
    fixed += ck_total;
    /* And one unit a paged group, for the same reason: a page size that is not a multiple of the
     * alignment pads too, and the carve below hands the groups everything that is left. */
    for (const auto& p : plans_) if (p.paged()) fixed += RAD_ALIGN_UNIT;

    fixed_    = fixed;
    ck_total_ = ck_total;
    ckpts_    = budget_ckpts;
    return RAD_OK;
}

/* WHAT A FULL CACHE WOULD COST, in bytes, for `ctx` tokens on every one of max_seqs sequences.
 * The paged groups are the only elastic term; a windowed group cannot address past its window and
 * is capped there, which is the same cap carve() applies and for the same reason. Anything above
 * this number is memory the block manager can never hand out, and it is the whole reason
 * plan_groups() is callable on its own: the budget resolver asks this before a pool exists. */
int64_t KVManager::ceiling(int64_t ctx) const {
    if (ctx <= 0) return 0;
    int64_t need = fixed_;
    for (const auto& p : plans_) {
        if (!p.paged() || p.block_size <= 0) continue;
        /* THE CONTEXT IS THE BOUND AND THE RESERVE IS NOT ADDED TO IT. The scheduler refuses a
         * reservation whose end would pass ctx_limit (ok_ctx in block_begin), so no sequence ever
         * asks ensure() for a position at or past the context; the reserve lives INSIDE the last
         * blocks of the context, which is what window_blocks already accounts for. Adding it here
         * would budget blocks the manager can never hand out -- the exact waste this ceiling
         * exists to prevent. */
        int64_t blocks = max_seqs_ * ((ctx + p.block_size - 1) / p.block_size);
        if (p.kind == RAD_KV_WINDOW && p.window_blocks > 0)
            blocks = std::min(blocks, max_seqs_ * p.window_blocks);
        need += blocks * p.bytes_per_block;
    }
    return need;
}

/* EVERY VECTOR THE CARVE FILLS IS PARALLEL TO plans_, AND THEY ARE NOT FILLED AT THE SAME POINT IN
 * IT. The block state is sized after the split is decided and the backing is taken after that, so
 * a carve that returns part way through leaves a manager whose plan table answers every guard and
 * whose block state is empty behind it. The entry points then index one vector having checked
 * another -- reserve_block is a bounds check against plans_ and a read of free_usable_ -- and the
 * read is out of bounds rather than a refusal.
 *
 * A refused carve is a configuration the operator is told about and the caller aborts on, so the
 * cost of this is not that the engine runs badly afterwards; it is that reaching the abort goes
 * through undefined behaviour, and a tier restoring blocks on the way there is one call away from
 * a segmentation fault instead of a return code. Putting the two back in agreement by emptying
 * both is one place, and it holds for every entry point rather than the ones anybody remembered. */
int KVManager::carve(Pool& kv_pool) {
    const int rc = carve_into(kv_pool);
    if (rc < 0) forget();
    return rc;
}

void KVManager::forget() {
    plans_.clear(); free_blocks_.clear(); refcount_.clear(); free_states_.clear();
    live_.clear(); free_usable_.clear(); bucket_blocks_.clear(); held_hist_.clear();
    chunk_blocks_.clear(); short_peak_.clear();
    pool_ = nullptr;
    clears_.clear();
    seqs_.clear(); pending_.clear(); ck_slot_base_.clear(); ck_free_.clear();
    ck_stride_ = ck_region_off_ = 0;
    ck_elastic_ = false;
    ck_high_water_ = 0;
    ck_backed_ = 0;
    configured_ = false;
    report_.clear();
}

int KVManager::carve_into(Pool& kv_pool) {
    /* The pool is whatever the operator stated or the resolver derived; nothing here re-decides
     * it. What this does decide is how the groups share it, which is arithmetic, not policy. */
    const int64_t budget_bytes = kv_pool.avail();
    const int64_t budget_ckpts = ckpts_;
    const int64_t fixed        = fixed_;
    const int64_t ck_total     = ck_total_;
    pool_base_ = kv_pool.base();
    pool_ = &kv_pool;

    if (fixed >= budget_bytes) {
        RAD_ERR("vram_kv does not fit its fixed costs:\n"
                "  linear/conv state  %10s  (%lld sequences)\n"
                "  checkpoints        %10s  (%lld x %s)\n"
                "  ------------------------------\n"
                "  fixed              %10s\n"
                "  --vram-kv-mib      %10s\n"
                "  SHORT BY           %10s\n"
                "Lower --max-seqs or the checkpoint slot count, or raise --vram-kv-mib. "
                "The engine will not shrink one of them for you (spec §6).",
                humanb(fixed - ck_total).c_str(), (long long)max_seqs_,
                humanb(ck_total).c_str(), (long long)budget_ckpts,
                humanb(ck_stride_).c_str(),
                humanb(fixed).c_str(), humanb(budget_bytes).c_str(),
                humanb(fixed - budget_bytes + 1).c_str());
        return RAD_E_NOMEM;
    }

    /* ---- the paged split. Not a policy: size the groups to exhaust at the same total token
     * count. Any other split strands memory in whichever group outlasts the others, and a
     * long-context server would then run out of full-attention blocks with a window pool still
     * half empty. ---- */
    int64_t remaining = budget_bytes - fixed;
    std::vector<int32_t> open;
    for (auto& p : plans_) if (p.paged()) open.push_back(p.index);

    int64_t tokens = 0;
    while (!open.empty()) {
        double cost = 0;
        for (int32_t i : open)
            cost += (double)plans_[i].bytes_per_block / (double)plans_[i].block_size;
        if (cost <= 0) break;
        tokens = (int64_t)((double)remaining / cost);

        /* A windowed group cannot use more than window_blocks per sequence; anything past that
         * is memory nobody will ever address. Cap it and give the bytes back to the others. */
        int32_t capped = -1;
        for (int32_t i : open) {
            const auto& p = plans_[i];
            if (p.kind != RAD_KV_WINDOW) continue;
            if (tokens / p.block_size > max_seqs_ * p.window_blocks) { capped = i; break; }
        }
        if (capped < 0) break;
        plans_[capped].n_blocks = max_seqs_ * plans_[capped].window_blocks;
        remaining -= plans_[capped].n_blocks * plans_[capped].bytes_per_block;
        open.erase(std::find(open.begin(), open.end(), capped));
    }
    for (int32_t i : open) plans_[i].n_blocks = tokens / plans_[i].block_size;

    for (auto& p : plans_) {
        if (!p.paged()) continue;
        if (p.n_blocks <= 0) {
            RAD_ERR("KV group '%s': a %s KV pool leaves 0 blocks (%s a page) after the fixed "
                    "linear/conv costs of %s. Raise --vram-kv-mib, lower --expert-vs-cache-ratio, "
                    "or lower --max-seqs.",
                    p.name.c_str(), humanb(budget_bytes).c_str(),
                    humanb(p.bytes_per_block).c_str(), humanb(fixed).c_str());
            return RAD_E_NOMEM;
        }
    }

    /* ---- carve. Every reservation comes from the KV pool and only from the KV pool: the mover
     * works inside the weight slab and cannot reach this (spec §6). ---- */
    int64_t paged_taken = 0;
    for (auto& p : plans_) {
        int64_t bytes = p.paged() ? p.n_blocks * p.bytes_per_block
                                  : p.n_states * p.bytes_per_state;
        if (p.paged()) paged_taken += bytes;
        int64_t off = 0;
        void* ptr = kv_pool.reserve(p.name, bytes, RAD_ALIGN_UNIT, &off);
        if (!ptr && !kv_pool.accounting_only()) {
            RAD_ERR("KV group '%s': %s did not fit the KV pool", p.name.c_str(),
                    humanb(bytes).c_str());
            return RAD_E_NOMEM;
        }
        p.base = ptr;
        p.pool_offset = off;
        p.pool_bytes = bytes;
    }
    /* --checkpoint-slots on a model with no recurrent state buys nothing, and allocating zero-byte
     * slots for it would leave the operator's flag looking honoured. Ignore it and say so. */
    if (budget_ckpts > 0 && checkpoint_bytes_ <= 0)
        RAD_WARN("--checkpoint-slots %lld ignored: this model declares no linear-attention state, "
                 "so there is nothing to snapshot", (long long)budget_ckpts);
    for (int64_t i = 0; checkpoint_bytes_ > 0 && i < budget_ckpts; ++i) {
        int64_t off = 0;
        if (!kv_pool.reserve("linear checkpoint", ck_stride_, ck_align(), &off)
            && !kv_pool.accounting_only()) {
            /* The arithmetic that is supposed to make this impossible, printed next to the
             * failure: `fixed` reserved budget_ckpts * checkpoint_bytes_ up front, so a checkpoint
             * that does not fit means the paged carve above spent more than `remaining`. */
            RAD_ERR("linear checkpoint slot %lld (%lld B) did not fit the KV pool: %lld B of "
                    "%lld B left after the paged groups took %lld B against a %lld B share",
                    (long long)i, (long long)checkpoint_bytes_,
                    (long long)kv_pool.avail(), (long long)budget_bytes,
                    (long long)paged_taken, (long long)(budget_bytes - fixed));
            return RAD_E_NOMEM;
        }
        if (i == 0) ck_region_off_ = off;
        ck_slot_base_.push_back(off);
        ck_free_.push_back((int32_t)i);
    }
    std::reverse(ck_free_.begin(), ck_free_.end());   /* pop_back hands out slot 0 first */

    /* ---- free lists ---- */
    free_blocks_.resize(plans_.size());
    refcount_.resize(plans_.size());
    cached_.resize(plans_.size());
    cache_only_.assign(plans_.size(), 0);
    free_states_.resize(plans_.size());
    for (size_t i = 0; i < plans_.size(); ++i) {
        const auto& p = plans_[i];
        if (p.paged()) {
            refcount_[i].assign((size_t)p.n_blocks, 0);
            cached_[i].assign((size_t)p.n_blocks, 0);
            /* ASCENDING, WHICH IS ALREADY A MIN-HEAP. A sorted run satisfies the heap property,
             * so the list starts valid and hands out block 0 first. See the loan block in kv.h
             * for why the lowest free id is the one to hand out. */
            free_blocks_[i].resize((size_t)p.n_blocks);
            for (int64_t b = 0; b < p.n_blocks; ++b) free_blocks_[i][(size_t)b] = (int32_t)b;
        } else {
            /* Only the LOGICAL slots are allocatable. Copy k of slot s is s + k * n_seq_states
             * and is never handed out on its own -- it belongs to whoever holds s. */
            const int64_t n_alloc = p.n_seq_states();
            free_states_[i].resize((size_t)n_alloc);
            for (int64_t s = 0; s < n_alloc; ++s)
                free_states_[i][(size_t)s] = (int32_t)(n_alloc - 1 - s);
        }
    }

    /* ---- backing ----
     *
     * EVERYTHING THE CARVE TOOK IS BACKED BEFORE ANYTHING RUNS, and the paged groups stay backed
     * for the life of the process: a pool on a device with virtual memory management is exactly
     * one without it, the same bytes in the same places at the same addresses. The loan moves
     * which blocks the cache hands out and never what is mapped, so it needs no driver call and
     * cannot leave an address the free list names with nothing behind it.
     *
     * The fixed part -- recurrent state and conv slots -- is never released. It is sized per
     * sequence rather than per token, so there is no idle fraction of it to give back, and a
     * state slot cannot be re-created on demand the way a page can.
     *
     * CHECKPOINT SLOTS ARE NOT PART OF THAT. They are a CACHE -- the retention policy creates and
     * retires them all run, and a slot nobody has taken holds nothing that has to survive -- so
     * backing them all here would make --checkpoint-slots a standing VRAM charge whether or not a
     * snapshot exists. On a linear-attention model a slot is a sizeable fraction of a gigabyte,
     * and on a card whose experts do not fit that charge is decode step time. The region is left
     * unbacked and each slot is committed as it is handed out, which makes the flag a cap. */
    if (pool_ && pool_->elastic()) {
        /* EVERYTHING EXCEPT THE CHECKPOINT REGION. The slots are reserved last, so the
         * second range is empty and this is a prefix -- written as a hole anyway, because
         * "commit everything below the slots" is the rule and "the slots are at the end" is a
         * detail of the carve above that nothing here should depend on. A prefix that stopped at
         * a region in the MIDDLE would leave every group past it addressable and unbacked, and
         * the paged groups are live from the first step: a fault at a valid-looking address on
         * the first write, which is the worst shape this class of bug has.
         *
         * Both edges are granule-aligned by construction (the slot reserve asks for ck_align()
         * and the stride is a whole number of granules), so neither commit rounds outward into
         * the region. That alignment is the whole reason it can be skipped at all. */
        const int64_t used     = pool_->used();
        const int64_t hole_off = ck_elastic_ && !ck_slot_base_.empty() ? ck_region_off_ : used;
        const int64_t hole_end = ck_elastic_ && !ck_slot_base_.empty()
                                     ? ck_slot_base_.back() + ck_stride_
                                     : used;
        int rc = pool_->commit(0, hole_off);
        if (rc >= 0 && used > hole_end) rc = pool_->commit(hole_end, used - hole_end);
        if (rc < 0) {
            RAD_ERR("KV pool: the carve reserved %s of address space and the card would not back "
                    "it", humanb(used - (hole_end - hole_off)).c_str());
            return rc;
        }
    }
    live_.assign(plans_.size(), 0);
    free_usable_.assign(plans_.size(), 0);
    bucket_blocks_.assign(plans_.size(), 1);
    chunk_blocks_.assign(plans_.size(), 0);
    short_peak_.assign(plans_.size(), 0);
    held_hist_.assign(plans_.size(), {});
    for (size_t i = 0; i < plans_.size(); ++i) {
        const auto& p = plans_[i];
        if (!p.paged()) continue;
        live_[i]        = p.n_blocks;
        free_usable_[i] = p.n_blocks;
        /* THE BUFFER FLOOR IS ONE STEP'S WORTH OF THIS GROUP. A step batches at most max_tok
         * tokens across every sequence in it, so a buffer this size means a prefill asks the
         * driver for memory once a chunk at worst and a decode step essentially never. Clamped to
         * the carve, since a group too small to hold one chunk has no floor to speak of. */
        const int64_t chunk = blocks_for(p.index, max_tok_);
        chunk_blocks_[i] = std::min(chunk > 0 ? chunk : 1, p.n_blocks);
        /* A FEW THOUSAND BUCKETS, whatever the carve. The histogram answers "how high does
         * anything still hold" for the loan, and a bucket is the resolution of that answer: at
         * this size the loan stops within a fraction of a percent of the carve of the highest
         * held block, and the walk that finds it is a few thousand counters. */
        bucket_blocks_[i] = std::max<int64_t>(1, ceil_div(p.n_blocks, 4096));
        held_hist_[i].assign((size_t)ceil_div(p.n_blocks, bucket_blocks_[i]), 0);
    }

    configured_ = true;
    report_ = bundling_.report();
    for (const auto& p : plans_) {
        if (p.paged())
            report_ += fmt("    %-14s %-6s %3lld layers  block %4lld tok  page %8s  "
                           "%7lld blocks = %9s  (%lld tokens)\n",
                           p.name.c_str(), kv_kind_name(p.kind), (long long)p.n_layers,
                           (long long)p.block_size, humanb(p.bytes_per_block).c_str(),
                           (long long)p.n_blocks, humanb(p.pool_bytes).c_str(),
                           (long long)(p.n_blocks * p.block_size));
        else
            report_ += fmt("    %-14s %-6s %3lld layers  state %8s  %7lld slots = %9s\n",
                           p.name.c_str(), kv_kind_name(p.kind), (long long)p.n_layers,
                           humanb(p.bytes_per_state).c_str(),
                           (long long)p.n_states, humanb(p.pool_bytes).c_str());
        /* WHAT A WINDOW'S CAP IS, because it is the one number a reader cannot derive from the
         * line above: the model's window, plus the speculative reach the scheduler reserves on
         * every verify, rounded up to the straddling block. */
        if (p.kind == RAD_KV_WINDOW)
            report_ += fmt("    %-14s        window cap %lld blocks x %s = %s a sequence, "
                           "of which %lld tok are speculative reserve\n",
                           p.name.c_str(), (long long)p.window_blocks,
                           humanb(p.bytes_per_block).c_str(),
                           humanb(p.window_blocks * p.bytes_per_block).c_str(),
                           (long long)p.window_reserve_tokens);
    }
    /* WHERE EACH GROUP STARTS, SO A FAULT ADDRESS CAN BE PLACED. A GPU memory fault names an
     * address and nothing else, and without this the only way to tell "kv_attn layer 7" from
     * "checkpoint slot 3" is to reconstruct the carve by hand from the sizes above. One line. */
    report_ += fmt("    pool at %p, %s; groups at +%lld..+%lld, checkpoints at +%lld\n",
                   pool_base_, humanb(pool_ ? pool_->bytes() : 0).c_str(),
                   (long long)(plans_.size() > 1 ? plans_[1].pool_offset : 0),
                   (long long)(plans_.empty() ? 0 : plans_.back().pool_offset +
                                                    plans_.back().pool_bytes),
                   (long long)(ck_slot_base_.empty() ? 0 : ck_slot_base_.front()));
    if (checkpoint_bytes_ > 0)
        report_ += fmt(
"    %-14s %-6s      %lld slots x %s = %s%s\n", "checkpoints", "linear",
                       (long long)ck_slot_base_.size(), humanb(ck_stride_).c_str(),
                       humanb((int64_t)ck_slot_base_.size() * ck_stride_).c_str(),
                       ck_elastic_ ? "  (address space; backed as slots are taken)" : "");
    return RAD_OK;
}

std::string KVManager::report() const { return report_; }

const KVGroupPlan* KVManager::plan(int32_t g) const {
    return (g >= 0 && g < (int32_t)plans_.size()) ? &plans_[(size_t)g] : nullptr;
}

/* ------------------------------------------------------------------ free lists */

/* THE LOWEST FREE BLOCK, NOT THE ONE FREED MOST RECENTLY. A stack hands back the id released
 * last, which after a busy period is the high block the loan wants -- so the highest held block
 * never falls and nothing can be lent. A min-heap keeps the held blocks packed at the bottom,
 * which is what leaves the top free to lend.
 *
 * It also makes the live gate free. The smallest free id is below live_ whenever ANY free id is,
 * so a pool with backed free blocks finds one at the top of the heap and never scans for it.
 *
 * Entries whose block is no longer free are skipped rather than removed: reserve_specific_block
 * takes one block out of the middle, and erasing from a heap costs a rebuild. A stale entry is
 * consumed by the one pop that meets it. */
int KVManager::grab_block(int32_t g, int32_t* out) {
    auto& fl = free_blocks_[(size_t)g];
    /* AN EMPTY POOL THAT HAS LENT BLOCKS TAKES THEM BACK HERE, on the path that needed the block,
     * because this is the moment the memory is worth more to the cache than to the slab holding
     * it. ensure() recalls in one call for the whole request; this is the fallback for the callers
     * that take one block at a time, and it takes back a buffer's worth so that a sequence built
     * block by block does not recall per block. */
    if (free_usable_[(size_t)g] <= 0) (void)grow_for(g, 1);
    if (free_usable_[(size_t)g] <= 0) return RAD_E_FULL;
    while (!fl.empty()) {
        std::pop_heap(fl.begin(), fl.end(), std::greater<int32_t>());
        const int32_t b = fl.back();
        fl.pop_back();
        if (refcount_[(size_t)g][(size_t)b] != 0) continue;    /* stale: taken out of the middle */
        /* AND NEVER PAST THE LIVE MARK. An id above it addresses a page this pool never mapped,
         * and the first thing to read it is a kernel -- a GPU memory fault with nothing to say
         * which block or which group. The count and the heap are two structures and this is the
         * one place they are required to agree, so it is the one place worth asking. */
        if (b >= live_[(size_t)g]) {
            RAD_ERR("KV group '%s': the free list offered block %d with the live mark at %lld and "
                    "%lld usable -- the count and the heap have gone out of step",
                    plans_[(size_t)g].name.c_str(), (int)b, (long long)live_[(size_t)g],
                    (long long)free_usable_[(size_t)g]);
            return RAD_E_STATE;
        }
        refcount_[(size_t)g][(size_t)b] = 1;
        --free_usable_[(size_t)g];
        note_held(g, b, +1);
        *out = b;
        return RAD_OK;
    }
    return RAD_E_FULL;
}

void KVManager::drop_block(int32_t g, int32_t b) {
    auto& rc = refcount_[(size_t)g][(size_t)b];
    if (rc <= 0) { RAD_ERR("KV group %d: double free of block %d", g, b); return; }
    if (cached_[(size_t)g][(size_t)b]) {
        if (rc == 2) ++cache_only_[(size_t)g];
        else if (rc == 1) { --cache_only_[(size_t)g]; cached_[(size_t)g][(size_t)b] = 0; }
    }
    if (--rc != 0) return;
    auto& fl = free_blocks_[(size_t)g];
    fl.push_back(b);
    std::push_heap(fl.begin(), fl.end(), std::greater<int32_t>());
    ++free_usable_[(size_t)g];
    note_held(g, b, -1);
}

void KVManager::add_ref(size_t g, int32_t b) {
    auto& rc = refcount_[g][(size_t)b];
    if (rc == 1 && cached_[g][(size_t)b]) --cache_only_[g];
    ++rc;
}

/* The held count per bucket, which is the only thing a loan needs to know: the highest bucket
 * anything still holds is where the loan has to stop. Maintained here rather than scanned on demand
 * because a scan of every refcount is not affordable between steps, and a stale answer would lend
 * a block out from under a live sequence. */
void KVManager::note_held(int32_t g, int32_t b, int delta) {
    auto& h = held_hist_[(size_t)g];
    if (h.empty()) return;
    const int64_t bucket = (int64_t)b / bucket_blocks_[(size_t)g];
    if (bucket >= 0 && bucket < (int64_t)h.size()) h[(size_t)bucket] += delta;
}

int KVManager::grab_state(int32_t g, int32_t* out) {
    auto& fl = free_states_[(size_t)g];
    if (fl.empty()) return RAD_E_FULL;
    *out = fl.back(); fl.pop_back();
    /* The slot arrives dirty -- see StateClear in kv.h. Recorded rather than zeroed because this
     * runs on the scheduler thread and the pool is device memory. EVERY COPY IS CLEARED, not
     * just the state: the second one holds a kernel's replay scratch, and a zero there is what
     * makes the first step after a prefill replay nothing (KVGroupPlan::state_copies). */
    const KVGroupPlan* p = plan(g);
    const int64_t copies = p ? p->state_copies : 1, span = p ? p->n_seq_states() : 0;
    for (int64_t k = 0; k < (copies > 0 ? copies : 1); ++k)
        clears_.push_back(StateClear{ g, (int32_t)(*out + k * span) });
    return RAD_OK;
}

void KVManager::drop_state(int32_t g, int32_t s) {
    if (s >= 0) free_states_[(size_t)g].push_back(s);
}

/* WHAT CAN BE HANDED OUT, WHICH IS NOT THE LENGTH OF THE FREE LIST. The list holds every id the
 * carve created; a cache lending the top of its range still has those ids in it and a slab's
 * weights behind them, and a stale entry left by reserve_specific_block is in there too. The
 * counter is the answer to the only question anybody asks -- how many more blocks can this group
 * give out right now -- and it is maintained wherever that number changes. */
int64_t KVManager::free_blocks(int32_t g) const {
    return (g >= 0 && g < (int32_t)free_usable_.size()) ? free_usable_[(size_t)g] : 0;
}
int64_t KVManager::held_blocks(int32_t g) const {
    if (g < 0 || g >= (int32_t)live_.size()) return 0;
    const int64_t held = live_[(size_t)g] - free_usable_[(size_t)g];
    return held > 0 ? held : 0;
}
/* THE CARVE, not the backing. This is the number every block id is addressed against and the one
 * the scheduler's capacity arithmetic means; live_blocks() is what has VRAM behind it. */
int64_t KVManager::total_blocks(int32_t g) const {
    const KVGroupPlan* p = plan(g); return p ? p->n_blocks : 0;
}
int64_t KVManager::live_blocks(int32_t g) const {
    return (g >= 0 && g < (int32_t)live_.size()) ? live_[(size_t)g] : 0;
}

int64_t KVManager::committed_bytes() const { return pool_ ? pool_->committed() : 0; }

/* ONE PREFILL CHUNK, OR THE LARGEST RECENT SHORTFALL. The floor is what a single step can consume
 * -- past it a chunked prefill would recall the loan on every chunk -- and the trend term is what
 * says the last few seconds asked for more than that. `short_peak_` decays in spare_tokens(), so a
 * server that has gone quiet settles back to the floor and lends the difference to the expert
 * plane. */
int64_t KVManager::buf_blocks(size_t i) const {
    if (i >= plans_.size()) return 0;
    const int64_t floor_b = i < chunk_blocks_.size() ? chunk_blocks_[i] : 0;
    const int64_t peak    = i < short_peak_.size()   ? short_peak_[i]   : 0;
    return peak > floor_b ? peak : floor_b;
}

int64_t KVManager::buffer_bytes() const {
    int64_t b = 0;
    for (size_t i = 0; i < plans_.size(); ++i)
        if (plans_[i].paged()) b += buf_blocks(i) * plans_[i].bytes_per_block;
    return b;
}

int64_t KVManager::ck_reserve_bytes() const {
    if (!ck_elastic_ || ck_stride_ <= 0) return 0;
    /* Counted from what is BACKED and not from the free list, because a rank that is not the one
     * holding the free list still backs the slots it writes, and its reserve has to cover them. */
    const int64_t want = std::min(ckpts_, ck_high_water_ + 1);
    return want > ck_backed_ ? (want - ck_backed_) * ck_stride_ : 0;
}

bool KVManager::ck_backed(int32_t slot) const {
    if (slot < 0 || slot >= (int32_t)ck_slot_base_.size()) return false;
    if (!ck_elastic_ || !pool_ || !pool_->elastic() || ck_stride_ <= 0) return true;
    return pool_->is_committed(ck_slot_base_[(size_t)slot], ck_stride_);
}

int KVManager::ck_room(int32_t slot) const {
    if (slot < 0 || slot >= (int32_t)ck_slot_base_.size()) return RAD_E_INVAL;
    if (ck_backed(slot) || !ck_free_bytes_ || ck_keep_free_ <= 0) return RAD_OK;
    int64_t free_now = 0;
    RAD_TRY(ck_free_bytes_(&free_now));
    return free_now - ck_stride_ >= ck_keep_free_ ? RAD_OK : RAD_E_FULL;
}

int KVManager::ck_commit(int32_t slot) {
    if (slot < 0 || slot >= (int32_t)ck_slot_base_.size()) return RAD_E_INVAL;
    if (ck_backed(slot)) return RAD_OK;
    if (pool_->commit(ck_slot_base_[(size_t)slot], ck_stride_) < 0) return RAD_E_FULL;
    if (++ck_backed_ > ck_high_water_) ck_high_water_ = ck_backed_;
    return RAD_OK;
}

int KVManager::back_checkpoint(int32_t slot) {
    RAD_TRY(ck_room(slot));
    return ck_commit(slot);
}

/* EVERY CARD'S ANSWER BEFORE ANY CARD'S PAGES. Asking first means a slot one card refuses costs
 * the others nothing: on a driver that keeps what it maps, pages committed for a slot that is then
 * not handed out would be charged to that card for the life of the process. */
int KVManager::ck_back_everywhere(int32_t slot) {
    RAD_TRY(ck_room(slot));
    for (KVManager* p : ck_peers_) RAD_TRY(p->ck_room(slot));
    RAD_TRY(ck_commit(slot));
    for (KVManager* p : ck_peers_) RAD_TRY(p->ck_commit(slot));
    return RAD_OK;
}

int KVManager::back_free_checkpoints(int64_t* withdrawn) {
    std::vector<int32_t> keep;
    keep.reserve(ck_free_.size());
    int64_t gone = 0;
    for (const int32_t s : ck_free_) {
        const int rc = ck_back_everywhere(s);
        if (rc == RAD_E_FULL) { ++gone; continue; }
        if (rc < 0) return rc;
        keep.push_back(s);
    }
    ck_free_.swap(keep);
    if (withdrawn) *withdrawn = gone;
    return RAD_OK;
}

int64_t KVManager::ck_align() const {
    if (!ck_elastic_) return (int64_t)RAD_ALIGN_UNIT;
    const int64_t g = Pool::granule();
    return g > (int64_t)RAD_ALIGN_UNIT ? g : (int64_t)RAD_ALIGN_UNIT;
}

/* WHAT THE GROUP CAN STILL HAND OUT, WHICH IS NOT WHAT IT HAS BELOW THE MARK. A lent block is
 * still carved, still addressed and still the operator's budget -- it is one recall away, and the
 * recall happens on the path that needs it. So every admission decision is made against this
 * number and not against the mark, and a cache that is lending admits exactly what one that is
 * not would.
 *
 * Deciding on the mark instead would be a livelock: a 200K-token prefill asks for its blocks in
 * one call, a cache that had lent everything above its idle size would refuse, and the thing that
 * brings the loan back is demand it just refused. */
int64_t KVManager::available_blocks(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    if (!p || !p->paged()) return 0;
    const int64_t lent = p->n_blocks - live_blocks(g);
    return free_blocks(g) + (lent > 0 ? lent : 0);
}

bool KVManager::lendable(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    return p && p->kind == RAD_KV_FULL && p->n_blocks > 0 && p->block_size > 0 &&
           p->bytes_per_block > 0;
}

int64_t KVManager::lent_blocks(size_t i, int64_t tokens) const {
    const KVGroupPlan& p = plans_[i];
    if (!lendable(p.index) || tokens <= 0) return 0;
    const int64_t b = tokens / p.block_size;
    return b < p.n_blocks ? b : p.n_blocks;
}

int64_t KVManager::held_top(size_t i) const {
    const auto& h = held_hist_[i];
    for (size_t b = h.size(); b-- > 0;)
        if (h[b] > 0) return std::min<int64_t>(((int64_t)b + 1) * bucket_blocks_[i], plans_[i].n_blocks);
    return 0;
}

/* Take back enough of the loan that `blocks` more can be handed out. Unbounded: this is a request
 * that has already been admitted, and the alternative to serving it is refusing a request the
 * budget says fits. It takes back the shortfall PLUS the buffer, so a sequence that grows a block
 * at a time does not come back here for every block. */
int KVManager::grow_for(int32_t g, int64_t blocks) {
    if (g < 0 || g >= (int32_t)plans_.size()) return RAD_E_INVAL;
    const KVGroupPlan& p = plans_[(size_t)g];
    if (!p.paged()) return RAD_OK;
    const int64_t short_by = blocks - free_usable_[(size_t)g];
    if (short_by <= 0) return RAD_OK;

    /* WHAT WAS ASKED FOR AND WAS NOT THERE, which is the one honest measure of how big the buffer
     * should have been. Recorded before the recall rounds it up, and never from the recall
     * itself -- the recall already contains the buffer, so feeding it back would size the buffer
     * from its own last value. */
    if ((size_t)g < short_peak_.size() && short_by > short_peak_[(size_t)g])
        short_peak_[(size_t)g] = short_by;

    if (!lendable(g) || loan_ <= 0 || live_[(size_t)g] >= p.n_blocks) return RAD_E_FULL;
    int64_t live = live_[(size_t)g] + short_by + buf_blocks((size_t)g);
    if (live > p.n_blocks) live = p.n_blocks;
    /* The loan that leaves this group `live` blocks, in the unit every group shares. It can only
     * shrink: a shortfall in one group is never a reason to lend more of another. */
    int64_t keep = (p.n_blocks - live) * p.block_size;
    if (keep > loan_) keep = loan_;
    if (reclaim_) {
        const int rc = reclaim_(keep);
        if (rc < 0) {
            RAD_ERR("KV group '%s': a request needs %lld more blocks and the expert slab could not "
                    "give back the loan past %lld tokens: %s", p.name.c_str(),
                    (long long)short_by, (long long)keep, rad_strerror(rc));
            return rc;
        }
    }
    RAD_TRY(set_loan(keep));
    return free_usable_[(size_t)g] >= blocks ? RAD_OK : RAD_E_FULL;
}

int64_t KVManager::spare_tokens() {
    int64_t spare = -1;
    for (size_t i = 0; i < plans_.size(); ++i) {
        /* THE BURST IS FORGOTTEN A LITTLE ON EVERY PASS. Without a decay the buffer would be the
         * high-water mark of the whole run, so one 200K admission would pin its own worth of VRAM
         * for as long as the process lived -- the opposite of the point. An eighth a tick is a
         * half-life of about half a second and a return to the floor in a few, which is slower
         * than a burst arrives and faster than a workload changes. */
        if (i < short_peak_.size()) short_peak_[i] -= short_peak_[i] / 8;
        if (!lendable(plans_[i].index)) continue;
        const KVGroupPlan& p = plans_[i];
        int64_t keep = held_top(i) + buf_blocks(i);
        if (keep > p.n_blocks) keep = p.n_blocks;
        const int64_t t = (p.n_blocks - keep) * p.block_size;
        if (spare < 0 || t < spare) spare = t;
    }
    return spare > 0 ? spare : 0;
}

int KVManager::set_loan(int64_t tokens) {
    if (tokens < 0) tokens = 0;
    /* CHECKED WHOLE BEFORE ANYTHING MOVES, so a refused loan leaves every group where it was. */
    for (size_t i = 0; i < plans_.size(); ++i) {
        if (!lendable(plans_[i].index)) continue;
        const int64_t live = plans_[i].n_blocks - lent_blocks(i, tokens);
        if (live < live_[i] && held_top(i) > live) {
            RAD_ERR("KV group '%s': asked to lend down to block %lld with a block held at or above "
                    "%lld", plans_[i].name.c_str(), (long long)live,
                    (long long)(held_top(i) - bucket_blocks_[i]));
            return RAD_E_STATE;
        }
    }
    for (size_t i = 0; i < plans_.size(); ++i) {
        if (!lendable(plans_[i].index)) continue;
        const int64_t live = plans_[i].n_blocks - lent_blocks(i, tokens);
        /* THE FREE COUNT MOVES WITH THE MARK, NOT THE LIST. Every id the carve created stays in
         * the heap; what changes is how many of them may be handed out. The ids between the two
         * marks are all free -- nothing is held above the lower of them -- so the count moves by
         * exactly the distance. */
        free_usable_[i] += live - live_[i];
        live_[i] = live;
    }
    loan_ = tokens;
    return RAD_OK;
}

int KVManager::follow(const KVManager& leader) {
    if (leader.loan_ == loan_) return RAD_OK;
    for (size_t i = 0; i < plans_.size() && i < leader.plans_.size(); ++i) {
        if (!lendable(plans_[i].index)) continue;
        const int64_t live = plans_[i].n_blocks - lent_blocks(i, leader.loan_);
        free_usable_[i] += live - live_[i];
        live_[i] = live;
    }
    loan_ = leader.loan_;
    return RAD_OK;
}
int64_t KVManager::free_states(int32_t g) const {
    return (g >= 0 && g < (int32_t)free_states_.size()) ? (int64_t)free_states_[(size_t)g].size() : 0;
}

/* ONE GROUP'S SHARE OF THE CARD, AND WHAT IS HELD OF IT. Paged and stateful groups are counted by
 * the objects they actually hand out, because those are the only ones whose occupancy is tracked:
 * `pool_bytes` may exceed the product for a group whose pool was rounded up when it was carved,
 * and the rounding is capacity nothing can be allocated from rather than capacity in use. */
int64_t KVManager::pool_bytes(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    if (!p) return 0;
    /* THE PART THE CACHE KEEPS, because this figure is answering "how much of the card is the KV
     * cache". A group whose top blocks are lent still addresses them and still reports them in
     * total_blocks(), but the expert slab is holding weights in them, and the slab reports those
     * bytes as its own (Mover::flex_used_bytes); counting them here as well would count them twice. */
    if (p->paged())    return live_blocks(g) * p->bytes_per_block;
    if (p->stateful()) return p->n_states * p->bytes_per_state;
    return 0;
}

int64_t KVManager::pool_bytes_carved(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    if (!p) return 0;
    if (p->paged())    return p->n_blocks * p->bytes_per_block;
    if (p->stateful()) return p->n_states * p->bytes_per_state;
    return 0;
}

int64_t KVManager::pool_bytes_used(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    if (!p) return 0;
    if (p->paged())    return (live_blocks(g) - free_blocks(g)) * p->bytes_per_block;
    /* IN LOGICAL SLOTS, THEN SCALED. The free list holds only logical slots -- copy k of slot s
     * belongs to whoever holds s and is never handed out alone -- while n_states counts every
     * copy, so subtracting one from the other reads a speculating deployment's idle pool as
     * half full. */
    if (p->stateful()) {
        const int64_t copies = p->state_copies > 0 ? p->state_copies : 1;
        return (p->n_seq_states() - free_states(g)) * copies * p->bytes_per_state;
    }
    return 0;
}

int64_t KVManager::blocks_for(int32_t g, int64_t n_tokens) const {
    const KVGroupPlan* p = plan(g);
    if (!p || !p->paged() || n_tokens <= 0) return 0;
    int64_t n = ceil_div(n_tokens, p->block_size);
    if (p->kind == RAD_KV_WINDOW) n = std::min(n, p->window_blocks);
    return n;
}

bool KVManager::can_ever_fit(int64_t n_tokens) const {
    for (const auto& p : plans_) {
        /* Both kinds NAMED, not "paged or else". The null group at handle 0 is neither, so an
         * `else` branch that assumed stateful would refuse every sequence: it holds no states. */
        if (p.paged())         { if (blocks_for(p.index, n_tokens) > p.n_blocks) return false; }
        else if (p.stateful()) { if (p.n_states < 1) return false; }
    }
    return true;
}

/* ------------------------------------------------------------------ sequences */

KVManager::Seq* KVManager::find(uint64_t s) {
    auto it = seqs_.find(s); return it == seqs_.end() ? nullptr : &it->second;
}
const KVManager::Seq* KVManager::find(uint64_t s) const {
    auto it = seqs_.find(s); return it == seqs_.end() ? nullptr : &it->second;
}
bool KVManager::has_sequence(uint64_t s) const { return find(s) != nullptr; }

int KVManager::add_sequence(uint64_t id) {
    if (!configured_) return RAD_E_STATE;
    if (seqs_.count(id)) return RAD_E_DUPLICATE;
    Seq s;
    s.id = id;
    s.blocks.resize(plans_.size());
    s.first_block_pos.assign(plans_.size(), 0);
    s.state.assign(plans_.size(), -1);
    s.dirty_from.assign(plans_.size(), 0);
    seqs_.emplace(id, std::move(s));
    return RAD_OK;
}

void KVManager::free_sequence(uint64_t id) {
    Seq* s = find(id);
    if (!s) return;
    for (size_t g = 0; g < plans_.size(); ++g) {
        for (int32_t b : s->blocks[g]) drop_block((int32_t)g, b);
        s->blocks[g].clear();
        if (s->state[g] >= 0) { drop_state((int32_t)g, s->state[g]); s->state[g] = -1; }
    }
    seqs_.erase(id);
    ++seq_frees_;
}

int KVManager::preempt(uint64_t id) {
    Seq* s = find(id);
    if (!s) return RAD_E_NOTFOUND;
    for (size_t g = 0; g < plans_.size(); ++g) {
        for (int32_t b : s->blocks[g]) drop_block((int32_t)g, b);
        s->blocks[g].clear();
        s->first_block_pos[g] = 0;
        s->dirty_from[g] = 0;
        /* The recurrent state goes too. A GDN state that was not snapshotted cannot be resumed,
         * so a preempted sequence replays from its last checkpoint or from zero -- which is what
         * "preemption by recompute" means for a linear layer (spec §7.1). */
        if (s->state[g] >= 0) { drop_state((int32_t)g, s->state[g]); s->state[g] = -1; }
    }
    s->n_tokens = 0;
    ++seq_frees_;
    return RAD_OK;
}

/* ------------------------------------------------------------------ rollback (spec §10) */
/* The other direction from `ensure`, and deliberately NOT `preempt`. A verify wrote KV for
 * 1 + n_spec positions; the sampler accepted a prefix; the rest must go back to the free list.
 * The sequence keeps its state slots and its window base -- see kv.h for why touching the
 * stateful groups here would be actively wrong. */
int KVManager::rollback(uint64_t id, int64_t n_tokens) {
    Seq* s = find(id);
    if (!s) return RAD_E_NOTFOUND;
    if (n_tokens < 0) return RAD_E_INVAL;
    /* Monotone in the other direction is `ensure`'s job. Saying RAD_OK rather than refusing keeps
     * "every draft token was accepted" -- the good case -- off a special path in the caller. */
    if (n_tokens >= s->n_tokens) return RAD_OK;

    for (size_t g = 0; g < plans_.size(); ++g) {
        const auto& p = plans_[g];
        if (!p.paged()) continue;
        int64_t first = s->first_block_pos[g] / p.block_size;
        int64_t want  = ceil_div(n_tokens, p.block_size) - first;
        if (want < 0) {
            /* Below what a WINDOW group still retains, which needs n_spec to exceed the whole
             * window. Drop everything and rebase rather than index a block table from a negative
             * offset -- the KV for those positions was recycled and is not coming back. */
            want = 0;
            s->first_block_pos[g] = (n_tokens / p.block_size) * p.block_size;
            s->dirty_from[g] = 0;
        }
        while ((int64_t)s->blocks[g].size() > want) {
            drop_block((int32_t)g, s->blocks[g].back());
            s->blocks[g].pop_back();
        }
        s->dirty_from[g] = std::min<int64_t>(s->dirty_from[g], (int64_t)s->blocks[g].size());
    }
    s->n_tokens = n_tokens;
    return RAD_OK;
}

int64_t KVManager::first_block_pos(uint64_t id, int32_t g) const {
    const Seq* s = find(id);
    if (!s || g < 0 || g >= (int32_t)s->first_block_pos.size()) return 0;
    return s->first_block_pos[(size_t)g];
}

int64_t KVManager::seq_tokens(uint64_t id) const { const Seq* s = find(id); return s ? s->n_tokens : 0; }

int64_t KVManager::take_table_changes(uint64_t id, int32_t g) const {
    const Seq* s = find(id);
    if (!s || g < 0 || g >= (int32_t)s->dirty_from.size()) return 0;
    const int64_t d = s->dirty_from[(size_t)g];
    s->dirty_from[(size_t)g] = kNoDirty;
    return d;
}

const std::vector<int32_t>& KVManager::block_table(uint64_t id, int32_t g) const {
    const Seq* s = find(id);
    if (!s || g < 0 || g >= (int32_t)s->blocks.size()) return kNoBlocks;
    return s->blocks[(size_t)g];
}

int32_t KVManager::state_slot(uint64_t id, int32_t g) const {
    const Seq* s = find(id);
    if (!s || g < 0 || g >= (int32_t)s->state.size()) return -1;
    return s->state[(size_t)g];
}

/* The tiers' restore path. grab_block is private because every ordinary caller reaches a block
 * through a sequence; a restore has none yet -- it has bytes and needs somewhere to put them
 * before the prefix cache re-indexes them and takes the reference. */
int KVManager::reserve_block(int32_t g, int32_t* out) {
    if (!out) return RAD_E_INVAL;
    *out = -1;
    if (g < 0 || g >= (int32_t)plans_.size()) return RAD_E_INVAL;
    if (!plans_[(size_t)g].paged()) return RAD_E_INVAL;
    if (free_usable_[(size_t)g] <= 0) return RAD_E_FULL;
    return grab_block(g, out) == RAD_OK && *out >= 0 ? RAD_OK : RAD_E_FULL;
}

int64_t KVManager::reserve_blocks(int32_t g, int64_t n, std::vector<int32_t>* out) {
    if (!out || n <= 0) return 0;
    if (g < 0 || g >= (int32_t)plans_.size() || !plans_[(size_t)g].paged()) return 0;
    if (free_usable_[(size_t)g] < n) (void)grow_for(g, n);
    int64_t got = 0;
    while (got < n && free_usable_[(size_t)g] > 0) {
        int32_t b = -1;
        if (grab_block(g, &b) != RAD_OK || b < 0) break;
        out->push_back(b);
        ++got;
    }
    return got;
}

/* TAKE ONE PARTICULAR BLOCK, for the tiers' restore under tensor parallelism. Block ids are
 * SHARED ACROSS RANKS -- RadBatch carries one block table and every rank indexes its own shard of
 * the pool with it -- so a restore cannot let each rank pick its own. Rank 0 decides and the
 * others take the same id here.
 *
 * It refuses rather than substituting when the id is not free, because a rank quietly restoring
 * into a different block than its peers is the shape of bug that produces correct-looking output
 * on one card and garbage on another. */
int KVManager::reserve_specific_block(int32_t g, int32_t b) {
    if (g < 0 || g >= (int32_t)plans_.size() || !plans_[(size_t)g].paged()) return RAD_E_INVAL;
    if (b < 0 || b >= plans_[(size_t)g].n_blocks) return RAD_E_INVAL;
    /* PAST THE LIVE MARK IS A REFUSAL. A block above live_ is lent to the expert slab, and
     * restoring into one would overwrite the weights a kernel is reading out of it. Rank 0 chooses
     * from what it can hand out, so every rank can; a rank that could not is a rank whose loan is
     * deeper than its peers', and taking that block silently would be the garbled-on-one-card bug
     * this function already exists to prevent. */
    if (b >= live_[(size_t)g]) return RAD_E_STATE;
    if (refcount_[(size_t)g][(size_t)b] != 0) return RAD_E_STATE;
    /* The heap entry is LEFT BEHIND rather than erased -- erasing from a heap costs a rebuild,
     * and grab_block skips an entry whose block is no longer free. */
    refcount_[(size_t)g][(size_t)b] = 1;
    --free_usable_[(size_t)g];
    note_held(g, b, +1);
    return RAD_OK;
}

int KVManager::retain_block(int32_t g, int32_t b) {
    if (g < 0 || g >= (int32_t)refcount_.size()) return RAD_E_INVAL;
    if (b < 0 || b >= (int32_t)refcount_[(size_t)g].size()) return RAD_E_INVAL;
    if (refcount_[(size_t)g][(size_t)b] <= 0) {
        RAD_ERR("KV group %d: retain of free block %d -- the cache is naming a block it does not "
                "hold, which would serve one sequence another's context", g, b);
        return RAD_E_STATE;
    }
    add_ref((size_t)g, b);
    cached_[(size_t)g][(size_t)b] = 1;     /* at two or more: shared, so not counted */
    return RAD_OK;
}

void KVManager::cache_reserved_block(int32_t g, int32_t b) {
    if (g < 0 || g >= (int32_t)refcount_.size()) return;
    if (b < 0 || b >= (int32_t)refcount_[(size_t)g].size()) return;
    if (refcount_[(size_t)g][(size_t)b] <= 0) return;
    uint8_t& mark = cached_[(size_t)g][(size_t)b];
    if (!mark) { mark = 1; if (refcount_[(size_t)g][(size_t)b] == 1) ++cache_only_[(size_t)g]; }
}

/* A MARKED BLOCK'S RELEASE IS THE CACHE'S. The cache holds the only marked reference, and the
 * other callers -- a restore giving back what it reserved -- release blocks the cache was never
 * handed. So the mark goes with the release, and a block a sequence still holds stops counting
 * as the cache's. */
void KVManager::release_block(int32_t g, int32_t b) {
    if (g < 0 || g >= (int32_t)refcount_.size()) return;
    if (b < 0 || b >= (int32_t)refcount_[(size_t)g].size()) return;
    uint8_t& mark = cached_[(size_t)g][(size_t)b];
    if (mark && refcount_[(size_t)g][(size_t)b] > 1) mark = 0;
    drop_block(g, b);
}

int64_t KVManager::cache_only_blocks(int32_t g) const {
    if (g < 0 || g >= (int32_t)cache_only_.size()) return 0;
    return cache_only_[(size_t)g];
}

int32_t KVManager::block_refcount(int32_t g, int32_t b) const {
    if (g < 0 || g >= (int32_t)refcount_.size()) return 0;
    if (b < 0 || b >= (int32_t)refcount_[(size_t)g].size()) return 0;
    return refcount_[(size_t)g][(size_t)b];
}

std::vector<BlockCopy> KVManager::take_pending_copies() {
    std::vector<BlockCopy> out;
    out.swap(pending_);
    return out;
}

/* ------------------------------------------------------------------ ensure */

int KVManager::ensure(uint64_t id, int64_t n_tokens) {
    Seq* s = find(id);
    if (!s) return RAD_E_NOTFOUND;
    if (n_tokens <= s->n_tokens) return RAD_OK;

    /* Pass 0: recycle whatever the window has slid past. This only frees, so it cannot fail, and
     * doing it first means the availability check below sees the memory it releases. */
    for (size_t g = 0; g < plans_.size(); ++g) {
        const auto& p = plans_[g];
        if (p.kind != RAD_KV_WINDOW) continue;
        int64_t want_end = ceil_div(n_tokens, p.block_size);
        int64_t first    = s->first_block_pos[g] / p.block_size;
        int64_t target   = std::max<int64_t>(first, want_end - p.window_blocks);
        while (first < target && !s->blocks[g].empty()) {
            drop_block((int32_t)g, s->blocks[g].front());
            s->blocks[g].erase(s->blocks[g].begin());
            s->dirty_from[g] = 0;
            ++first;
        }
        if (s->blocks[g].empty()) first = target;   /* nothing held: rebase to the window */
        s->first_block_pos[g] = first * p.block_size;
    }

    /* Pass 1: can every group serve this? Check before touching anything, so a group that runs
     * out does not leave the sequence half-extended and the caller unable to say what it owns. */
    std::vector<int64_t> need(plans_.size(), 0);
    int64_t cow = 0;
    for (size_t g = 0; g < plans_.size(); ++g) {
        const auto& p = plans_[g];
        if (p.paged()) {
            int64_t first = s->first_block_pos[g] / p.block_size;
            int64_t want  = ceil_div(n_tokens, p.block_size) - first;
            need[g] = want - (int64_t)s->blocks[g].size();
            if (need[g] < 0) need[g] = 0;
            /* Copy-on-write: the partially filled block is the only one a write can reach, and if
             * somebody else holds it we must not write through. One extra block, per group. */
            if (s->n_tokens > 0 && s->n_tokens % p.block_size != 0) {
                int64_t bi = s->n_tokens / p.block_size - first;
                if (bi >= 0 && bi < (int64_t)s->blocks[g].size()
                    && refcount_[g][(size_t)s->blocks[g][(size_t)bi]] > 1) { need[g] += 1; ++cow; }
            }
            if (need[g] > available_blocks((int32_t)g)) return RAD_E_FULL;
            RAD_TRY(grow_for((int32_t)g, need[g]));
        } else if (p.stateful() && s->state[g] < 0) {
            if (free_states_[g].empty()) return RAD_E_FULL;
        }
    }
    (void)cow;

    /* Pass 2: commit. Nothing below can fail. */
    for (size_t g = 0; g < plans_.size(); ++g) {
        const auto& p = plans_[g];
        if (p.paged()) {
            int64_t first = s->first_block_pos[g] / p.block_size;
            if (s->n_tokens > 0 && s->n_tokens % p.block_size != 0) {
                int64_t bi = s->n_tokens / p.block_size - first;
                if (bi >= 0 && bi < (int64_t)s->blocks[g].size()) {
                    int32_t old = s->blocks[g][(size_t)bi];
                    if (refcount_[g][(size_t)old] > 1) {
                        int32_t fresh = -1;
                        grab_block((int32_t)g, &fresh);
                        s->blocks[g][(size_t)bi] = fresh;
                        s->dirty_from[g] = std::min<int64_t>(s->dirty_from[g], bi);
                        drop_block((int32_t)g, old);
                        pending_.push_back({(int32_t)g, old, fresh});
                    }
                }
            }
            int64_t want = ceil_div(n_tokens, p.block_size) - first;
            if ((int64_t)s->blocks[g].size() < want)
                s->dirty_from[g] = std::min<int64_t>(s->dirty_from[g],
                                                     (int64_t)s->blocks[g].size());
            while ((int64_t)s->blocks[g].size() < want) {
                int32_t b = -1;
                grab_block((int32_t)g, &b);
                s->blocks[g].push_back(b);
            }
        } else if (p.stateful() && s->state[g] < 0) {
            grab_state((int32_t)g, &s->state[g]);
        }
    }
    s->n_tokens = n_tokens;
    return RAD_OK;
}

/* ------------------------------------------------------------------ adopt / fork */

int KVManager::adopt(uint64_t id, int32_t g, const std::vector<int32_t>& blocks, int64_t n_tokens) {
    Seq* s = find(id);
    if (!s) return RAD_E_NOTFOUND;
    const KVGroupPlan* p = plan(g);
    if (!p || !p->paged()) return RAD_E_INVAL;
    /* A partial block is never shared: the next token written into it would corrupt whoever else
     * holds it, and the copy-on-write that would prevent that is pure cost on a block the cache
     * cannot serve to anyone else either. Whole blocks only. */
    if (n_tokens % p->block_size != 0) return RAD_E_INVAL;
    if ((int64_t)blocks.size() != n_tokens / p->block_size) return RAD_E_INVAL;
    if (!s->blocks[(size_t)g].empty()) return RAD_E_STATE;
    if (s->n_tokens != 0 && s->n_tokens != n_tokens) return RAD_E_INVAL;

    for (int32_t b : blocks) {
        if (b < 0 || b >= p->n_blocks) return RAD_E_INVAL;
        add_ref((size_t)g, b);
    }
    s->blocks[(size_t)g] = blocks;
    s->dirty_from[(size_t)g] = 0;
    s->n_tokens = n_tokens;
    return RAD_OK;
}

int KVManager::fork(uint64_t parent, uint64_t child) {
    Seq* pa = find(parent);
    if (!pa) return RAD_E_NOTFOUND;
    if (seqs_.count(child)) return RAD_E_DUPLICATE;

    /* Pass 1: the child needs its own state slot in every stateful group before anything is
     * shared. A recurrent state is one object and not a chain -- there is no "share the first N
     * tokens of a GDN state" -- so a fork of a linear-attention model costs a whole slot, and the
     * caller restores into it from a checkpoint (spec §7.3). */
    for (size_t g = 0; g < plans_.size(); ++g)
        if (plans_[g].stateful() && free_states_[g].empty()) return RAD_E_FULL;

    int st = add_sequence(child);
    if (st < 0) return st;
    Seq* ch = find(child);
    ch->n_tokens = pa->n_tokens;
    for (size_t g = 0; g < plans_.size(); ++g) {
        if (plans_[g].paged()) {
            ch->blocks[g] = pa->blocks[g];
            ch->first_block_pos[g] = pa->first_block_pos[g];
            for (int32_t b : ch->blocks[g]) add_ref(g, b);
        } else {
            grab_state((int32_t)g, &ch->state[g]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ make_room */

int KVManager::make_room(uint64_t id, int64_t n_tokens,
                         const std::vector<uint64_t>& victims,
                         std::vector<uint64_t>* preempted) {
    /* If it could not fit an empty pool, preempting the whole server would not help. Fail it here
     * rather than freeing every other sequence and then deadlocking on it (spec §17). */
    if (!can_ever_fit(n_tokens)) {
        RAD_WARN("request %llu wants %lld tokens, which exceeds the whole KV pool. Failing it "
                 "rather than preempting the server for a request that still would not fit.",
                 (unsigned long long)id, (long long)n_tokens);
        return RAD_E_FULL;
    }
    Seq* s = find(id);
    if (!s) return RAD_E_NOTFOUND;

    auto fits = [&]() {
        for (size_t g = 0; g < plans_.size(); ++g) {
            const auto& p = plans_[g];
            if (p.paged()) {
                int64_t first = s->first_block_pos[g] / p.block_size;
                int64_t want  = ceil_div(n_tokens, p.block_size) - first;
                if (p.kind == RAD_KV_WINDOW) want = std::min(want, p.window_blocks);
                int64_t extra = want - (int64_t)s->blocks[g].size() + 1;  /* +1 for a COW copy */
                if (extra > available_blocks((int32_t)g)) return false;
            } else if (p.stateful() && s->state[g] < 0 && free_states_[g].empty()) return false;
        }
        return true;
    };

    if (fits()) return RAD_OK;
    for (uint64_t v : victims) {
        if (v == id) continue;
        if (preempt(v) == RAD_OK && preempted) preempted->push_back(v);
        if (fits()) return RAD_OK;
    }
    return fits() ? RAD_OK : RAD_E_FULL;
}

/* ------------------------------------------------------------------ checkpoints */

/* A SLOT IS BACKED WHEN IT IS HANDED OUT, NOT WHEN IT IS CARVED. --checkpoint-slots states how
 * many snapshots the server may hold at once; what it costs the card is how many it is holding.
 *
 * THE CARD REFUSING IS A FULL POOL AND NOT AN ERROR. The slot stays in the free list and the
 * caller sees RAD_E_FULL, which is the answer it already handles -- the retention policy runs and
 * the allocation is retried. Anything else would turn a card that is momentarily busy into a
 * failed request. */
int KVManager::alloc_checkpoint(int32_t* slot_out) {
    if (ck_free_.empty()) return RAD_E_FULL;
    /* A SLOT EVERY CARD HAS BACKED ALREADY, WHERE ONE IS FREE: it costs no commit and no reading
     * of the card, and a card at its floor still has every slot it backed before. Otherwise the
     * top of the free list, backed on every card or not handed out at all. */
    size_t at = ck_free_.size() - 1;
    for (size_t i = ck_free_.size(); i-- > 0;) {
        const int32_t s = ck_free_[i];
        bool all = ck_backed(s);
        for (const KVManager* p : ck_peers_) all = all && p->ck_backed(s);
        if (all) { at = i; break; }
    }
    const int32_t slot = ck_free_[at];
    const int rc = ck_back_everywhere(slot);
    if (rc < 0) return rc;
    ck_free_.erase(ck_free_.begin() + (std::ptrdiff_t)at);
    *slot_out = slot;
    return RAD_OK;
}

void KVManager::free_checkpoint(int32_t slot) {
    if (slot < 0 || slot >= (int32_t)ck_slot_base_.size()) return;
    for (int32_t f : ck_free_) if (f == slot) return;      /* double free: refuse, do not corrupt */
    ck_free_.push_back(slot);
    /* AFTER the free list, so a release the driver refuses still leaves the slot allocatable: the
     * next commit of the same range is idempotent and the bytes are correct either way. */
    /* AND ONLY WHERE A RELEASE IS WORTH MAKING. See set_release_frees_card: on a driver that does
     * not return decommitted VRAM, unmapping these pages buys the card nothing and costs a commit
     * on the next allocation of the slot. It is not merely wasted work: the range is unmapped
     * while the slot id goes straight back on the free list, so anything still addressing it
     * reads a page that is no longer there. */
    if (ck_elastic_ && release_frees_card_ && pool_ && pool_->elastic() && ck_stride_ > 0) {
        const int64_t off = ck_slot_base_[(size_t)slot];
        if (pool_->is_committed(off, ck_stride_) && ck_backed_ > 0) --ck_backed_;
        (void)pool_->decommit(off, ck_stride_);
    }
}

void* KVManager::checkpoint_ptr(int32_t slot) const {
    if (!pool_base_ || slot < 0 || slot >= (int32_t)ck_slot_base_.size()) return nullptr;
    return (char*)pool_base_ + ck_slot_base_[(size_t)slot];
}

/* One walk, both directions. Keeping save and restore as a single traversal is deliberate: the
 * blob has no header and no self-description, so the only thing that makes it readable is that
 * exactly one piece of code decides its order. Two functions that agreed today would be two
 * functions that could stop agreeing. */
int KVManager::checkpoint_copy(const int32_t* state_slots, int n_slots, int32_t ck_slot,
                               bool save, RadStream s) {
    /* THIS RANK'S PAGES, BEFORE THIS RANK'S COPY. alloc_checkpoint backed the slot on every rank
     * before it handed the id out, so this finds it backed; a refusal here is a slot that did not
     * come from there. See KVManager::back_checkpoint. */
    RAD_TRY(back_checkpoint(ck_slot));
    char* ck = (char*)checkpoint_ptr(ck_slot);
    if (!ck) return RAD_E_INVAL;
    if (!state_slots || n_slots < (int)plans_.size()) return RAD_E_INVAL;

    int64_t off = 0;
    for (size_t g = 0; g < plans_.size(); ++g) {
        const KVGroupPlan& p = plans_[g];
        if (!p.stateful() || p.n_layers <= 0) continue;
        const int32_t slot = state_slots[g];
        if (slot < 0) return RAD_E_STATE;          /* no state to save, or none to restore into */
        const int64_t nbytes = p.layer_bytes_per_state();
        void* st0 = state_ptr((int32_t)g, 0, slot);
        if (!st0) return RAD_E_STATE;
        const int64_t group_bytes = nbytes * p.n_layers;
        if (off + group_bytes > checkpoint_bytes_) {
            RAD_ERR("checkpoint slot %d holds %lld bytes and this sequence's linear state "
                    "needs more than that; configure() sized it from the same plans, so the "
                    "two have gone out of step", (int)ck_slot, (long long)checkpoint_bytes_);
            return RAD_E_STATE;
        }
        if (p.n_layers == 1 || p.layer_stride() == nbytes) {
            RAD_TRY(save ? rad_memcpy_async(ck + off, st0, group_bytes, s)
                         : rad_memcpy_async(st0, ck + off, group_bytes, s));
        } else {
            const int64_t spitch = save ? p.layer_stride() : nbytes;
            const int64_t dpitch = save ? nbytes : p.layer_stride();
            const void* src = save ? st0 : (ck + off);
            void* dst = save ? (ck + off) : st0;
            RAD_TRY(rad_memcpy_2d_async(dst, dpitch, src, spitch, nbytes, p.n_layers, s));
        }
        off += group_bytes;
    }
    return RAD_OK;
}

int KVManager::checkpoint_save(const int32_t* slots, int n, int32_t ck, RadStream s) {
    return checkpoint_copy(slots, n, ck, /*save=*/true, s);
}
int KVManager::checkpoint_restore(const int32_t* slots, int n, int32_t ck, RadStream s) {
    return checkpoint_copy(slots, n, ck, /*save=*/false, s);
}

/* HOW MANY COLUMNS A SEQUENCE'S state_index ROW HAS. One per physical copy would be the obvious
 * answer and it is not the one libr4d wants: its anchor scheme was written against a PAGED state
 * cache, where the page a sequence's state lives in moves by -1, 0 or +1 from step to step, so it
 * names each copy THREE times and picks the column at run time out of a marker it stamped into
 * the scratch. Radiance's linear pool does not page -- a slot is a slot for the life of the
 * sequence -- so the three columns naming a copy all name the SAME slot and the marker steers a
 * read that lands in the same place whatever it says.
 *
 * Three columns a copy, therefore, and column j names copy j * copies / width. */
int64_t KVManager::state_index_width(int32_t g) const {
    const KVGroupPlan* p = plan(g);
    if (!p || !p->stateful()) return 0;
    return p->state_copies > 1 ? p->state_copies * 3 : 1;
}

int KVManager::state_index_row(uint64_t seq, int32_t g, int32_t* out, int max) const {
    const int64_t w = state_index_width(g);
    if (w <= 0 || !out || max <= 0) return 0;
    const KVGroupPlan* p = plan(g);
    const int32_t base = state_slot(seq, g);
    const int64_t span = p ? p->n_seq_states() : 0, copies = p ? p->state_copies : 1;
    const int n = (int)(w < (int64_t)max ? w : (int64_t)max);
    for (int j = 0; j < n; ++j) {
        /* A sequence with no slot has none of its copies either: -1 all the way across, which is
         * the pad every kernel already tests for. */
        out[j] = base < 0 ? -1 : (int32_t)(base + (int64_t)j * copies / w * span);
    }
    return n;
}

void* KVManager::state_ptr(int32_t g, int32_t layer, int32_t s) const {
    const KVGroupPlan* p = plan(g);
    if (!p || !p->base || s < 0 || s >= p->n_states) return nullptr;
    if (layer < 0 || layer >= p->n_layers) return nullptr;
    return (char*)p->base + (int64_t)layer * p->layer_stride()
                          + (int64_t)s * p->layer_bytes_per_state();
}

}  /* namespace rad */
