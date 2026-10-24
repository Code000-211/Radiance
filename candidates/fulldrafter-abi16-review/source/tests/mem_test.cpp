/* mem_test.cpp -- core/mem: pools, the KV block manager, prefix caching, the SSD tier.
 *
 * The properties under test are the ones that are silent when they break: a free list that leaks
 * one block per fork looks fine for an hour and then the server stops admitting; a chained hash
 * that collides in the middle of a prefix serves one conversation another's context and produces
 * fluent wrong output with no way to bisect it (spec §17).
 */
#include "rad_test.h"

#include "mem/kv.h"
#include "mem/pools.h"
#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "mem/sstier.h"
#include "mem/kvtier.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "runtime/tierexec.h"
#include "sched/tree_state.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>

using namespace rad;

/* ------------------------------------------------------------------ fixtures */

static Config small_config() {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 1;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 8;
    c.checkpoint_interval = 64;
    c.prefix_cache = true;
    return c;
}

/* A hybrid model in miniature: full attention over 2 layers at a 16-token block, and GDN
 * recurrent state over 4 layers. Small numbers, real geometry -- the point is the arithmetic, not
 * the size. */
/* HANDLE 0 IS THE NULL GROUP. The builder reserves slot 0 of every handle table so a zero handle
 * means "none", and KVManager indexes its plans BY HANDLE -- so a group table whose first entry is
 * a real group gives that group handle 0, which every consumer reads as "no group". The sentinel
 * keeps these fixtures on the shape the engine actually produces. */
static KVGroupInfo null_group() { return KVGroupInfo{}; }

/* The HANDLES hybrid_groups() hands out, named because they are handles and not positions: with
 * the sentinel at 0 the paged group is 1 and the linear one is 2. */
enum : int32_t { G_NULL = 0, G_FULL = 1, G_GDN = 2 };

static std::vector<KVGroupInfo> hybrid_groups(bool with_linear = true) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    {
        KVGroupInfo k;
        k.name = "full";
        k.decl.kind = RAD_KV_FULL;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.block_size = 16;             /* off the resolved attention kernel, never a constant */
        k.layers = { 0, 1 };
        g.push_back(k);                /* page = 16*2*8*2*2*2 = 2048 B */
    }
    if (with_linear) {
        KVGroupInfo k;
        k.name = "gdn";
        k.decl.kind = RAD_KV_LINEAR;
        k.decl.dtype = RAD_F32;
        k.decl.n_head_kv = 4;
        k.decl.state_dim[0] = 16;
        k.decl.state_dim[1] = 16;
        k.layers = { 2, 3, 4, 5 };
        g.push_back(k);                /* state = 4*16*16*4*4 = 16 KiB */
    }
    return g;
}

struct Fixture {
    Pools pools;
    KVManager kv;
    PrefixCache pc;
    Config cfg = small_config();

    int build(bool with_linear = true, int64_t ckpt_slots = 4) {
        cfg.checkpoint_slots = with_linear ? ckpt_slots : 0;   /* --checkpoint-slots */
        return build_groups(hybrid_groups(with_linear));
    }
    int build_groups(const std::vector<KVGroupInfo>& groups) {
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        s = kv.configure(groups, cfg, pools.kv());
        if (s < 0) return s;
        return pc.configure(cfg, &kv, cfg.checkpoint_interval);
    }
};

static std::vector<int32_t> tok_run(int32_t base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}

/* Give the sequence `id` enough blocks for `n_tok` and publish its whole blocks under `tokens`. */
static void publish(Fixture& f, uint64_t id, const std::vector<int32_t>& tokens, int64_t n_tok) {
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, {}, bt, n_tok));
}

/* ================================================================== pools */

TEST(pool_budgets_are_stated_not_inferred) {
    Config c;
    c.vram_weights_mib = 0;     /* the operator said nothing */
    c.vram_kv_mib = 0;
    Pools p;
    CHECK_EQ(p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator()), RAD_E_INVAL);
    CHECK(p.fit_report().find("--vram-weights-mib") != std::string::npos);
    CHECK(p.fit_report().find("--vram-kv-mib") != std::string::npos);
    CHECK(!p.configured());
}

TEST(pool_oversubscription_is_refused_by_name) {
    Config c;
    c.vram_weights_mib  = 24 * 1024;
    c.vram_kv_mib       = 12 * 1024;
    c.gpu_headroom_mib  =  1 * 1024;
    Pools p;
    const int64_t capacity = 32ll * 1024 * 1024 * 1024;   /* a 32 GiB card */
    CHECK_EQ(p.configure(c, capacity, &rad_malloc_allocator(), &rad_malloc_allocator()),
             RAD_E_NOMEM);
    /* By name and by how much -- refusing to start beats silently shrinking something (spec §6). */
    const std::string& r = p.fit_report();
    CHECK(r.find("vram_weights") != std::string::npos);
    CHECK(r.find("vram_kv") != std::string::npos);
    CHECK(r.find("OVER BY") != std::string::npos);
    CHECK(r.find("4.00 GiB") != std::string::npos);  /* 36 GiB wanted against a 32 GiB card */
    CHECK(!p.configured());
}

TEST(pool_reservation_never_falls_through) {
    Config c = small_config();
    Pools p;
    CHECK_OK(p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));

    /* The KV pool is 1 MiB. Ask for 2 and it is refused, by name, and the pool is untouched --
     * there is no path from here to the weight slab, which is the structural half of §6: the
     * mover works inside its own slab and cannot evict a KV block to promote an expert. */
    void* q = p.kv().reserve("oversized block array", 2 << 20, RAD_ALIGN_UNIT);
    CHECK(q == nullptr);
    CHECK(!p.kv().ok());
    REQUIRE_EQ((long long)p.kv().shortfalls().size(), 1ll);
    CHECK_EQ(p.kv().shortfalls()[0].pool, std::string("vram_kv"));
    CHECK_EQ(p.kv().shortfalls()[0].missing(), (2ll << 20) - (1ll << 20));
    CHECK_EQ(p.kv().used(), 0ll);
    CHECK_EQ(p.weights().used(), 0ll);          /* nothing leaked into the weight slab */

    void* ok = p.kv().reserve("block array", 512 << 10, RAD_ALIGN_UNIT);
    CHECK(ok != nullptr);
    CHECK_EQ(p.kv().used(), 512ll << 10);
}

/* A BACKING ALLOCATION THE ALLOCATOR REFUSES IS REPORTED AS A REFUSAL, with the allocator's reason
 * where it has one -- not as "0 B available", which reads as a measurement of the card and sends
 * the reader looking for memory that was never measured. */
TEST(a_refused_backing_allocation_says_it_was_refused_and_why) {
    struct Refuses final : MemAllocator {
        void* alloc(int64_t, int) override { return nullptr; }
        void  free(void*, int) override {}
        const char* name() const override { return "refuses"; }
    } none;
    Config c = small_config();
    Pools p;
    CHECK_EQ(p.configure(c, 0, &none, &none), RAD_E_NOMEM);
    const std::string& r = p.fit_report();
    CHECK(r.find("backing allocation of") != std::string::npos);
    CHECK(r.find("refused: the allocator returned nothing") != std::string::npos);
    CHECK(r.find("0 B available") == std::string::npos);
    CHECK(!p.configured());
}

/* ================================================================== the derived budget */

TEST(derived_budget_takes_the_statics_off_the_top_then_divides_the_rest) {
    Config c = small_config();
    c.vram_weights_mib = 0;              /* both unset: derive */
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    c.expert_cache_ratio = 0.75;
    c.max_seqs = 8;
    c.max_ctx  = 2048;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);   /* driver + code objects already resident */
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;                    /* this rank's shard of the non-expert weights */
    f.experts    = 200LL << 30;                  /* far more than fits: the plane is elastic */
    f.kv_ceiling = 100LL << 30;                  /* far above any share, so no cap fires */

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));

    CHECK_EQ(b.claimable, f.capacity - (512LL << 20) - (96LL << 20) - (144LL << 20));
    CHECK(b.derived_weights && b.derived_kv);

    /* THE FLOOR IS NOT A SHARE. The statics come out whole before the ratio sees anything, so
     * lowering the ratio can never demote an attention projection to host memory. */
    CHECK_EQ(b.statics, f.statics);
    CHECK_EQ(b.elastic, b.claimable - f.statics);
    CHECK_EQ(b.experts + b.kv, b.elastic);
    CHECK_EQ(b.weights, b.statics + b.experts);

    /* Three quarters and a quarter OF THE ELASTIC PART, which is the whole point: against the
     * claimable total the weights are a much larger fraction than 0.75, and that is correct. */
    const double share = (double)b.experts / (double)b.elastic;
    CHECK(share > 0.74 && share < 0.76);

    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena
                            - c.vram_weights_mib * MiB - c.vram_kv_mib * MiB;
    CHECK(unclaimed >= 96 * MiB);                 /* never less than the headroom asked for */
    CHECK(unclaimed < 98 * MiB);                  /* and never more than MiB rounding can strand */
}

TEST(the_ratio_moves_the_elastic_part_and_only_that) {
    /* The same card at two ratios. The statics are identical in both; everything that moves is
     * the division of what was left. */
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 4LL << 30;
    f.experts    = 200LL << 30;
    f.kv_ceiling = 100LL << 30;

    Config c7 = small_config();
    c7.vram_weights_mib = 0; c7.vram_kv_mib = 0; c7.gpu_headroom_mib = 96;
    c7.expert_cache_ratio = 0.7;
    VramBudget b7;
    CHECK_OK(vram_budget_resolve(c7, { f }, &b7));

    Config c3 = small_config();
    c3.vram_weights_mib = 0; c3.vram_kv_mib = 0; c3.gpu_headroom_mib = 96;
    c3.expert_cache_ratio = 0.3;
    VramBudget b3;
    CHECK_OK(vram_budget_resolve(c3, { f }, &b3));

    CHECK_EQ(b7.statics, b3.statics);             /* the floor does not move */
    CHECK_EQ(b7.elastic, b3.elastic);
    CHECK(b7.experts > b3.experts);               /* 0.7 gives the experts more than 0.3 does */
    CHECK(b7.kv < b3.kv);
    /* And even at 0.3 the weight pool still holds every static byte. */
    CHECK(b3.weights > b3.statics);
    CHECK_EQ(b3.weights, b3.statics + b3.experts);
}

TEST(neither_side_is_given_more_than_it_could_use) {
    /* A model that FITS. The expert plane is 4 GiB and the cache can address 1.2; between them
     * they cannot spend a 32 GiB card, and the remainder is reported as unusable rather than
     * allocated into a pool that would sit empty. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    c.expert_cache_ratio = 0.75;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 4LL << 30;
    f.kv_ceiling = 1200LL << 20;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, f.experts);
    CHECK_EQ(b.kv, f.kv_ceiling);
    CHECK_EQ(b.weights, b.statics + b.experts);
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);
    CHECK(b.unusable > 20LL << 30);               /* a big card and a small model */
}

TEST(a_dense_model_has_no_expert_share_to_lose) {
    /* A dense model in miniature: 42 dense layers and not one expert. The ratio's weight share has
     * nothing to claim it, and a share nothing can claim is not slack -- it is memory no pool will
     * ever touch. Handing it to the weights anyway reserves tens of gigabytes for 0 B of experts
     * and leaves the cache short of the worst case it was asked to back. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    c.expert_cache_ratio = 0.75;
    c.max_seqs = 8;
    c.max_ctx  = 131072;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    /* DELIBERATELY NOT A WHOLE MiB. With no expert share padding it, the weights budget is the
     * static footprint to the byte, and the pools take MiB and truncate -- so this is exactly
     * the case where a pool comes out short of the weights it was sized for. */
    f.statics    = (1331LL << 20) + 12345;
    f.experts    = 0;
    f.kv_ceiling = 10LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, 0ll);
    CHECK_EQ(b.kv, f.kv_ceiling);                 /* everything a paged cache can address */
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);

    /* AND THE FLOOR IS NEVER SHORT. The budget the pool is actually built from is the MiB
     * figure, not the byte one. */
    const int64_t MiB = 1024 * 1024;
    CHECK_EQ(b.weights, f.statics);               /* no expert share left to pad it */
    CHECK(c.vram_weights_mib * MiB >= f.statics); /* and the MiB figure the pool is built from covers it */
    CHECK(c.vram_weights_mib * MiB - f.statics < MiB);
}

/* THE ENGINE'S HARDWARE QUEUES ARE A FIXED TERM. Each one costs the card VRAM the driver backs it
 * with, and the streams are made after the card is read, so a split that left them out spent
 * them out of the headroom: 67 MiB a card on a two-card Flash-Next server, leaving 9 MiB free
 * under load against 96. The rank that needs most sets the term, as every fold here does. */
TEST(the_engine_s_queues_come_off_the_card_before_the_split) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    c.expert_cache_ratio = 0.75;
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.staging    = 98LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 200LL << 30;
    f.kv_ceiling = 100LL << 30;
    VramBudget b0;
    CHECK_OK(vram_budget_resolve(c, { f }, &b0));

    /* A fresh config: the resolver writes its answer back into the one it is given. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0;
    c2.vram_kv_mib      = 0;
    c2.gpu_headroom_mib = 96;
    c2.expert_cache_ratio = 0.75;
    VramFacts g = f, h = f;
    g.n_queues = 2; g.queue_bytes = 33LL << 20;
    h.n_queues = 3; h.queue_bytes = 33LL << 20;     /* a rank with a second compute lane */
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c2, { g, h }, &b));
    CHECK_EQ(b.queues, 99LL << 20);
    CHECK_EQ(b.n_queues, 3);
    CHECK_EQ(b.claimable, b0.claimable - (99LL << 20));
    CHECK(b.report.find("hardware queues") != std::string::npos);

    /* What is left after the pools is still the headroom, with the queues and the staging paid. */
    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena - f.staging - b.queues
                            - c2.vram_weights_mib * MiB - c2.vram_kv_mib * MiB;
    CHECK(unclaimed >= 96 * MiB);
    CHECK(unclaimed < 98 * MiB);
}

TEST(a_stated_budget_still_wins_and_the_other_side_still_derives) {
    Config c = small_config();
    c.vram_weights_mib = 20000;          /* stated */
    c.vram_kv_mib      = 0;              /* derived */
    c.gpu_headroom_mib = 96;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.kv_ceiling = 100LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(c.vram_weights_mib, 20000ll);        /* untouched */
    CHECK(!b.derived_weights && b.derived_kv);
    CHECK_EQ(b.kv, b.claimable);                  /* the KV pool gets everything that is left */
}

TEST(the_smallest_card_binds_and_a_short_one_refuses_by_name) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;

    VramFacts big{}, small{};
    big.capacity   = 32LL << 30;  big.free   = big.capacity - (512LL << 20);
    big.arena      = 144LL << 20; big.kv_ceiling = 100LL << 30;
    small = big;
    small.capacity = 16LL << 30;  small.free = small.capacity - (512LL << 20);

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { big, small }, &b));
    CHECK_EQ(b.capacity, 16LL << 30);             /* one split has to fit every card it is used on */

    /* And a card the fixed terms alone overrun is a refusal with the shortfall named, not a
     * silently shrunk pool. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0; c2.vram_kv_mib = 0;
    c2.gpu_headroom_mib = 96;
    VramFacts tiny{};
    tiny.capacity = 256LL << 20;
    tiny.free     = 64LL << 20;
    tiny.arena    = 144LL << 20;
    VramBudget b2;
    CHECK(vram_budget_resolve(c2, { tiny }, &b2) < 0);
    CHECK(b2.report.find("SHORT BY") != std::string::npos);
}


/* ================================================================== group sizing */

TEST(group_sizing_wastes_nothing) {
    /* The vllm-radiance case: 48 gated-delta-net + 16 full + 5 drafter layers. vLLM's rule picks
     * the smallest bundle, 5, which divides neither 48 nor 16, and allocates 75 slots for 69 real
     * layers across 15 groups. patch_kv_group_size's search recovers 72/9. Radiance allocates 69
     * in 3, because a KV group here is one state kind with its own pool and there is no uniform
     * page size to pad up to. */
    BundleComparison c = compare_bundling({ 48, 16, 5 });
    CHECK_EQ(c.stock_group_size, 5ll);
    CHECK_EQ(c.stock_groups, 15ll);
    CHECK_EQ(c.stock_slots, 75ll);
    CHECK_EQ(c.patched_group_size, 8ll);
    CHECK_EQ(c.patched_groups, 9ll);
    CHECK_EQ(c.patched_slots, 72ll);
    CHECK_EQ(c.our_groups, 3ll);
    CHECK_EQ(c.our_slots, 69ll);           /* == sum(bundles): zero padding, structurally */

    /* An n:1 model pads nothing under vLLM's rule, and must pad nothing here either. */
    BundleComparison n1 = compare_bundling({ 50, 10 });
    CHECK_EQ(n1.stock_slots, 60ll);
    CHECK_EQ(n1.our_slots, 60ll);
    CHECK_EQ(n1.our_groups, 2ll);
}

/* ================================================================== KV manager */

TEST(kv_configure_reports_and_sizes) {
    Fixture f;
    CHECK_OK(f.build());
    const KVGroupPlan* full = f.kv.plan(G_FULL);
    const KVGroupPlan* gdn  = f.kv.plan(G_GDN);
    CHECK(full && gdn);
    CHECK_EQ(full->block_size, 16ll);
    CHECK_EQ(full->bytes_per_block, 2048ll);      /* 16 tok * 2 kv-heads * 8 dim * 2 B * K/V * 2 layers */
    CHECK_EQ(gdn->bytes_per_state, 16384ll);      /* 4 heads * 16x16 * 4 B * 4 layers */
    CHECK_EQ(gdn->n_states, 8ll);                 /* exactly max_seqs; not a heuristic */
    CHECK_EQ(f.kv.checkpoint_bytes(), 16384ll);
    CHECK_EQ(f.kv.checkpoint_slots(), 4ll);
    CHECK(full->n_blocks > 0);
    /* 1 MiB minus 8 states, 4 checkpoints and TWO ALIGNMENT UNITS, all of it in pages of 2 KiB.
     * The two units are the budget admitting what the pool actually spends: every reserve()
     * aligns its offset up, so a carve that hands the paged groups every remaining byte overruns
     * by the padding. One is the paged group's own. The other is the checkpoint region's, and it
     * is an allowance rather than an exact figure -- the slots are reserved LAST, after every
     * group, so where the region starts is decided by the paged split that has not run yet.
     * Without it the last slot does not fit a plan the report has just called a fit, on whichever
     * models happen to leave the cursor off a boundary. */
    CHECK_EQ(full->n_blocks,
             ((1ll << 20) - 8 * 16384 - 4 * 16384 - 2 * RAD_ALIGN_UNIT) / 2048);
}

TEST(kv_refuses_a_group_with_no_block_size) {
    /* Block size comes off the resolved attention kernel. A zero here means the attention op did
     * not resolve, and the manager refuses rather than inventing 16. */
    auto g = hybrid_groups();
    g[G_FULL].block_size = 0;
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(g, f.cfg, f.pools.kv()), RAD_E_NOKERNEL);
}

TEST(kv_refuses_when_fixed_costs_do_not_fit) {
    /* --max-seqs 128 at 16 KiB of GDN state each is 2 MiB of fixed cost against a 1 MiB
     * --vram-kv-mib. Refused by name, with the arithmetic printed; not silently served at fewer
     * sequences than the operator asked for. */
    Fixture f;
    f.cfg.max_seqs = 128;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    /* And the checkpoint budget is a budget: the same config fits at 8 sequences with no
     * snapshots and stops fitting once --checkpoint-slots asks for more than is left. */
    Fixture g2;
    g2.cfg.max_seqs = 8;
    g2.cfg.checkpoint_slots = 0;
    CHECK_OK(g2.pools.configure(g2.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(g2.kv.configure(hybrid_groups(), g2.cfg, g2.pools.kv()));
    Fixture g3;
    g3.cfg.max_seqs = 8;
    g3.cfg.checkpoint_slots = 64;         /* 64 x 16 KiB = 1 MiB, on top of the state */
    CHECK_OK(g3.pools.configure(g3.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(g3.kv.configure(hybrid_groups(), g3.cfg, g3.pools.kv()), RAD_E_NOMEM);
}


TEST(a_manager_whose_carve_was_refused_refuses_rather_than_reads_past_a_vector) {
    /* THE GUARD AND THE READ ARE DIFFERENT VECTORS. plan_groups() fills the plan table and carve()
     * sizes the block state against it, at two points with a failure between them, so a refused
     * carve can leave a manager whose plans_ satisfy every bounds check and whose free_usable_ is
     * empty. reserve_block is that pair exactly -- a check against plans_ and a
     * read of free_usable_ -- and it is the tiers' restore path, reached from a cache on disk
     * rather than from the scheduler.
     *
     * The operator sees the refusal and the caller aborts; what this pins is that getting there is
     * a return code and not a read past the end of a vector. */
    Fixture f;
    f.cfg.max_seqs = 128;                 /* the fixed costs do not fit, as the case above shows */
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    int32_t b = 7;
    CHECK(f.kv.reserve_block(G_FULL, &b) < 0);
    CHECK_EQ(b, -1);                      /* and it did not leave a block id a caller would use */
    CHECK(f.kv.reserve_specific_block(G_FULL, 0) < 0);
    CHECK(f.kv.retain_block(G_FULL, 0) < 0);
    CHECK_EQ(f.kv.n_groups(), 0);
    CHECK_EQ(f.kv.total_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.live_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0);
}

TEST(kv_alloc_extend_free_and_the_free_list) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);
    const int64_t s0 = f.kv.free_states(G_GDN);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 1));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK(f.kv.state_slot(1, G_GDN) >= 0);                 /* the same manager owns the GDN slot */
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 1);
    CHECK_EQ(f.kv.free_states(G_GDN), s0 - 1);

    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK_OK(f.kv.ensure(1, 17));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 2ll);
    CHECK_OK(f.kv.ensure(1, 64));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 4ll);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
    CHECK(!f.kv.has_sequence(1));

    /* Churn: a free list that leaks one block per sequence looks fine for an hour. */
    for (uint64_t i = 0; i < 200; ++i) {
        CHECK_OK(f.kv.add_sequence(100 + i));
        CHECK_OK(f.kv.ensure(100 + i, 40));
        f.kv.free_sequence(100 + i);
    }
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
}

TEST(kv_copy_on_write_on_fork) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 24));                       /* 2 blocks; the second is half full */
    auto parent = f.kv.block_table(1, G_FULL);
    CHECK_EQ((long long)parent.size(), 2ll);

    CHECK_OK(f.kv.fork(1, 2));
    CHECK(f.kv.block_table(2, G_FULL) == parent);            /* shared, no copy yet */
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[0]), 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 2);              /* a fork of 24 tokens costs no blocks */
    CHECK(f.kv.state_slot(2, G_GDN) >= 0);
    CHECK(f.kv.state_slot(2, G_GDN) != f.kv.state_slot(1, G_GDN));  /* a GDN state cannot be shared */

    /* The child writes. Only the PARTIAL block is copied; the full one stays shared. */
    CHECK_OK(f.kv.ensure(2, 32));
    auto child = f.kv.block_table(2, G_FULL);
    CHECK_EQ(child[0], parent[0]);
    CHECK(child[1] != parent[1]);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 1);
    auto copies = f.kv.take_pending_copies();
    CHECK_EQ((long long)copies.size(), 1ll);
    CHECK_EQ(copies[0].src_block, parent[1]);
    CHECK_EQ(copies[0].dst_block, child[1]);
    CHECK_EQ(copies[0].group, (int32_t)G_FULL);

    /* The parent now holds its partial block alone, so its next write copies nothing. */
    CHECK_OK(f.kv.ensure(1, 32));
    CHECK_EQ(f.kv.block_table(1, G_FULL)[1], parent[1]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);

    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
}

TEST(kv_preemption_by_recompute_and_the_undeadlockable_request) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t total = f.kv.total_blocks(G_FULL);

    /* A request bigger than the whole pool is failed, not preempted for. Preempting the entire
     * server would not make it fit, and pretending otherwise is the deadlock §17 names. */
    CHECK(!f.kv.can_ever_fit((total + 8) * 16));
    CHECK_OK(f.kv.add_sequence(1));
    std::vector<uint64_t> none;
    CHECK_EQ(f.kv.make_room(1, (total + 8) * 16, none, nullptr), RAD_E_FULL);
    f.kv.free_sequence(1);

    /* Fill the pool, then admit one more by preempting. Four sequences, leaving state slots for
     * the hog and the newcomer -- max_seqs is 8 and a state slot is per sequence, exactly. */
    const int64_t per = 16 * 16;             /* 16 blocks each */
    for (uint64_t i = 1; i <= 4; ++i) {
        CHECK_OK(f.kv.add_sequence(i));
        CHECK_OK(f.kv.ensure(i, per));
    }

    /* Drain the rest so the next ensure genuinely cannot be served. */
    uint64_t hog = 900;
    CHECK_OK(f.kv.add_sequence(hog));
    CHECK_OK(f.kv.ensure(hog, f.kv.free_blocks(G_FULL) * 16));
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0ll);

    uint64_t newcomer = 999;
    CHECK_OK(f.kv.add_sequence(newcomer));
    CHECK_EQ(f.kv.ensure(newcomer, per), RAD_E_FULL);

    std::vector<uint64_t> victims = { hog, 1 };
    std::vector<uint64_t> preempted;
    CHECK_OK(f.kv.make_room(newcomer, per, victims, &preempted));
    CHECK(!preempted.empty());
    /* seq_tokens() == 0 IS the "needs re-prefill" reading -- see KVManager::preempt. */
    CHECK_EQ(f.kv.seq_tokens(preempted[0]), 0ll);
    CHECK(f.kv.block_table(preempted[0], G_FULL).empty());
    CHECK_OK(f.kv.ensure(newcomer, per));
}

TEST(kv_window_group_recycles_below_the_window) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "swa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = 64;
    k.block_size = 16;
    k.layers = { 0 };
    g.push_back(k);

    Config cfg = small_config();
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));
    /* window 64 at block 16, with the two tokens an ordinary decode reserves: the cap is
     * window_blocks_for(64, 16, 2) = 5 -- the four window blocks plus the straddling one. */
    CHECK_EQ(kv.plan(G_FULL)->window_blocks, 5ll);
    /* Sized against the cap, not against the whole budget: blocks past the window are memory
     * nobody will ever address. */
    CHECK_EQ(kv.total_blocks(G_FULL), cfg.max_seqs * 5);

    const int64_t n0 = kv.free_blocks(G_FULL);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 80));
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);
    CHECK_OK(kv.ensure(1, 400));
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);   /* recycled, not grown */
    CHECK_EQ(kv.free_blocks(G_FULL), n0 - 5);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);
}

/* THE WINDOW CAP MUST COVER THE RESERVATION, NOT JUST THE WINDOW, AND IT MUST BE EXACT.
 *
 * The attention mask in libref/ref_attn.cpp skips key j when pos - j >= window, so the query at
 * a committed length v reads the closed range [v - window + 1, v]. The manager recycles while
 * `ceil(reserve_end / block) - cap` blocks lie below that range, and a verify's reserve_end runs
 * `2*(n_spec + 1)` tokens past the committed end (its own 1 + n_spec rows, plus the next issued
 * step's). At every block offset on the way, the oldest position the next accepted query still
 * reads must map to the block it mapped to before the reservation.
 *
 * The sweep is the point: a defect here appears only at particular remainders of v modulo the
 * block size, and a single sample per window width reports green while the server loses history
 * once every few hundred tokens. */
TEST(kv_window_cap_keeps_history_across_speculative_reserve_and_reject) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "swa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = 64;
    k.block_size = 16;
    k.layers = { 0 };
    g.push_back(k);

    Config cfg = small_config();
    cfg.n_spec = 8;
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));

    const int64_t B = 16, W = 64;
    const int64_t reserve = speculative_window_reserve(cfg.n_spec);   /* 2 * (8 + 1) */
    const int64_t cap = window_blocks_for(W, B, reserve);
    CHECK_EQ(kv.plan(G_FULL)->window_blocks, cap);
    CHECK_EQ(kv.plan(G_FULL)->window_reserve_tokens, reserve);
    /* Exact, and TIGHTER than the old ceil(window/block) + 1 for an aligned window only when
     * the straddle is already inside the ceiling -- here it is one block more. */
    CHECK_EQ(cap, (W + reserve - 2 + B - 1) / B + 1);

    CHECK_OK(kv.add_sequence(1));
    /* Map position -> block id, the only mapping a kernel can read. */
    auto block_at = [&](int64_t pos) -> int32_t {
        const std::vector<int32_t>& bt = kv.block_table(1, G_FULL);
        const int64_t first = kv.first_block_pos(1, G_FULL);
        const int64_t bi = (pos - first) / B;
        if (pos < first || bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return bt[(size_t)bi];
    };
    auto mapped = [&](int64_t lo, int64_t hi) {
        for (int64_t p = lo < 0 ? 0 : lo; p <= hi; ++p)
            if (block_at(p) < 0) return false;
        return true;
    };

    /* v is the committed length at the start of a step. At every offset: take the block ids the
     * query at v reads, make the step's full reservation, and require that they did not move --
     * then reject every draft and require that the next query's window is still addressable. */
    for (int64_t v = 1; v < 1200; ++v) {
        CHECK_OK(kv.ensure(1, v));
        /* Written rows are 0..v-1; the query the step issues sits at v and reads
         * [v-W+1, v], so v itself must be addressable but holds nothing yet. */
        const int64_t lo = v - W + 1 > 0 ? v - W + 1 : 0;
        std::vector<int32_t> before;
        for (int64_t p = lo; p <= v - 1; ++p) {
            before.push_back(block_at(p));
            CHECK(before.back() >= 0);
        }

        /* The step's own reservation: 1 + n_spec verified rows and a block drafter's span. A
         * reservation that evicts anything in [v-W+1, v-1] loses history the next accepted query
         * still attends to, and at a block granularity that only shows at some offsets of v. */
        CHECK_OK(kv.ensure(1, v + reserve));
        CHECK(mapped(lo, v));
        size_t i = 0;
        for (int64_t p = lo; p <= v - 1; ++p, ++i)
            CHECK_EQ(block_at(p), before[i]);

        /* Every draft rejected: one token commits. The rows committed so far must all still be
         * addressable; the row the next query writes is added by the next ensure, so it is
         * checked after the reservation below rather than here. */
        CHECK_OK(kv.rollback(1, v + 1));
        CHECK(mapped(v + 2 - W, v));

        /* And the next step's reservation, on top of the rejected one, may not move the ids the
         * query at v+1 reads, and must cover the row it writes. */
        CHECK_OK(kv.ensure(1, v + 1 + reserve));
        CHECK(mapped(v + 2 - W, v + 1));

        /* Never more than the cap, whatever the reservation. */
        CHECK((int64_t)kv.block_table(1, G_FULL).size() <= cap);
        /* Back to the committed length, so the next iteration's window is measured from it --
         * this is the rejection the loop is about, not a leftover reservation. */
        CHECK_OK(kv.rollback(1, v + 1));
    }
    kv.free_sequence(1);
}

/* CAPACITY IS CONSTANT IN THE CONTEXT, at the deployment's real geometry: a 2048-token window at
 * a 16-token block with the DFlash2 depth of 8, planned at two contexts three orders of magnitude
 * apart. Neither the per-sequence block cap nor the byte ceiling may grow with the context -- that
 * is the whole point of declaring a window, and a regression here is a pool that stops fitting. */
TEST(kv_window_capacity_is_constant_at_long_context) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "draft";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 8;
    k.decl.head_dim = 128;
    k.decl.window = 2048;
    k.block_size = 16;
    k.layers = { 0, 1, 2, 3, 4 };
    g.push_back(k);

    Config cfg = small_config();
    cfg.n_spec = 8;
    cfg.checkpoint_slots = 0;
    cfg.max_seqs = 4;

    KVManager a, b;
    CHECK_OK(a.plan_groups(g, cfg));
    Config long_ctx = cfg;
    long_ctx.max_ctx = 262144;
    CHECK_OK(b.plan_groups(g, long_ctx));

    const int64_t reserve = speculative_window_reserve(cfg.n_spec);
    CHECK_EQ(a.plan(G_FULL)->window_blocks, window_blocks_for(2048, 16, reserve));
    CHECK_EQ(a.plan(G_FULL)->window_blocks, 130ll);          /* 128 history + 2 of reach */
    CHECK_EQ(a.plan(G_FULL)->window_blocks, b.plan(G_FULL)->window_blocks);
    /* A FULL-LENGTH CACHE COSTS THE SAME AT ANY CONTEXT PAST THE WINDOW: the window caps the
     * group, so the ceiling stops growing with `ctx` the moment ctx passes the window. At 2048
     * the cache is still being filled and costs only the window's own 128 blocks. */
    const int64_t bpb = a.plan(G_FULL)->bytes_per_block;
    const int64_t capped = a.fixed_bytes() + cfg.max_seqs * 130 * bpb;
    CHECK_EQ(a.ceiling(8192), capped);
    CHECK_EQ(a.ceiling(262144), capped);
    CHECK_EQ(b.ceiling(262144), capped);
    CHECK_EQ(a.ceiling(2048), a.fixed_bytes() + cfg.max_seqs * 128 * bpb);
    /* One page is 16 tokens x 8 kv heads x 128 dims x 2 bytes x 2 (K and V) x 5 layers. */
    CHECK_EQ(a.plan(G_FULL)->bytes_per_block, 327680ll);
    /* In this fixture's geometry the window alone is 40 MiB a sequence and the cap's two extra
     * blocks are 2 x 320 KiB on top -- the documented price of the reserve, and the same number
     * at 8K context or at 256K. */
    CHECK_EQ(128 * 327680, 41943040ll);
    CHECK_EQ(130 * 327680, 42598400ll);
}

TEST(kv_conv_slot_is_a_rolling_window) {
    /* libr4d's conv update treats the state cache as a rolling window of width-1 + num_spec
     * entries and reads at the slot the last accepted token left, so a speculative rejection is a
     * change of read offset rather than a recompute (spec §10). The manager's job is to make the
     * slot big enough for that, and it is the SAME manager -- two allocators would disagree about
     * what a sequence owns. */
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "conv";
    k.decl.kind = RAD_KV_CONV;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.conv_width = 4;
    k.layers = { 0, 1 };
    g.push_back(k);                           /* handle 1 */
    g.push_back(hybrid_groups()[G_FULL]);     /* a paged group beside it, as in a real hybrid */

    Config cfg = small_config();
    cfg.n_spec = 3;
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));

    /* The conv group is handle 1 here -- the sentinel is 0, as it is in every group table. */
    const int32_t G_CONV = 1;
    CHECK_EQ(kv.plan(G_CONV)->conv_slots, 6ll);                    /* (4-1) + 3 */
    CHECK_EQ(kv.plan(G_CONV)->bytes_per_state, 6ll * 2 * 8 * 2 * 2);
    CHECK_EQ(kv.plan(G_CONV)->n_states, cfg.max_seqs);

    const int64_t s0 = kv.free_states(G_CONV);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 8));
    CHECK(kv.state_slot(1, G_CONV) >= 0);
    CHECK_EQ(kv.free_states(G_CONV), s0 - 1);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_states(G_CONV), s0);
}

/* THE SLOT THE MANAGER HANDS OUT IS THE SLOT THE KERNELS INDEX. The kernels take a conv slot's
 * layout from the state tensor's last extent, conv_width-1 + n_spec; the manager clears a slot
 * when it hands it out and saves and restores it at a checkpoint in the layout bytes_per_state
 * describes. A builder figure one entry wider a slot -- what the allocation once carried --
 * made handing slot s to a new sequence zero the start of slot s+1's window, the window of a
 * sequence still decoding. The manager refuses the disagreement rather than serve with it. */
TEST(kv_refuses_a_state_slot_the_kernels_would_not_index) {
    auto group = [](int kind, int64_t bytes_per_state) {
        KVGroupInfo k;
        k.name = "state";
        k.decl.kind = kind;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.decl.state_dim[0] = 4;
        k.decl.state_dim[1] = 8;
        k.decl.conv_width = 4;
        k.layers = { 0, 1 };
        k.bytes_per_state = bytes_per_state;
        return k;
    };
    auto configure = [](const KVGroupInfo& k) -> int {
        std::vector<KVGroupInfo> g;
        g.push_back(null_group());
        g.push_back(k);
        Config cfg = small_config();
        cfg.n_spec = 3;
        cfg.checkpoint_slots = 0;
        Pools p;
        if (p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()) < 0)
            return RAD_E_NOMEM;
        KVManager kv;
        return kv.configure(g, cfg, p.kv());
    };
    /* Conv: (4-1) + 3 = 6 entries of 2 x 8 bf16 channels, over two layers. */
    CHECK_OK(configure(group(RAD_KV_CONV, 6LL * 2 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_CONV, 7LL * 2 * 8 * 2 * 2)), RAD_E_SHAPE);
    /* Linear: 2 heads of 4 x 8 bf16, over two layers. */
    CHECK_OK(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2 + 64)), RAD_E_SHAPE);
}

TEST(kv_adopts_the_blocks_a_cache_hit_found) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto prompt = tok_run(555, 96);
    publish(f, 1, prompt, 64);                 /* 4 blocks indexed */

    std::vector<std::vector<int32_t>> found;
    CacheHit h = f.pc.lookup(prompt, {}, &found);
    CHECK_EQ(h.n_attn_tokens, 64);
    CHECK_EQ((long long)found.size(), 1ll);
    CHECK_EQ((long long)found[0].size(), 4ll);

    const int64_t before = f.kv.free_blocks(G_FULL);
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[0], found[0], h.n_attn_tokens));
    CHECK_EQ(f.kv.free_blocks(G_FULL), before);                 /* a hit costs no blocks */
    CHECK_EQ(f.kv.seq_tokens(2), 64ll);
    CHECK_EQ(f.kv.block_refcount(G_FULL, found[0][0]), 3);      /* seq 1 + cache + seq 2 */

    /* A partial block is never adopted: the next token written into it would corrupt whoever else
     * holds it, and the cache cannot serve it to anyone anyway. */
    CHECK_OK(f.kv.add_sequence(3));
    CHECK_EQ(f.kv.adopt(3, 0, found[0], 60), RAD_E_INVAL);

    /* Extending past the hit allocates only the new blocks, and the shared ones stay shared. */
    CHECK_OK(f.kv.ensure(2, 96));
    CHECK_EQ((long long)f.kv.block_table(2, G_FULL).size(), 6ll);
    CHECK_EQ(f.kv.block_table(2, G_FULL)[0], found[0][0]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);
}

/* ================================================================== chained hash */

TEST(chained_hash_identity) {
    auto a = tok_run(1000, 16);
    auto b = tok_run(2000, 16);
    auto shared = tok_run(3000, 16);
    std::vector<MMContentKey> no_mm;

    BlockHash root;
    BlockHash a0 = chain_block_hash(root, a.data(), 16, nullptr, 0, 0);
    BlockHash b0 = chain_block_hash(root, b.data(), 16, nullptr, 0, 0);
    CHECK(a0 != b0);
    CHECK(a0.valid());

    /* The same prefix hits. */
    CHECK(chain_block_hash(root, a.data(), 16, nullptr, 0, 0) == a0);

    /* The same SPAN under a different prefix does not. This is the whole point of chaining: a
     * block sitting in the middle of another conversation must not be served here. */
    BlockHash a1 = chain_block_hash(a0, shared.data(), 16, nullptr, 0, 16);
    BlockHash b1 = chain_block_hash(b0, shared.data(), 16, nullptr, 0, 16);
    CHECK(a1 != b1);

    /* The same span, same parent, different position is also a different block. */
    CHECK(chain_block_hash(a0, shared.data(), 16, nullptr, 0, 32) != a1);

    /* Multimodal hashes BY CONTENT. Identical placeholder token ids, different embeddings. */
    float img1[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float img2[8] = { 1, 2, 3, 4, 5, 6, 7, 9 };
    MMContentKey m1{ 0, 16, mm_content_hash(img1, sizeof img1) };
    MMContentKey m2{ 0, 16, mm_content_hash(img2, sizeof img2) };
    CHECK(m1.hash != m2.hash);
    BlockHash h1 = chain_block_hash(root, a.data(), 16, &m1, 1, 0);
    BlockHash h2 = chain_block_hash(root, a.data(), 16, &m2, 1, 0);
    CHECK(h1 != h2);
    CHECK(h1 != a0);                                     /* and neither aliases the text-only block */
    CHECK(chain_block_hash(root, a.data(), 16, &m1, 1, 0) == h1);   /* same image, same hash */
}

/* ================================================================== prefix cache */

TEST(prefix_cache_hits_the_shared_prefix_only) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK(f.pc.enabled());
    CHECK_EQ(f.pc.block_size(), 16ll);

    auto p1 = tok_run(10, 64);                  /* 4 blocks */
    publish(f, 1, p1, 64);

    /* Same prefix, longer prompt: three blocks reused (the fourth is the tail we keep). */
    auto p2 = p1;
    for (int i = 0; i < 32; ++i) p2.push_back(500 + i);
    CacheHit h = f.pc.lookup(p2, {});
    CHECK_EQ(h.n_attn_tokens, 64);

    /* Diverging at block 2: only blocks 0 and 1 are reusable. */
    auto p3 = p1;
    p3[35] = 77777;
    CacheHit h3 = f.pc.lookup(p3, {});
    CHECK_EQ(h3.n_attn_tokens, 32);

    /* A block in the middle of a DIFFERENT prefix does not collide: p4's blocks 1..3 are
     * byte-identical to p1's, but its block 0 is not, so nothing matches. */
    auto p4 = p1;
    for (int i = 0; i < 16; ++i) p4[i] = 9000 + i;
    CHECK_EQ(f.pc.lookup(p4, {}).n_attn_tokens, 0);

    /* A full hit is backed off by one block: the step needs a token to compute a logit from. */
    CHECK_EQ(f.pc.lookup(p1, {}).n_attn_tokens, 48);
}

TEST(prefix_cache_lru_eviction_order) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));

    auto a = tok_run(100, 32), b = tok_run(200, 32), c = tok_run(300, 32);
    publish(f, 1, a, 16);           /* one indexed block each */
    publish(f, 2, b, 16);
    publish(f, 3, c, 16);
    CHECK_EQ(f.pc.size(), 3ll);

    /* Touch A, so the order becomes B, C, A. */
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);
    auto order = f.pc.lru_order();
    CHECK_EQ((long long)order.size(), 3ll);

    CHECK_EQ(f.pc.evict_lru(1), 1ll);
    CHECK_EQ(f.pc.size(), 2ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 0);      /* B was the oldest and went first */
    CHECK_EQ(f.pc.lookup(c, {}).n_attn_tokens, 16);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);

    /* Eviction releases the cache's reference; it does not free a block a sequence still holds. */
    CHECK(f.kv.block_refcount(G_FULL, f.kv.block_table(2, G_FULL)[0]) >= 1);
}

TEST(prefix_cache_holds_a_reference_on_every_indexed_block) {
    /* The invariant that makes a hit safe: a block named by the cache is never in the free list,
     * so it can never be handed to another sequence underneath a cache entry. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    int32_t blk = f.kv.block_table(1, G_FULL)[0];
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 2);           /* sequence + cache */
    const int64_t before = f.kv.free_blocks(G_FULL);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 1);           /* still held by the cache */
    f.pc.clear();
    CHECK_EQ(f.kv.free_blocks(G_FULL), before + 2);
}

/* WHAT THE CACHE IS HOLDING, which is the figure that tells a full KV pool apart from a leaking
 * one. The pool counts a held block whoever holds it, and on a server carrying conversations most
 * of a full pool is finished turns kept for their next one -- so with a single number an operator
 * watching 300K tokens held behind one live request has no way to see which it is. */
TEST(the_cache_reports_the_context_it_is_holding) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);

    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);

    /* A SECOND CONVERSATION SHARING THE PREFIX ADDS ONLY WHAT IT ADDED. Entries are keyed by the
     * chain, so the blocks the two have in common are one entry and are held once -- which is
     * why this figure is a share of the pool and not a sum over sessions. */
    auto b = tok_run(100, 32);
    for (size_t i = 16; i < b.size(); ++i) b[i] += 1000;
    publish(f, 2, b, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 48);

    /* And it falls when the cache gives blocks back, which is the half that makes it an answer
     * to "is the pool releasing anything". */
    f.pc.evict_lru(1);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);
    f.pc.clear();
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);
}

/* ================================================================== the split lookup */

TEST(split_lookup_reuses_attention_between_checkpoints) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    CHECK_EQ(f.pc.checkpoint_interval(), 64ll);
    CHECK_EQ(f.pc.block_size(), 16ll);

    /* The scheduler splits a chunk here, so a checkpoint is always written rather than being
     * whatever a step boundary happened to land on (spec §7.3b). */
    CHECK_EQ(f.pc.next_checkpoint_after(0), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(1), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(64), 128ll);
    CHECK_EQ(f.pc.next_checkpoint_after(65), 128ll);
    CHECK(f.pc.is_checkpoint(128));
    CHECK(!f.pc.is_checkpoint(80));

    auto prompt = tok_run(7000, 336);         /* 21 blocks */
    publish(f, 1, prompt, 320);               /* 20 blocks indexed */

    /* With no checkpoint the attention hit stands alone and the linear layers replay from zero. */
    CacheHit h0 = f.pc.lookup(prompt, {});
    CHECK_EQ(h0.n_attn_tokens, 320);
    CHECK_EQ(h0.n_linear_tokens, 0);
    CHECK_EQ(h0.checkpoint_slot, -1);

    /* Write a checkpoint at 128. Its key is the chained block hash at that position, which is
     * exactly what the attention lookup already computed. */
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(/*session=*/1, 128, hs[128 / 16 - 1], &slot));
    CHECK(slot >= 0);

    /* RESERVED IS NOT WRITTEN. The reservation happens when the chunk is PLANNED; the state only
     * exists after the step has run. Offering the slot in between hands out a buffer still
     * holding the previous tenant's history, which is a wrong answer that reads perfectly -- so
     * the lookup must skip it until commit_checkpoint says the copy was issued. */
    CacheHit pending = f.pc.lookup(prompt, {});
    CHECK_EQ(pending.n_attn_tokens, 320);
    CHECK_EQ(pending.n_linear_tokens, 0);
    CHECK_EQ(pending.checkpoint_slot, -1);
    f.pc.commit_checkpoint(hs[128 / 16 - 1]);

    /* The hit lands BETWEEN checkpoints: attention is reused to 320, the linear state is restored
     * at 128, and only [128, 320) replays through the linear layers. That split is the whole
     * reason the interval is decoupled from the block size. */
    CacheHit h1 = f.pc.lookup(prompt, {});
    CHECK_EQ(h1.n_attn_tokens, 320);
    CHECK_EQ(h1.n_linear_tokens, 128);
    CHECK_EQ(h1.checkpoint_slot, slot);

    /* A later checkpoint shortens the replay. */
    int32_t slot2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 256, hs[256 / 16 - 1], &slot2));
    f.pc.commit_checkpoint(hs[256 / 16 - 1]);
    CacheHit h2 = f.pc.lookup(prompt, {});
    CHECK_EQ(h2.n_attn_tokens, 320);
    CHECK_EQ(h2.n_linear_tokens, 256);
    CHECK_EQ(h2.checkpoint_slot, slot2);

    /* A checkpoint under a DIFFERENT prefix is not reachable from here. */
    auto other = tok_run(9000, 336);
    CHECK_EQ(f.pc.lookup(other, {}).n_linear_tokens, 0);
}

TEST(checkpoint_interval_must_be_a_block_multiple) {
    Fixture f;
    f.cfg.checkpoint_interval = 100;          /* not a multiple of 16 */
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    f.cfg.checkpoint_slots = 2;
    CHECK_OK(f.kv.configure(hybrid_groups(true), f.cfg, f.pools.kv()));
    /* It would never produce a single hit, so it is refused rather than served at 0%. */
    CHECK_EQ(f.pc.configure(f.cfg, &f.kv, f.cfg.checkpoint_interval), RAD_E_INVAL);
}

/* ================================================================== retention */

TEST(geometric_backoff_keeps_the_tip_and_its_ancestors) {
    GeometricBackoff g;
    std::vector<int64_t> pos;
    for (int i = 1; i <= 16; ++i) pos.push_back(i * 64);     /* 64 .. 1024 */
    std::vector<int64_t> keep;
    g.select(pos, &keep);
    /* tip, then 1x, 2x, 4x, 8x back: 1024, 960, 896, 768, 512. */
    std::vector<int64_t> want = { 512, 768, 896, 960, 1024 };
    CHECK_EQ((long long)keep.size(), (long long)want.size());
    for (size_t i = 0; i < want.size() && i < keep.size(); ++i) CHECK_EQ(keep[i], want[i]);

    /* log(n): a 128K session at a 2048-token interval is 64 checkpoints and keeps 7 of them. */
    std::vector<int64_t> big;
    for (int i = 1; i <= 64; ++i) big.push_back(i * 2048);
    g.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 7ll);
    CHECK_EQ(keep.back(), 64ll * 2048);                       /* THE TIP IS ALWAYS KEPT */

    /* The alternatives are classes, not a rewrite (spec §19.1). */
    KeepTipOnly t; t.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 1ll);
    CHECK_EQ(keep[0], 64ll * 2048);
    KeepAll a; a.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 64ll);
}

TEST(checkpoint_retention_is_budgeted_and_keeps_the_tip) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(4242, 16 * 64);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 64; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    /* Eight checkpoints at 64, 128, ... 512 fill the pool exactly. */
    for (int i = 1; i <= 8; ++i) {
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(1, i * 64, hs[i * 4 - 1], &s));
    }
    CHECK_EQ((long long)f.pc.checkpoints(1).size(), 8ll);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth forces retention. Geometric backoff over {64..512} keeps 512, 448, 384, 256 and
     * releases the rest -- and the tip survives, because that is what an appending agent
     * transcript actually needs. */
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 9 * 64, hs[9 * 4 - 1], &s9));
    auto held = f.pc.checkpoints(1);
    CHECK(held.size() <= 6);
    CHECK_EQ(held.back(), 9ll * 64);
    bool has_512 = false;
    for (int64_t p : held) if (p == 512) has_512 = true;
    CHECK(has_512);
}

TEST(a_dead_sessions_tip_is_evictable_so_the_slot_pool_cannot_wedge) {
    /* One checkpoint each from more DISTINCT sessions than there are slots. Every one of them is
     * its session's tip, so a rule that made a tip unevictable would wedge the pool here: the
     * ninth request gets RAD_E_FULL and so does every request after it, for the life of the
     * process. A session is a request id, requests end, and nothing retires their checkpoints
     * (they are keyed by prefix hash so the next turn can still find them), so "tip" is not a
     * proxy for "live". */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(909, 16 * 8);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 8; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }

    /* One checkpoint each, under a DISTINCT prefix, so each takes its own slot rather than
     * sharing the one an equal hash would find. Eight sessions fill eight slots exactly. */
    for (uint64_t sess = 1; sess <= 8; ++sess) {
        BlockHash h = hs[3];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
        CHECK(s >= 0);
    }
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth session must still be served, by retiring the least recently used tip. */
    BlockHash fresh = hs[3];
    fresh.lo ^= 0x5eedull;
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(99, 64, fresh, &s9));
    CHECK(s9 >= 0);
    /* AND THE SESSION IT EVICTED IS FORGOTTEN. Eight in, one evicted, one added: eight tracked.
     * An index with no eraser -- retain() and the eviction above emptying a session's vector and
     * leaving the key -- grows by one entry per request for the life of the process, and
     * reserve_checkpoint walks every key on an allocation failure. */
    CHECK_EQ(f.pc.sessions_tracked(), 8ll);
}

TEST(a_sessions_index_entry_does_not_outlive_its_last_snapshot) {
    /* The other way a session empties: RETENTION rather than eviction. Drive one session past the
     * slot budget under keep-tip-only, which releases everything but the tip on the first squeeze,
     * then evict that tip from under it by way of other sessions. Nothing may be left behind. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    f.pc.set_policy(std::make_unique<KeepTipOnly>());

    auto prompt = tok_run(313, 16 * 32);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 32; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    for (int i = 1; i <= 5; ++i) {
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(7, i * 64, hs[i * 4 - 1], &s));
    }
    /* The squeeze happens on the FIFTH reservation: retention runs over {64,128,192,256} and
     * tip-only keeps 256, then 320 is allocated into the freed space. Two, not one. */
    CHECK_EQ((long long)f.pc.checkpoints(7).size(), 2ll);
    CHECK_EQ(f.pc.sessions_tracked(), 1ll);

    /* Four more sessions, one snapshot each: session 7's lone tip is the LRU and goes. */
    for (uint64_t sess = 20; sess < 24; ++sess) {
        BlockHash h = hs[9];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
    }
    CHECK(f.pc.checkpoints(7).empty());
    CHECK_EQ(f.pc.sessions_tracked(), 4ll);
}

/* ================================================================== SSD tier */

TEST(ssd_tier_round_trip_and_verify) {
    char dir[] = "/tmp/rad_sstier_XXXXXX";
    if (!mkdtemp(dir)) { CHECK(false); return; }

    SSDTierConfig c;
    c.dir = dir;
    c.block_payload = 2048;
    c.block_tokens = 16;
    c.ckpt_payload = 4096;
    c.n_block_slots = 4;
    c.n_ckpt_slots = 2;
    c.fingerprint = 0xfeedfaceull;
    c.persistent = false;
    /* Buffered on purpose: /tmp is usually tmpfs, which refuses O_DIRECT. Without this flag the
     * open is REFUSED rather than downgraded, which is the behaviour a server wants. */
    c.allow_buffered = true;

    SSDTier t;
    CHECK_OK(t.open(c));
    CHECK(t.enabled());
    CHECK(SSDTier::disclosure(dir).find("UNENCRYPTED") != std::string::npos);

    auto toks = tok_run(31337, 16);
    std::vector<uint8_t> payload(2048), back(2048, 0);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)(i * 7 + 3);
    BlockHash key{ 0x1122334455667788ull, 0x99aabbccddeeff00ull };

    CHECK_OK(t.put_block(key, toks.data(), 16, payload.data(), 2048));
    CHECK_OK(t.get_block(key, toks.data(), 16, back.data(), 2048));
    CHECK(back == payload);

    /* A key nobody wrote is a miss, not a fault. */
    BlockHash other{ 1, 2 };
    CHECK_EQ(t.get_block(other, toks.data(), 16, back.data(), 2048), RAD_E_NOTFOUND);

    /* The token ids are compared on the way back in, so a hash collision is a MISS and never a
     * wrong answer -- and the poisoned slot is dropped rather than failing every future probe. */
    auto wrong = tok_run(1, 16);
    CHECK_EQ(t.get_block(key, wrong.data(), 16, back.data(), 2048), RAD_E_NOTFOUND);
    CHECK_EQ(t.stats().verify_misses, 1ll);
    CHECK_EQ(t.get_block(key, toks.data(), 16, back.data(), 2048), RAD_E_NOTFOUND);

    /* Checkpoints go in the same store, keyed by the same chained hash. */
    std::vector<uint8_t> ck(4096, 0xa5), ckb(4096, 0);
    CHECK_OK(t.put_checkpoint(key, 2048, ck.data(), 4096));
    CHECK_OK(t.get_checkpoint(key, 2048, ckb.data(), 4096));
    CHECK(ckb == ck);
    /* The position is part of the identity: a checkpoint restored at the wrong offset is silent
     * numerical garbage. */
    CHECK_EQ(t.get_checkpoint(key, 4096, ckb.data(), 4096), RAD_E_NOTFOUND);

    t.close();
    ::rmdir(dir);
}

/* A PERSISTENT STORE COMES BACK WHOLE. The index is rebuilt from the slot headers at open, by
 * readers that each own a run of slots, so this writes more slots than there are readers and
 * leaves some free: every block written must be found again, the free slots must be what the next
 * writes take (no eviction until the store is full), and a store written under another model's
 * fingerprint must recover nothing. */
TEST(ssd_tier_persistent_store_recovers_on_reopen) {
    char dir[] = "/tmp/rad_sstier_XXXXXX";
    if (!mkdtemp(dir)) { CHECK(false); return; }

    SSDTierConfig c;
    c.dir = dir;
    c.block_payload = 4096;
    c.block_tokens = 4;
    c.ckpt_payload = 4096;
    c.n_block_slots = 203;
    c.n_ckpt_slots = 3;
    c.fingerprint = 0x5eedull;
    c.persistent = true;
    c.allow_buffered = true;

    const int64_t written = 150;
    auto key_of = [](int64_t i) {
        return BlockHash{ 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1), (uint64_t)i ^ 0xabcdefull };
    };
    std::vector<uint8_t> payload(4096), back(4096);
    {
        SSDTier t;
        CHECK_OK(t.open(c));
        for (int64_t i = 0; i < written; ++i) {
            auto toks = tok_run((uint32_t)(i + 7), 4);
            for (size_t b = 0; b < payload.size(); ++b) payload[b] = (uint8_t)(b * 13 + i);
            CHECK_OK(t.put_block(key_of(i), toks.data(), 4, payload.data(), 4096));
        }
        std::vector<uint8_t> ck(4096, 0x3c);
        CHECK_OK(t.put_checkpoint(key_of(0), 2048, ck.data(), 4096));
        t.close();
    }
    {
        SSDTier t;
        CHECK_OK(t.open(c));
        for (int64_t i = 0; i < written; ++i) {
            auto toks = tok_run((uint32_t)(i + 7), 4);
            for (size_t b = 0; b < payload.size(); ++b) payload[b] = (uint8_t)(b * 13 + i);
            std::fill(back.begin(), back.end(), 0);
            CHECK_OK(t.get_block(key_of(i), toks.data(), 4, back.data(), 4096));
            CHECK(back == payload);
        }
        std::vector<uint8_t> ckb(4096, 0);
        CHECK_OK(t.get_checkpoint(key_of(0), 2048, ckb.data(), 4096));
        CHECK(ckb == std::vector<uint8_t>(4096, 0x3c));

        /* The free slots are exactly the ones nobody wrote: filling them evicts nothing, and the
         * write after that has to. */
        auto toks = tok_run(99, 4);
        for (int64_t i = written; i < c.n_block_slots; ++i)
            CHECK_OK(t.put_block(key_of(i), toks.data(), 4, payload.data(), 4096));
        CHECK_EQ(t.stats().evictions, 0ll);
        CHECK_OK(t.put_block(key_of(c.n_block_slots), toks.data(), 4, payload.data(), 4096));
        CHECK_EQ(t.stats().evictions, 1ll);
        t.close();
    }
    {
        SSDTierConfig o = c;
        o.fingerprint = 0x5eedull + 1;
        SSDTier t;
        CHECK_OK(t.open(o));
        auto toks = tok_run(7, 4);
        CHECK_EQ(t.get_block(key_of(0), toks.data(), 4, back.data(), 4096), RAD_E_NOTFOUND);
        t.close();
    }
    ::unlink((std::string(dir) + "/blocks.bin").c_str());
    ::unlink((std::string(dir) + "/ckpts.bin").c_str());
    ::rmdir(dir);
}

TEST(ssd_token_bucket_is_a_shared_budget) {
    TokenBucket off;
    off.configure(0, 0);
    CHECK(!off.enabled());
    CHECK(off.try_take(1 << 30, SSDConsumer::PrefixCache));   /* off means off */

    TokenBucket b;
    b.configure(1000, 1000);
    CHECK(b.enabled());
    CHECK(b.try_take(600, SSDConsumer::PrefixCache));
    /* The weight tier draws on the SAME budget -- that is the entire point of one object. */
    CHECK(!b.try_take(600, SSDConsumer::WeightTier));
    CHECK_EQ(b.use(SSDConsumer::PrefixCache).bytes, 600ll);
    CHECK_EQ(b.use(SSDConsumer::WeightTier).bytes, 0ll);
    CHECK(b.report().find("PLACEHOLDER") != std::string::npos);
}

/* ================================================================== the KV loan
 *
 * WHAT THESE ARE HOLDING DOWN. The cache and the expert plane share one card, and the split
 * between them is made once at startup against a worst case nobody runs. So the cache lends the
 * expert slab the blocks it is not holding, in place -- which is only safe if a block is never lent
 * while anything holds it, and only USEFUL if the held blocks stay packed at the bottom of the pool
 * so that there is a top to lend.
 *
 * Both of those fail quietly. A block lent out from under a sequence is expert weights written
 * over its KV, which reads as fluent text with a different hash; a pool whose held blocks are
 * scattered simply never lends, and the only symptom is that the feature does nothing. So the
 * tests here are about the mark and the ordering. None of them needs virtual memory management:
 * a loan maps and unmaps nothing.
 */

struct Lender {
    Pools     pools;
    KVManager kv;
    Config    cfg = small_config();

    /* THE BUFFER IS SET BY THE CHUNK, because that is what it is derived from: the floor is one
     * step's worth of the group, and a step batches at most --max-num-batched-tokens tokens. A
     * test that wants a bigger buffer asks for a bigger chunk, which is the same lever the
     * running engine has and not a second one built for the test. */
    int build(int64_t chunk_tokens = 2048) {
        cfg.vram_kv_mib     = 256;
        cfg.max_tok         = chunk_tokens;
        cfg.checkpoint_slots = 0;
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        return kv.configure(hybrid_groups(/*with_linear=*/false), cfg, pools.kv());
    }
    int64_t bs() const { return kv.plan(G_FULL)->block_size; }
};

TEST(a_new_cache_lends_nothing_and_hands_out_its_whole_carve) {
    Lender e;
    CHECK_OK(e.build());
    /* THE FIRST REQUEST OF A RUN DOES NOT WAIT FOR A RECALL. Lending is the arbiter's decision,
     * made once the cache has said what it can spare, and not a starting state. */
    CHECK_EQ(e.kv.loan_tokens(), 0);
    CHECK(e.kv.lendable(G_FULL));
    CHECK_EQ(e.kv.live_blocks(G_FULL), e.kv.total_blocks(G_FULL));
    CHECK_EQ(e.kv.free_blocks(G_FULL), e.kv.total_blocks(G_FULL));
}

TEST(an_idle_cache_lends_all_but_its_buffer) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK(spare > 0);
    CHECK_OK(e.kv.set_loan(spare));
    CHECK_EQ(e.kv.loan_tokens(), spare);

    /* What it keeps is the buffer, not nothing: a cache that lent everything would recall on the
     * first request. */
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live > 0);
    CHECK(live < all);
    CHECK_EQ(all - live, spare / e.bs());
    CHECK_EQ(e.kv.free_blocks(G_FULL), live);
    CHECK_EQ(e.kv.held_blocks(G_FULL), 0);
    /* The cache's own figure shrinks by what it lent; the carve, which is the budget, does not. */
    CHECK_EQ(e.kv.pool_bytes(G_FULL), live * e.kv.plan(G_FULL)->bytes_per_block);
    CHECK_EQ(e.kv.pool_bytes_carved(G_FULL), all * e.kv.plan(G_FULL)->bytes_per_block);

    /* AND THE MARK IS WHAT THE FREE LIST OBEYS. A block above it is the slab's, and a restore
     * that names one is refused rather than written over a unit's weights. */
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));

    /* Giving the loan back is moving the mark: every id was in the list all along. */
    CHECK_OK(e.kv.set_loan(0));
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);
    CHECK_EQ(e.kv.free_blocks(G_FULL), all - 1);
}

TEST(a_block_something_holds_is_never_lent) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);

    /* Take every block, then give back all but the highest. The pool is now almost empty and the
     * one thing in it is at the top, which is the case the loan has to refuse. */
    std::vector<int32_t> got;
    int32_t b = -1;
    while (e.kv.reserve_block(G_FULL, &b) == RAD_OK) got.push_back(b);
    CHECK_EQ((int64_t)got.size(), all);
    int32_t top = 0;
    for (int32_t g : got) top = std::max(top, g);
    CHECK_EQ((int64_t)top, all - 1);
    for (int32_t g : got) if (g != top) e.kv.release_block(G_FULL, g);

    /* One block, at the top, and nothing can be lent. The alternative -- lending around it -- is
     * not available: the pool is layer-major and the boundary is a mark, not a set. */
    CHECK_EQ(e.kv.spare_tokens(), 0);
    CHECK_EQ(e.kv.set_loan(e.bs()), RAD_E_STATE);
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);            /* and a refusal moved nothing */
    CHECK_EQ(e.kv.loan_tokens(), 0);

    /* Let it go and the loan can follow. Everything that makes a block free -- a sequence ending,
     * an eviction, an idle-tier demotion -- reaches this the same way. */
    e.kv.release_block(G_FULL, top);
    CHECK(e.kv.spare_tokens() > 0);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    CHECK(e.kv.live_blocks(G_FULL) < all);
}

TEST(the_lowest_free_block_is_the_one_handed_out) {
    Lender e;
    CHECK_OK(e.build());
    int32_t a = -1, b = -1, c = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &a));
    CHECK_OK(e.kv.reserve_block(G_FULL, &b));
    CHECK_OK(e.kv.reserve_block(G_FULL, &c));
    CHECK(a < b && b < c);

    /* Freed in the WORST order for a stack: the highest last. A stack would hand back c, which
     * after a busy period is exactly the high block a loan wants -- so the highest held block
     * would never fall and nothing could ever be lent. */
    e.kv.release_block(G_FULL, a);
    e.kv.release_block(G_FULL, b);
    e.kv.release_block(G_FULL, c);
    int32_t again = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &again));
    CHECK_EQ(again, a);
}

/* THE LIVELOCK THIS DESIGN HAS TO NOT HAVE, and it is worth stating plainly because it is the one
 * way a lending cache can be worse than a fixed one rather than merely no better.
 *
 * A long prefill asks for its blocks in ONE call. A cache that could only take its loan back on a
 * timer would refuse that call; the scheduler would wait, or preempt something, and neither
 * produces the demand that would have brought the loan back -- because the demand it just refused
 * was the demand. So admission is decided against the whole carve, and the recall happens on the
 * path that needed it, through the owner of the slabs. */
TEST(a_request_past_the_mark_takes_the_loan_back) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK_OK(e.kv.set_loan(spare));
    const int64_t idle = e.kv.live_blocks(G_FULL);

    int calls = 0;
    int64_t asked = -1;
    e.kv.set_reclaim([&](int64_t t) { ++calls; asked = t; return (int)RAD_OK; });

    /* Longer than the cache is keeping, and well inside what it was carved for. */
    const int64_t tokens = (idle + all) / 2 * e.bs();
    CHECK(e.kv.can_ever_fit(tokens));
    CHECK_OK(e.kv.add_sequence(1));
    CHECK_OK(e.kv.ensure(1, tokens));
    CHECK_EQ(calls, 1);                                 /* one recall for the whole request */
    CHECK(asked >= 0 && asked < spare);
    CHECK_EQ(e.kv.loan_tokens(), asked);
    CHECK(e.kv.live_blocks(G_FULL) > idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), (tokens + e.bs() - 1) / e.bs());
    /* AND THE BUFFER CAME BACK WITH IT, so the next block does not recall again. */
    CHECK(e.kv.free_blocks(G_FULL) > 0);

    /* A REQUEST PAST THE CARVE IS STILL REFUSED. The budget is still the budget: lending decides
     * who uses the carve, never how much the operator said the cache may address. */
    CHECK_OK(e.kv.add_sequence(2));
    CHECK_EQ(e.kv.ensure(2, all * e.bs()), RAD_E_FULL);
}

/* A RECALL THE SLABS CANNOT MAKE IS A REQUEST THAT CANNOT BE SERVED, AND NOTHING MOVES. The
 * alternative -- taking the blocks back regardless -- hands a sequence memory an expert unit is
 * still being read out of. */
TEST(a_recall_the_slab_cannot_make_refuses_the_request) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    const int64_t loan = e.kv.loan_tokens();
    e.kv.set_reclaim([](int64_t) { return (int)RAD_E_DEVICE; });

    CHECK_OK(e.kv.add_sequence(1));
    CHECK(e.kv.ensure(1, (idle + all) / 2 * e.bs()) < 0);
    CHECK_EQ(e.kv.loan_tokens(), loan);
    CHECK_EQ(e.kv.live_blocks(G_FULL), idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), 0);
}

/* AND THE BUFFER IT KEEPS IS ONE STEP'S WORTH, which is the floor a chunked prefill needs so that
 * it recalls once a chunk rather than once a block. A bigger chunk is a bigger buffer, and the
 * relation is the one thing about the derivation an operator can still influence. */
TEST(the_derived_buffer_is_one_chunk_of_the_group) {
    Lender small, big;
    CHECK_OK(small.build(/*chunk_tokens=*/2048));
    CHECK_OK(big.build(/*chunk_tokens=*/65536));
    CHECK(big.kv.buffer_bytes() > small.kv.buffer_bytes());
    CHECK(big.kv.spare_tokens() < small.kv.spare_tokens());
}

/* A RESTORE CANNOT LAND ABOVE THE MARK. Under tensor parallelism rank 0 picks the block id and
 * every other rank takes the same one, so a rank whose loan is deeper than its peers' must refuse
 * rather than write over the weights its slab keeps there. */
TEST(a_named_block_above_the_mark_is_refused) {
    Lender e;
    CHECK_OK(e.build());
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live < e.kv.total_blocks(G_FULL));

    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));
    /* Taken out of the middle, and the free count says so even though the heap still names it. */
    CHECK_EQ(e.kv.free_blocks(G_FULL), live - 1);
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1), RAD_E_STATE);
}

/* A FOLLOWER LENDS WHAT THE LEADER LENDS. Block ids are shared across ranks and only rank 0 knows
 * what is held, so a follower's mark is the leader's -- in both directions. */
TEST(a_follower_takes_the_leaders_mark) {
    Lender lead, fol;
    CHECK_OK(lead.build());
    CHECK_OK(fol.build());
    CHECK_OK(lead.kv.set_loan(lead.kv.spare_tokens()));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.loan_tokens(), lead.kv.loan_tokens());
    CHECK_EQ(fol.kv.live_blocks(G_FULL), lead.kv.live_blocks(G_FULL));
    CHECK_OK(lead.kv.set_loan(0));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.live_blocks(G_FULL), fol.kv.total_blocks(G_FULL));
}

/* A CHECKPOINT SLOT COSTS THE CARD WHEN IT IS TAKEN, NOT WHEN IT IS CARVED. --checkpoint-slots is
 * how many snapshots the server MAY hold, not how much VRAM it holds regardless: a region backed
 * whole at startup would never be released. On a linear-attention model a slot is a sizeable
 * fraction of a gigabyte, so a server that had never written a snapshot would still pay the expert
 * plane's price for every one of them. */
/* A hybrid model whose recurrent state is DELIBERATELY LARGE -- at least two commit granules of
 * snapshot, whatever this driver's granularity is. That is the shape a real linear-attention
 * model has, and it is the only shape for which a checkpoint slot is managed elastically. The
 * second state dimension is solved for rather than written down, because a constant that happened
 * to clear a 2 MiB granule would quietly stop testing anything on a device with a larger one. */
static std::vector<KVGroupInfo> big_linear_groups(int64_t want_state_bytes) {
    std::vector<KVGroupInfo> g = hybrid_groups(/*with_linear=*/false);
    KVGroupInfo k;
    k.name = "gdn";
    k.decl.kind = RAD_KV_LINEAR;
    k.decl.dtype = RAD_F32;
    k.decl.n_head_kv = 4;
    k.decl.state_dim[0] = 256;
    k.layers = { 2, 3, 4, 5 };
    /* bytes = n_head_kv * d0 * d1 * sizeof(f32) * n_layers */
    const int64_t per_d1 = 4ll * 256 * 4 * (int64_t)k.layers.size();
    k.decl.state_dim[1] = (int32_t)std::max<int64_t>(16, (want_state_bytes + per_d1 - 1) / per_d1);
    g.push_back(k);
    return g;
}

struct BigCkpt {
    Pools     pools;
    KVManager kv;
    Config    cfg = small_config();

    int build(int64_t ckpt_slots = 4) {
        cfg.vram_kv_mib      = 1024;
        cfg.max_seqs         = 2;
        cfg.max_tok          = 256;
        cfg.checkpoint_slots = ckpt_slots;
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        return kv.configure(big_linear_groups(2 * Pool::granule()), cfg, pools.kv());
    }
};

TEST(a_checkpoint_slot_is_backed_when_it_is_handed_out) {
    if (!VMem::supported()) {
        std::fprintf(stderr, "  SKIP a_checkpoint_slot_is_backed_when_it_is_handed_out: no VMM\n");
        return;
    }
    BigCkpt f;
    CHECK_OK(f.build(/*ckpt_slots=*/4));
    REQUIRE(f.pools.kv().elastic());
    CHECK_EQ(f.kv.checkpoint_slots(), 4);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
    CHECK(f.kv.checkpoint_bytes() >= Pool::granule());

    const int64_t idle = f.pools.kv().committed();
    int32_t a = -1, b = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&a));
    const int64_t one = f.pools.kv().committed();
    CHECK(one > idle);                              /* taking one costs the card */
    CHECK_OK(f.kv.alloc_checkpoint(&b));
    CHECK(f.pools.kv().committed() > one);

    /* AND GIVING IT BACK RELEASES IT EXACTLY, which is what the granule-aligned stride buys: an
     * unaligned slot would round its release inward and leave a partial granule backed forever. */
    f.kv.free_checkpoint(b);
    CHECK_EQ(f.pools.kv().committed(), one);
    f.kv.free_checkpoint(a);
    CHECK_EQ(f.pools.kv().committed(), idle);

    /* The slot comes back at the same address and really backed: the pointer never moved, only
     * the pages under it. Asked of the pool rather than by writing to it -- an elastic range is
     * DEVICE memory even here, because it comes from the driver and not from the fixture's
     * allocators, so a host store to it is a fault and not a test. */
    int32_t again = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&again));
    void* p = f.kv.checkpoint_ptr(again);
    REQUIRE(p != nullptr);
    const int64_t off = (char*)p - (char*)f.pools.kv().base();
    CHECK(f.pools.kv().is_committed(off, f.kv.checkpoint_bytes()));
    f.kv.free_checkpoint(again);
    CHECK(!f.pools.kv().is_committed(off, f.kv.checkpoint_bytes()));
}

/* THE OTHER HALF OF THE SAME DECISION. A snapshot smaller than a granule cannot be committed on
 * its own without rounding the slot up to one, which on this fixture's 16 KiB state is 128x the
 * address space to manage a charge of 64 KiB. Those slots stay in the eagerly backed region --
 * so the whole point of the test is that handing one out moves NOTHING, and the slot works
 * anyway. Without this the engine would silently pick the mode and nothing would say which. */
TEST(a_snapshot_below_a_granule_is_not_managed) {
    if (!VMem::supported()) {
        std::fprintf(stderr, "  SKIP a_snapshot_below_a_granule_is_not_managed: no VMM\n");
        return;
    }
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    REQUIRE(f.kv.checkpoint_bytes() < Pool::granule());

    const int64_t idle = f.pools.kv().committed();
    int32_t a = -1, b = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&a));
    CHECK_OK(f.kv.alloc_checkpoint(&b));
    CHECK_EQ(f.pools.kv().committed(), idle);       /* nothing to back and nothing to reclaim */
    void* p = f.kv.checkpoint_ptr(a);
    REQUIRE(p != nullptr);
    const int64_t off = (char*)p - (char*)f.pools.kv().base();
    CHECK(f.pools.kv().is_committed(off, f.kv.checkpoint_bytes()));   /* backed all along */
    f.kv.free_checkpoint(a);
    f.kv.free_checkpoint(b);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
}

/* A SIMULATED CARD: `cap` bytes, of which the pools on it have committed what they report. The
 * floor tests cannot read the real device -- memory the previous test's fixtures released is still
 * coming back while this one reads, so the answer moves under them -- and two ranks here share
 * one device where the engine's are on two. A card per rank is what the engine has. */
struct SimCard {
    int64_t cap = 0;
    std::vector<const Pool*> pools;
    int64_t used() const {
        int64_t u = 0;
        for (const Pool* p : pools) u += p->committed();
        return u;
    }
    KVManager::FreeBytes reader() const {
        return [this](int64_t* free_now) -> int { *free_now = cap - used(); return RAD_OK; };
    }
    /* The floor that lets `extra` more slots of `stride` fit and not one more: half a slot of
     * slack either side of the boundary. */
    int64_t floor_for(int64_t extra, int64_t stride) const {
        return cap - used() - extra * stride - stride / 2;
    }
};

/* A SLOT IS BACKED ON EVERY RANK OR HANDED OUT ON NONE. If rank 0 backs a slot at allocation and
 * rank 1's card refuses it at the snapshot copy, the engine stops mid-serve. So the peer is asked
 * before anything is committed: its refusal is a full pool at allocation, which the prefix cache
 * answers, and the leader has spent nothing on the slot it could not hand out. */
TEST(a_checkpoint_slot_is_backed_on_every_rank_or_none) {
    if (!VMem::supported()) {
        std::fprintf(stderr, "  SKIP a_checkpoint_slot_is_backed_on_every_rank_or_none: no VMM\n");
        return;
    }
    BigCkpt lead, peer;
    CHECK_OK(lead.build(/*ckpt_slots=*/4));
    CHECK_OK(peer.build(/*ckpt_slots=*/4));
    lead.kv.set_checkpoint_peers({ &peer.kv });
    const int64_t stride = lead.kv.checkpoint_stride();
    REQUIRE(stride > 0);
    SimCard lcard, pcard;
    lcard.cap = pcard.cap = 64ll << 30;
    lcard.pools = { &lead.pools.kv() };
    pcard.pools = { &peer.pools.kv() };

    /* Room on the leader's card and none on the peer's: nothing is handed out. */
    lead.kv.set_checkpoint_floor(lcard.reader(), lcard.floor_for(4, stride));
    peer.kv.set_checkpoint_floor(pcard.reader(), pcard.floor_for(0, stride));
    const int64_t l0 = lead.pools.kv().committed(), p0 = peer.pools.kv().committed();
    int32_t s = -1;
    CHECK_EQ(lead.kv.alloc_checkpoint(&s), RAD_E_FULL);
    CHECK_EQ(lead.pools.kv().committed(), l0);       /* asked first, so nothing was spent */
    CHECK_EQ(peer.pools.kv().committed(), p0);
    CHECK_EQ(lead.kv.free_checkpoint_slots(), 4);

    /* Room for one slot on the peer's card: that one is backed on both, and the next one is
     * refused by the peer and costs the leader nothing. */
    peer.kv.set_checkpoint_floor(pcard.reader(), pcard.floor_for(1, stride));
    CHECK_OK(lead.kv.alloc_checkpoint(&s));
    CHECK_EQ(lead.pools.kv().committed(), l0 + stride);
    CHECK_EQ(peer.pools.kv().committed(), p0 + stride);
    int32_t t = -1;
    CHECK_EQ(lead.kv.alloc_checkpoint(&t), RAD_E_FULL);
    CHECK_EQ(lead.pools.kv().committed(), l0 + stride);
    CHECK_EQ(peer.pools.kv().committed(), p0 + stride);

    /* And a copy into the slot finds it backed on the peer without touching the card. */
    CHECK_OK(peer.kv.back_checkpoint(s));
    CHECK_EQ(peer.pools.kv().committed(), p0 + stride);
    lead.kv.free_checkpoint(s);
}

/* ON A DRIVER THAT KEEPS WHAT IT MAPS, EVERY SLOT IS BACKED AT BRINGUP, and the ones the card has
 * no room for above its floor are withdrawn -- never handed out, so serving never commits. */
TEST(back_free_checkpoints_withdraws_what_the_card_cannot_hold) {
    if (!VMem::supported()) {
        std::fprintf(stderr, "  SKIP back_free_checkpoints_withdraws_what_the_card_cannot_hold: "
                             "no VMM\n");
        return;
    }
    BigCkpt f;
    CHECK_OK(f.build(/*ckpt_slots=*/4));
    f.kv.set_release_frees_card(false);
    const int64_t stride = f.kv.checkpoint_stride();
    SimCard card;
    card.cap = 64ll << 30;
    card.pools = { &f.pools.kv() };
    f.kv.set_checkpoint_floor(card.reader(), card.floor_for(2, stride));
    const int64_t idle = f.pools.kv().committed();

    int64_t gone = -1;
    CHECK_OK(f.kv.back_free_checkpoints(&gone));
    CHECK_EQ(gone, 2);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 2);
    const int64_t backed = f.pools.kv().committed();
    CHECK_EQ(backed, idle + 2 * stride);

    /* Both survivors are handed out with no commit, and then the pool is full. */
    int32_t a = -1, b = -1, c = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&a));
    CHECK_OK(f.kv.alloc_checkpoint(&b));
    CHECK(a != b);
    CHECK_EQ(f.pools.kv().committed(), backed);
    CHECK_EQ(f.kv.alloc_checkpoint(&c), RAD_E_FULL);

    /* A freed slot keeps its pages on this driver and is the one handed out next. */
    f.kv.free_checkpoint(a);
    CHECK_OK(f.kv.alloc_checkpoint(&c));
    CHECK_EQ(c, a);
    CHECK_EQ(f.pools.kv().committed(), backed);
}

RAD_TEST_MAIN()

/* ================================================================== idle session tiers
 *
 * The state machine only, with no bytes moving: every test here drives IdleTiers directly and
 * reports jobs back as the engine would. That is the whole point of the object being separable --
 * a tier's use-after-free hazards live in this bookkeeping rather than in the transfers, and none
 * of them needs a GPU to exercise.
 */

static IdleTiers::Config tier_config(int64_t host_bytes, int64_t disk_bytes, int64_t entry) {
    IdleTiers::Config c;
    c.host_cap_bytes = host_bytes;
    c.disk_cap_bytes = disk_bytes;
    c.entry_bytes = entry;
    return c;
}
static BlockHash bh(uint64_t n) { return BlockHash{n, n * 0x9e3779b97f4a7c15ull + 1}; }
static const uint64_t SEC = 1000000000ull;

/* Every copy that is waiting, planned and reported the way the engine does once it has landed. */
static std::vector<TierJob> copy_now(IdleTiers& t, bool ok = true) {
    std::vector<TierJob> jobs;
    t.plan_writeback(1 << 20, &jobs);
    for (TierJob& j : jobs) t.report(j, ok);
    return jobs;
}

/* Every copy waiting for the disk, written and reported the way the IO thread does. */
static std::vector<TierJob> disk_now(IdleTiers& t, bool ok = true) {
    std::vector<TierJob> jobs;
    t.plan(1 << 20, &jobs);
    for (TierJob& j : jobs) t.report(j, ok);
    return jobs;
}

/* A chain's disk part read into host memory, planned and reported the way the IO thread and the
 * engine do; what a restore of it then finds. */
static std::vector<TierJob> fetch_now(IdleTiers& t, const std::vector<BlockHash>& chain,
                                      bool ok = true) {
    std::vector<TierJob> jobs;
    t.plan_fetch(chain, &jobs, 1);
    for (TierJob& j : jobs) { j.started = true; t.report(j, ok); }
    return jobs;
}

/* An entry's VRAM given up the way the cache gives it up, once nothing else is reading it. */
static bool drop_now(IdleTiers& t, const BlockHash& k) {
    if (!t.droppable(k)) return false;
    t.drop_commit(k);
    return true;
}
/* A snapshot's device slot given up the way the cache gives it up under slot pressure. */
static bool detach_now(IdleTiers& t, const BlockHash& k) {
    if (!t.snapshot_backed(k)) return false;
    t.snapshot_detached(k);
    return true;
}

/* A disk store in a fresh directory, for the tests whose subject is the leg to disk. A null disk
 * pointer turns that tier off, and a test would then pass having exercised half the transitions. */
struct TmpDisk {
    SSDTier disk;
    int open(int64_t payload, int64_t ck_payload, int64_t slots, int64_t ck_slots) {
        char dir[] = "/tmp/rad_kvdisk_XXXXXX";
        if (!mkdtemp(dir)) return RAD_E_IO;
        SSDTierConfig sc;
        sc.dir = dir;
        sc.block_payload = payload;
        sc.block_tokens = 16;
        sc.ckpt_payload = ck_payload;
        sc.n_block_slots = slots;
        sc.n_ckpt_slots = ck_slots;
        sc.allow_buffered = true;       /* /tmp is tmpfs and refuses O_DIRECT */
        return disk.open(sc);
    }
};

TEST(a_published_entry_is_copied_at_once_and_keeps_its_vram) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    CHECK(t.host_on());

    t.note_used(bh(1), {10}, 1 * SEC);
    CHECK(t.writeback_waiting());

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].kind == TierJob::Kind::WriteBack);
    CHECK(jobs[0].key == bh(1));
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 10);
    CHECK(jobs[0].host_slot >= 0);

    /* PENDING EXCLUDES IT FROM A SECOND PLAN, and from a drop: a copy still out has not landed,
     * so the VRAM copy is the only complete one there is. */
    std::vector<TierJob> again;
    t.plan_writeback(16, &again);
    CHECK_EQ(again.size(), 0u);
    CHECK(!t.droppable(bh(1)));

    t.report(jobs[0], true);
    /* A COPY IS NOT A MOVE. The entry is still in VRAM, where a return finds it for free; what has
     * changed is that giving the VRAM up now costs nothing. */
    CHECK(t.tier_of(bh(1)) == KVTier::Device);
    CHECK(t.droppable(bh(1)));
    CHECK_EQ(t.stats().to_host, 1);
    CHECK_EQ(t.stats().dropped, 0);
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    REQUIRE_EQ(clean.size(), 1u);
    CHECK(clean[0] == bh(1));
}

TEST(giving_up_a_clean_entry_s_vram_moves_no_bytes) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    t.note_used(bh(1), {10}, 1 * SEC);
    copy_now(t);

    CHECK(drop_now(t, bh(1)));
    CHECK(t.tier_of(bh(1)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(1)));
    CHECK_EQ(t.stats().dropped, 1);
    CHECK_EQ(t.stats().to_host, 1);               /* the one copy, made before */
    CHECK_EQ(t.arena().used_slots(), 1);          /* the copy IS the entry now */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);                    /* and nothing is left to copy */
}

TEST(a_failed_copy_leaves_the_entry_in_vram_and_tries_again) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1024, 0, 1024), nullptr));   /* exactly one slot */
    t.note_used(bh(1), {10}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(t.arena().full());

    t.report(jobs[0], false);
    CHECK(t.tier_of(bh(1)) == KVTier::Device);    /* the pool still holds the blocks */
    CHECK(!t.droppable(bh(1)));            /* and nothing may let them go */
    CHECK(!t.arena().full());                     /* the slot went back */
    CHECK_EQ(t.stats().failures, 1);
    CHECK_EQ(t.stats().to_host, 0);

    /* A copy that failed is not a decision, it is an attempt. */
    std::vector<TierJob> retry;
    t.plan_writeback(16, &retry);
    CHECK_EQ(retry.size(), 1u);
}

/* ONE COPY A RUN is what makes a long conversation affordable at a 4-token block, so entries
 * planned together have to sit together. */
TEST(a_batch_of_copies_takes_consecutive_arena_slots) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 0, 1024), nullptr));
    for (uint64_t i = 1; i <= 8; ++i) t.note_used(bh(i), {(int32_t)(10 + i)}, i * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 8u);
    for (size_t i = 0; i < jobs.size(); ++i)
        CHECK_EQ(jobs[i].host_slot, jobs[0].host_slot + (int32_t)i);
    /* And oldest first: the conversation that finished first is the first copied. */
    CHECK(jobs[0].key == bh(1));
    CHECK(jobs[7].key == bh(8));
}

TEST(host_arena_runs_stop_at_a_used_slot_and_wrap) {
    HostArena a;
    CHECK_OK(a.open(1024, 8 * 1024));
    REQUIRE_EQ(a.capacity(), 8);
    int32_t got = 0;
    CHECK_EQ(a.alloc_run(3, &got), 0);
    CHECK_EQ(got, 3);
    CHECK_EQ(a.alloc_run(3, &got), 3);
    CHECK_EQ(got, 3);
    a.free(1);
    /* The run ends where the arena ends; the rest is asked for again. */
    CHECK_EQ(a.alloc_run(4, &got), 6);
    CHECK_EQ(got, 2);
    /* And the cursor wraps to the hole it left behind. */
    CHECK_EQ(a.alloc_run(4, &got), 1);
    CHECK_EQ(got, 1);
    CHECK(a.full());
    CHECK_EQ(a.alloc_run(1, &got), -1);
    CHECK_EQ(got, 0);
    /* A double free is refused rather than counted: the used count is what the view reports. */
    a.free(1);
    a.free(1);
    CHECK_EQ(a.used_slots(), 7);
}

TEST(a_full_arena_holds_the_copy_back_and_makes_room_from_the_oldest_host_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 0, 1024), nullptr));   /* two slots, no disk */
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));
    drop_now(t, bh(2));

    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);             /* nowhere to put it */
    CHECK(t.starved());
    /* Still queued, not forgotten -- but not planned again until a slot is freed: asking again
     * before then pops the queue, finds no slot and puts it back. */
    CHECK(!t.writeback_waiting());
    CHECK(t.tier_of(bh(3)) == KVTier::Device);

    /* The maintenance plan frees exactly what was refused, oldest first -- onto the floor, with
     * no disk tier, and the cache is told so it stops offering the entry. */
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].kind == TierJob::Kind::DropHost);
    CHECK(room[0].key == bh(1));
    t.report(room[0], true);
    CHECK_EQ(t.stats().host_drops, 1);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(1));
    CHECK(!t.starved());
    CHECK(t.writeback_waiting());

    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK(t.droppable(bh(3)));
}

TEST(forgetting_an_entry_mid_copy_does_not_erase_it_under_the_engine) {
    /* THE LIFETIME BUG THIS CLASS IS FOR. An entry evicted from the prefix cache while the engine
     * is still copying it must not have its record erased: the job the engine holds names it, and
     * report() dereferences it afterwards. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    t.note_used(bh(7), {70}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(!t.doomed(bh(7)));

    t.forget(bh(7));                              /* evicted while pending */
    CHECK(t.doomed(bh(7)));

    t.report(jobs[0], true);                      /* the engine finishes anyway */
    CHECK(t.doomed(bh(7)));                       /* gone altogether: unknown reads as doomed */
    CHECK(t.tier_of(bh(7)) == KVTier::Device);    /* and as the device, which is "not ours" */
    CHECK_EQ(t.arena().used_slots(), 0);          /* its slot came back */
    CHECK(!t.droppable(bh(7)));
}

TEST(a_restore_keeps_the_host_copy_so_the_entry_lands_clean) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    t.note_used(bh(5), {50}, 0);
    copy_now(t);
    drop_now(t, bh(5));

    TierJob up;
    CHECK(t.plan_promote(bh(5), &up));
    CHECK(up.kind == TierJob::Kind::ToDevice);
    CHECK(up.host_slot >= 0);
    /* A promotion already outstanding is not handed out twice. */
    TierJob dup;
    CHECK(!t.plan_promote(bh(5), &dup));

    up.blocks = {99};                             /* the engine allocated a different block */
    up.started = true;
    t.report(up, true, 100 * SEC);
    CHECK(t.tier_of(bh(5)) == KVTier::Device);
    CHECK_EQ(t.stats().promoted, 1);
    CHECK_EQ(t.stats().host_hits, 1);
    /* THE COPY STAYED, so the VRAM can go again for nothing and there is nothing to copy. */
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(t.droppable(bh(5)));
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    CHECK(std::find(clean.begin(), clean.end(), bh(5)) != clean.end());
}

TEST(only_what_a_turn_added_is_copied_after_a_restore) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 0, 1024), nullptr));
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 1 * SEC);
    copy_now(t);
    for (uint64_t i = 1; i <= 3; ++i) {
        drop_now(t, bh(i));
        TierJob up;
        REQUIRE(t.plan_promote(bh(i), &up));
        up.blocks = {(int32_t)(20 + i)};
        up.started = true;
        t.report(up, true, 10 * SEC);
    }
    /* The next turn publishes the whole chain again, two blocks longer. */
    for (uint64_t i = 1; i <= 5; ++i) t.note_used(bh(i), {(int32_t)(20 + i)}, 20 * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 2u);
    CHECK(jobs[0].key == bh(4));
    CHECK(jobs[1].key == bh(5));
}

TEST(a_failed_restore_that_started_drops_the_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    up.blocks = {31};
    up.started = true;
    t.report(up, false);
    /* Gone, not left on the host tier: the scatter may have written some layers and not others,
     * so the blocks no longer hold a function of the tokens and a miss is the only safe answer --
     * and the cache has to hear it, or it would serve the ids it last had. */
    CHECK(t.tier_of(bh(3)) == KVTier::Device);
    CHECK(!t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 0);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(3));
}

TEST(a_restore_that_never_started_leaves_the_entry_where_it_was) {
    /* THE POOL RAN OUT BEFORE THIS ENTRY. Nothing about it was touched, so its host copy is exactly
     * what it was and dropping it would throw away a conversation for nothing. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    t.report(up, false);
    CHECK(t.tier_of(bh(3)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 1);
    TierJob again;
    CHECK(t.plan_promote(bh(3), &again));
}

TEST(a_recomputed_entry_gives_up_its_tier_copy_and_is_copied_afresh) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 0, 1024), nullptr));
    t.note_used(bh(2), {20}, 1 * SEC);
    copy_now(t);
    drop_now(t, bh(2));
    CHECK_EQ(t.arena().used_slots(), 1);

    t.rebound(bh(2), {77}, 10 * SEC);
    CHECK(t.tier_of(bh(2)) == KVTier::Device);
    CHECK_EQ(t.arena().used_slots(), 0);
    CHECK(!t.droppable(bh(2)));            /* the new blocks have no copy yet */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 77);
}

TEST(the_session_view_reports_where_a_conversation_actually_is) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 0, 1024), nullptr));
    t.set_model("q38-flashnext");

    std::vector<BlockHash> chain = {bh(1), bh(2), bh(3)};
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(42, chain, 48, 0);

    std::vector<IdleTiers::SessionInfo> s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].id, 42u);
    CHECK(s[0].model == "q38-flashnext");
    CHECK_EQ(s[0].n_tokens, 48);
    CHECK_EQ(s[0].bytes, 3 * 1024);
    CHECK_EQ(s[0].on_device, 3);

    /* Two of the three fit in the arena, so the session straddles two tiers and the view says so
     * rather than picking one. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].on_host, 2);
    CHECK_EQ(s[0].on_device, 1);
}

/* WHAT A RETURN REUSES IS THE DEEPEST SNAPSHOT, not the blocks: on a hybrid model the scheduler
 * clamps a hit to the last linear checkpoint it can reach. And what is off the card is counted
 * once, however many conversations share it. */
TEST(the_session_view_says_what_a_return_reuses_and_what_is_off_the_card) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 0, 1024), nullptr));

    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    /* A second conversation past the same first two blocks. */
    t.note_used(bh(9), {9}, 100 * SEC);
    t.note_session(2, {bh(1), bh(2), bh(9)}, 48, 100 * SEC);

    IdleTiers::Occupancy occ;
    std::vector<IdleTiers::SessionInfo> s = t.sessions(&occ);
    CHECK_EQ(s.size(), 2u);
    CHECK_EQ(s[0].resume_tokens, 0);             /* no snapshot: a return reuses nothing */
    CHECK_EQ(occ.host_bytes, 0);
    CHECK_EQ(occ.host_cap_bytes, 2048);
    CHECK_EQ(occ.disk_cap_bytes, 0);             /* off, so no capacity to hold it against */

    /* The shared blocks are copied first -- they were published first -- and the arena holds
     * two; each is one slot however many rows name it. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions(&occ);
    CHECK_EQ(occ.host_bytes, t.arena().used_slots() * 1024);
    CHECK_EQ(t.arena().used_slots(), 2);
    int64_t named = 0;
    for (const auto& r : s) named += r.on_host;
    CHECK(named > t.arena().used_slots());

    /* Snapshots on the chains: the deepest one a chain reaches is where its return resumes, and a
     * snapshot on the shared prefix is reachable from both. */
    IdleTiers u;
    CHECK_OK(u.configure(tier_config(0, 0, 1024), nullptr));
    for (uint64_t i : {1, 2, 3, 9}) u.note_used(bh(i), {(int32_t)i}, 0);
    u.note_snapshot(bh(1), 0, 16);
    u.note_snapshot(bh(9), 1, 48);
    u.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    u.note_session(2, {bh(1), bh(2), bh(9)}, 48, 1 * SEC);
    s = u.sessions();
    CHECK_EQ(s.size(), 2u);
    for (const auto& r : s) {
        if (r.id == 2) CHECK_EQ(r.resume_tokens, 48);
        if (r.id == 1) CHECK_EQ(r.resume_tokens, 16);
    }
}

TEST(tiers_stay_off_until_a_size_is_given) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0, 1024), nullptr));
    CHECK(!t.host_on());
    CHECK(!t.any_on());
    t.note_used(bh(1), {10}, 0);
    CHECK(!t.writeback_waiting());
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    t.plan(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);
    /* AND NO PER-BLOCK STATE, which is the cost that would otherwise be paid by every operator
     * who never asked for a tier: a record per cached entry is a map proportional to the cache.
     * The SESSION record is a different size of thing -- one row a conversation -- and is kept
     * either way; the test below is the one that pins that. */
    CHECK_EQ(t.sessions().size(), 0u);
}

/* ================================================================== the linear half
 *
 * On a hybrid model an entry's attention blocks are only part of a session. The rest is a
 * recurrent-state SNAPSHOT in a checkpoint slot, of which there are --checkpoint-slots for the
 * whole server, and a tier that copied the blocks and left the snapshot behind would free the
 * plentiful resource and hold the scarce one -- while the session that came back would replay
 * every linear layer from token zero and report a cache hit doing it.
 *
 * Still no bytes: every test drives the policy and reports jobs back as the engine would.
 */

static IdleTiers::Config ck_config(int64_t host_bytes, int64_t disk_bytes, int64_t entry,
                                   int64_t ck_bytes, int64_t ck_cap, int64_t ck_disk_slots) {
    IdleTiers::Config c = tier_config(host_bytes, disk_bytes, entry);
    c.ckpt_bytes      = ck_bytes;
    c.ckpt_cap_bytes  = ck_cap;
    c.ckpt_disk_slots = ck_disk_slots;
    return c;
}

TEST(a_snapshot_is_copied_with_the_blocks_it_is_keyed_by) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 1 << 20, 1024, 2048, 8192, 4), &d.disk));
    CHECK(t.ckpt_on());
    CHECK(t.disk_on());

    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), /*dev_slot=*/7, /*pos=*/2048);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    /* ONE JOB, BOTH HALVES. They are stored in different places and moved by different copies,
     * but they are decided together. */
    CHECK_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].ck_dev, 7);
    CHECK(jobs[0].ck_host >= 0);
    CHECK_EQ(jobs[0].ck_pos, 2048ll);
    CHECK(t.snapshot_reading(bh(1)));
    t.report(jobs[0], true);
    CHECK(!t.snapshot_reading(bh(1)));
    CHECK_EQ(t.stats().ck_to_host, 1ll);

    /* THE BLOCKS GO AND THE SNAPSHOT STAYS in its device slot, which nothing else could use;
     * the copy is what lets the checkpoint pool take the slot later without losing the state. */
    CHECK(drop_now(t, bh(1)));
    CHECK(t.resident_off_device(bh(1)));
    CHECK(t.snapshot_backed(bh(1)));
    CHECK(detach_now(t, bh(1)));                  /* the pool wanted the slot */

    /* ...and on to disk, both halves in one write, as COPIES: the host slots stay. */
    CHECK(t.disk_waiting());
    jobs.clear();
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].kind == TierJob::Kind::ToDisk);
    CHECK(jobs[0].host_slot >= 0);
    CHECK(jobs[0].ck_host >= 0);
    t.report(jobs[0], true);
    CHECK_EQ(t.stats().to_disk, 1ll);
    CHECK_EQ(t.stats().ck_to_disk, 1ll);
    CHECK_EQ(t.arena().used_slots(), 1ll);
    CHECK_EQ(t.ckpt_arena().used_slots(), 1ll);
    CHECK(!t.disk_waiting());

    /* The arena wants the slot: it goes for nothing, and the entry is a disk entry. */
    CHECK_EQ(t.make_room(1), 1ll);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Disk);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK_EQ(t.stats().host_freed, 1ll);

    /* ...and back: read into a host slot first, then promoted out of it with the device slot the
     * engine allocated -- and the entry keeps both host copies, clean. */
    TierJob up;
    CHECK(!t.plan_promote(bh(1), &up));           /* not out of disk: a restore reads host memory */
    const std::vector<TierJob> rd = fetch_now(t, {bh(1)});
    REQUIRE_EQ(rd.size(), 1u);
    CHECK(rd[0].kind == TierJob::Kind::ToHost);
    CHECK_EQ(rd[0].ck_host, -1);                  /* the snapshot's host copy never left */
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Host);
    CHECK(t.plan_promote(bh(1), &up));
    CHECK(up.ck_want);
    CHECK_EQ(up.host_slot, rd[0].host_slot);
    CHECK(up.ck_host >= 0);
    up.blocks = {11};
    up.ck_dev = 3;
    up.started = true;
    t.report(up, true);
    CHECK_EQ(t.stats().ck_promoted, 1ll);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);
    CHECK_EQ(t.arena().used_slots(), 1ll);
    CHECK_EQ(t.ckpt_arena().used_slots(), 1ll);
    CHECK(t.droppable(bh(1)));
    CHECK(t.snapshot_backed(bh(1)));
    /* And the disk copies survived the promotion: nothing is written twice. */
    CHECK(!t.disk_waiting());
}

TEST(a_snapshot_that_cannot_be_placed_does_not_hold_the_blocks_in_vram) {
    /* THE BLOCKS ARE WORTH MORE. They are most of the bytes and all of the attention, and a
     * snapshot with nowhere to go is still sitting in its device slot -- correct, and reachable
     * again the moment the blocks come back. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 0, 0), nullptr));
    CHECK(t.host_on());
    CHECK(!t.ckpt_on());                          /* no room was given for snapshots */

    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    std::vector<TierJob> jobs = copy_now(t);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK_EQ(jobs[0].ck_host, -1);                /* nothing carried */
    CHECK_EQ(t.stats().to_host, 1ll);
    CHECK_EQ(t.stats().ck_to_host, 0ll);

    CHECK(drop_now(t, bh(1)));
    CHECK(!t.snapshot_backed(bh(1)));             /* the device slot is the only home it has */
    /* And the cache was NOT told to drop it: the snapshot never moved and never became wrong. */
    CHECK_EQ(t.stats().ck_drops, 0ll);
}

TEST(the_snapshot_store_evicts_one_of_its_own_rather_than_starving) {
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 2048, 0), nullptr));
    CHECK_EQ(t.ckpt_arena().capacity(), 1ll);     /* room for exactly one */

    std::vector<BlockHash> lost;
    t.set_snapshot_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    drop_now(t, bh(1));
    detach_now(t, bh(1));                         /* its only copy of the state is on the host */

    t.note_used(bh(2), {11}, 50 * SEC);
    t.note_snapshot(bh(2), 8, 2048);
    std::vector<TierJob> jobs = copy_now(t);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].ck_host >= 0);                  /* it got the slot the first one held */

    /* THE FIRST SNAPSHOT IS GONE AND THE CACHE WAS TOLD. An index entry left behind would be a
     * checkpoint the cache goes on offering whose bytes nobody holds. */
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(1));
    CHECK_EQ(t.stats().ck_drops, 1ll);
    /* The first entry's BLOCKS are untouched: losing half an entry is not losing it. */
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Host);
}

/* A BACKUP EVICTED IS NOT COPIED AGAIN. Two snapshots, both still in their device slots, and room
 * for one copy: each pass that re-queued the evicted one would copy it back and evict the other --
 * a hundred megabytes a rank each way, forever, enough to saturate the link. */
TEST(two_snapshots_and_one_slot_settle_instead_of_trading_places) {
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 2048, 0), nullptr));
    REQUIRE_EQ(t.ckpt_arena().capacity(), 1ll);
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    drop_now(t, bh(1));                  /* on the host list, its snapshot still on the device */
    t.note_used(bh(2), {11}, 2 * SEC);
    t.note_snapshot(bh(2), 8, 4096);
    copy_now(t);                         /* takes the one slot from bh(1)'s backup */
    int copies = 0;
    for (int pass = 0; pass < 8; ++pass) copies += (int)copy_now(t).size();
    CHECK_EQ(copies, 0);
    CHECK_EQ(t.stats().ck_to_host, 2ll);
    CHECK_EQ(t.stats().ck_drops, 0ll);   /* neither was lost: both are in their device slots */
    /* A NEW COMMIT at the key is new state, and wants a copy again. */
    t.note_snapshot(bh(1), 9, 2048);
    CHECK(t.writeback_waiting());
}

TEST(the_snapshot_evicted_is_the_earliest_position_not_the_first_in_the_list) {
    /* A SESSION'S RECORDS ALL GO IDLE AT ONCE and reach the host list in whatever order they were
     * dropped, so list order is effectively arbitrary within a session. Taking the tip while
     * keeping a snapshot from earlier in the same transcript is the worst available outcome: the
     * tip is the one position an appending conversation asks for. Position decides. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 4096, 0), nullptr));
    CHECK_EQ(t.ckpt_arena().capacity(), 2ll);

    std::vector<BlockHash> lost;
    t.set_snapshot_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    /* The TIP reaches the list first -- exactly the case an oldest-first rule gets wrong. */
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, /*pos=*/8192);
    t.note_used(bh(2), {11}, 1 * SEC);
    t.note_snapshot(bh(2), 8, /*pos=*/2048);
    CHECK_EQ(copy_now(t).size(), 2u);
    for (uint64_t k : { 1, 2 }) { drop_now(t, bh(k)); detach_now(t, bh(k)); }
    CHECK_EQ(t.ckpt_arena().used_slots(), 2ll);

    /* A third wants a slot and there is none. */
    t.note_used(bh(3), {12}, 50 * SEC);
    t.note_snapshot(bh(3), 9, 2048);
    std::vector<TierJob> jobs = copy_now(t);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].ck_host >= 0);

    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(2));      /* position 2048, not the tip at 8192 */
}

TEST(a_snapshot_the_cache_retires_gives_its_tier_storage_back) {
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    CHECK_EQ(t.ckpt_arena().used_slots(), 1ll);

    /* The retention policy dropped it, or a reservation at the same prefix replaced it. Either
     * way the cache is the owner of that decision and the bytes come back here. Nothing was
     * reading the slot, so it is the cache's to free. */
    CHECK(!t.release_snapshot(bh(1)));
    CHECK_EQ(t.ckpt_arena().used_slots(), 0ll);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);   /* the blocks are still stored */
    CHECK(!t.snapshot_backed(bh(1)));
}

TEST(the_cache_retiring_an_on_device_snapshot_unregisters_its_slot) {
    /* THE SLOT NUMBER IS THE HAZARD. note_snapshot records a snapshot's DEVICE slot at commit,
     * because a copy has to know what to read. If the retention policy then frees that slot
     * without saying so, this record still names it -- and the pool has since handed it to another
     * session. The next copy reads THAT session's recurrent state into the tier and restores it
     * as this one's. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), /*dev_slot=*/7, 2048);

    t.release_snapshot(bh(1));          /* PrefixCache::retire_ckpt */

    std::vector<TierJob> jobs = copy_now(t);
    REQUIRE_EQ(jobs.size(), 1u);        /* the blocks are still copied */
    CHECK_EQ(jobs[0].ck_dev, -1);       /* and nothing names slot 7 any more */
    CHECK_EQ(jobs[0].ck_host, -1);
    CHECK_EQ(t.stats().ck_to_host, 0ll);
    CHECK_EQ(t.ckpt_arena().used_slots(), 0ll);
}

TEST(a_snapshot_a_copy_is_reading_is_freed_only_after_the_copy_reports) {
    /* THE SLOT MAY BE DECOMMITTED WHEN IT IS FREED, and an asynchronous copy out of an unmapped
     * page is a device fault. So a release that lands while the copy is out hands the slot to the
     * tiers, and they give it back once the copy has reported. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(t.snapshot_reading(bh(1)));

    CHECK(t.release_snapshot(bh(1)));             /* the tiers took the slot */
    std::vector<int32_t> frees;
    t.take_checkpoint_frees(&frees);
    CHECK(frees.empty());                         /* not yet: the copy is still out */

    t.report(jobs[0], true);
    t.take_checkpoint_frees(&frees);
    REQUIRE_EQ(frees.size(), 1u);
    CHECK_EQ(frees[0], 7);
    /* The bytes that landed describe a snapshot nobody holds, so they are not kept. */
    CHECK_EQ(t.stats().ck_to_host, 0ll);
    CHECK_EQ(t.ckpt_arena().used_slots(), 0ll);
    CHECK_EQ(t.stats().to_host, 1ll);             /* the blocks were fine */
}

TEST(a_new_commit_at_a_key_replaces_the_host_copy_of_its_snapshot) {
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    CHECK_EQ(t.ckpt_arena().used_slots(), 1ll);

    t.note_snapshot(bh(1), 9, 2048);              /* recomputed and saved into a fresh slot */
    CHECK_EQ(t.ckpt_arena().used_slots(), 0ll);
    CHECK(t.writeback_waiting());
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].blocks.empty());                /* the blocks' copy is still good */
    CHECK_EQ(jobs[0].ck_dev, 9);
}

TEST(a_record_that_only_ever_held_a_snapshot_is_forgotten_with_it) {
    /* note_snapshot runs at COMMIT, which is mid-request; note_used runs when the chain is
     * published, which is when the request closes. A request that never publishes -- a rebased
     * block table, a cancel before the first whole block -- would otherwise leave a record with
     * no blocks, no storage and no way to be reached again, one per checkpoint, forever. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    t.note_snapshot(bh(1), 7, 2048);
    /* NOT COPIED UNTIL ITS CHAIN IS PUBLISHED, so a request's checkpoints are planned together
     * and the deepest goes first -- and a request that never publishes costs no copy at all. */
    CHECK_EQ(copy_now(t).size(), 0u);
    CHECK(!t.snapshot_backed(bh(1)));

    t.release_snapshot(bh(1));
    CHECK_EQ(t.ckpt_arena().used_slots(), 0ll);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);   /* which is what "no record" answers */
    CHECK(!t.writeback_waiting());

    /* Published after the commit: the snapshot goes with the blocks. */
    t.note_snapshot(bh(2), 8, 2048);
    t.note_used(bh(2), {12}, 1 * SEC);
    std::vector<TierJob> jobs = copy_now(t);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK_EQ(jobs[0].blocks.size(), 1u);
    CHECK(jobs[0].ck_host >= 0);
    CHECK(t.snapshot_backed(bh(2)));
}

TEST(a_restore_that_cannot_place_the_state_keeps_its_host_copy) {
    /* NO DEVICE SLOT WAS FREE for the snapshot. This turn replays its linear layers, but the host
     * copy is intact, and dropping it would make one full slot pool cost the session its state
     * for good. It comes back with the blocks next time. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 8192, 0), nullptr));
    std::vector<BlockHash> lost;
    t.set_snapshot_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {10}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    drop_now(t, bh(1));
    detach_now(t, bh(1));

    TierJob up;
    CHECK(t.plan_promote(bh(1), &up));
    CHECK(up.ck_want);
    up.blocks = {11};
    up.ck_dev = -1;
    up.started = true;
    t.report(up, true);

    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);   /* the attention hit survives */
    CHECK_EQ(t.stats().promoted, 1ll);
    CHECK_EQ(t.stats().ck_promoted, 0ll);
    CHECK(lost.empty());
    CHECK_EQ(t.ckpt_arena().used_slots(), 1ll);

    /* Out again, and back again -- and this time the state is asked for. */
    CHECK(drop_now(t, bh(1)));
    TierJob next;
    CHECK(t.plan_promote(bh(1), &next));
    CHECK(next.ck_want);
    CHECK(next.ck_host >= 0);
}

TEST(a_snapshot_left_behind_by_the_budget_is_copied_on_a_later_pass) {
    /* A snapshot is a hundred megabytes on a production hybrid, so only a couple are copied a
     * pass. The blocks do not wait for them -- and the snapshots are not forgotten either: a
     * record whose state has no copy stays queued until it has one. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 1 << 16, 0), nullptr));
    for (int i = 1; i <= 5; ++i) {
        t.note_used(bh((uint64_t)i), {10 + i}, 1 * SEC);
        t.note_snapshot(bh((uint64_t)i), 7 + i, 2048 * i);
    }
    std::vector<TierJob> jobs = copy_now(t);
    CHECK_EQ(jobs.size(), 5u);
    int with_ck = 0;
    for (const TierJob& j : jobs) with_ck += j.ck_host >= 0;
    CHECK_EQ(with_ck, 2);

    jobs = copy_now(t);
    CHECK_EQ(jobs.size(), 2u);
    for (const TierJob& j : jobs) { CHECK(j.blocks.empty()); CHECK(j.ck_host >= 0); }
    jobs = copy_now(t);
    CHECK_EQ(jobs.size(), 1u);
    CHECK_EQ(copy_now(t).size(), 0u);

    CHECK_EQ(t.stats().to_host, 5ll);
    CHECK_EQ(t.stats().ck_to_host, 5ll);
    CHECK_EQ(t.stats().ck_starved, 0ll);
}

TEST(the_snapshot_arena_goes_to_a_session_s_tip_before_its_older_checkpoints) {
    /* A session commits its checkpoints in order, so its tip is queued last -- and the tip is the
     * one its next turn resumes from. One slot, two snapshots: the deeper one gets it. */
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 0, 1024, 2048, 2048, 0), nullptr));
    REQUIRE_EQ(t.ckpt_arena().capacity(), 1);
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    t.note_snapshot(bh(2), 8, 4096);
    const std::vector<TierJob> jobs = copy_now(t);
    CHECK_EQ(jobs.size(), 2u);
    for (const TierJob& j : jobs) CHECK_EQ(j.ck_host >= 0, j.key == bh(2));
    CHECK_EQ(t.stats().ck_to_host, 1ll);
    CHECK(t.snapshot_backed(bh(2)));
    CHECK(!t.snapshot_backed(bh(1)));
}

TEST(a_snapshot_copy_can_take_the_slot_of_one_a_session_in_vram_holds) {
    /* A snapshot whose session stays in VRAM must not keep the arena for as long as it stays: a
     * search of only the off-device lists for a slot to take never finds it, because a live
     * session is on none. Once the disk holds that copy, giving its slot up costs nothing. */
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 16, 16));
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 1 << 20, 1024, 2048, 2048, 16), &d.disk));
    REQUIRE_EQ(t.ckpt_arena().capacity(), 1);
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    copy_now(t);
    disk_now(t);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);

    t.note_used(bh(2), {12}, 2 * SEC);
    t.note_snapshot(bh(2), 8, 4096);
    copy_now(t);
    CHECK_EQ(t.stats().ck_to_host, 2ll);
    CHECK_EQ(t.stats().ck_starved, 0ll);
    CHECK(t.snapshot_backed(bh(1)));              /* through its disk copy */
    CHECK(t.snapshot_backed(bh(2)));
    CHECK_EQ(t.stats().ck_drops, 0ll);
}

TEST(the_blocks_go_to_disk_without_waiting_for_the_snapshot_budget) {
    /* Only a couple of snapshots are written a plan -- one is a hundred megabytes on a production
     * hybrid -- and the blocks do not wait for them: a record whose snapshot the budget left
     * behind sends its blocks and goes to the back of the queue for the rest. */
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 16, 16));
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 1 << 20, 1024, 2048, 1 << 16, 16), &d.disk));
    for (int i = 1; i <= 4; ++i) {
        t.note_used(bh((uint64_t)i), {10 + i}, 1 * SEC);
        t.note_snapshot(bh((uint64_t)i), 7 + i, 2048);
    }
    copy_now(t);
    copy_now(t);
    CHECK_EQ(t.stats().ck_to_host, 4ll);

    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 4u);
    int with_ck = 0;
    for (const TierJob& j : jobs) { CHECK(j.host_slot >= 0); with_ck += j.ck_host >= 0; }
    CHECK_EQ(with_ck, 2);
    for (TierJob& j : jobs) t.report(j, true);
    CHECK_EQ(t.stats().to_disk, 4ll);
    CHECK_EQ(t.stats().ck_to_disk, 2ll);

    CHECK(t.disk_waiting());
    jobs.clear();
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 2u);
    for (const TierJob& j : jobs) { CHECK_EQ(j.host_slot, -1); CHECK(j.ck_host >= 0); }
    for (TierJob& j : jobs) t.report(j, true);
    CHECK_EQ(t.stats().to_disk, 4ll);
    CHECK_EQ(t.stats().ck_to_disk, 4ll);
    CHECK(!t.disk_waiting());
}


/* ------------------------------------------------------------------ the cache's half of it */

TEST(a_detached_snapshot_is_not_offered_and_a_reattached_one_is) {
    /* THE INDEX ENTRY OUTLIVES THE BYTES ON PURPOSE. A snapshot the idle tiers hold has no slot
     * in the pool, so it cannot be served -- but it is exactly what a promotion puts back, and
     * the position and session it stands for are not recoverable from anywhere else. The state
     * it is in while tiered is the one a reserved-but-unwritten snapshot is already in, handled
     * by the same branch in lookup(). */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));

    auto prompt = tok_run(7100, 336);
    publish(f, 1, prompt, 320);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    const BlockHash key = hs[128 / 16 - 1];

    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, key, &slot));
    f.pc.commit_checkpoint(key);
    CHECK_EQ(f.pc.lookup(prompt, {}).checkpoint_slot, slot);

    const int64_t free_before = f.kv.free_checkpoint_slots();
    CHECK_OK(f.pc.detach_snapshot(key));
    /* THE SLOT IS THE POINT. There are --checkpoint-slots of them for the whole server, and
     * moving the bytes without giving the slot back would free the plentiful resource and keep
     * the scarce one. */
    CHECK_EQ(f.kv.free_checkpoint_slots(), free_before + 1);
    CHECK_EQ(f.pc.snapshots_off_device(), 1ll);

    CacheHit tiered = f.pc.lookup(prompt, {});
    CHECK_EQ(tiered.n_attn_tokens, 320);      /* the blocks are still here */
    CHECK_EQ(tiered.n_linear_tokens, 0);      /* the state is not */
    CHECK_EQ(tiered.checkpoint_slot, -1);

    int32_t back = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&back));
    CHECK_OK(f.pc.reattach_snapshot(key, back));
    CHECK_EQ(f.pc.snapshots_off_device(), 0ll);
    CacheHit restored = f.pc.lookup(prompt, {});
    CHECK_EQ(restored.n_linear_tokens, 128);
    CHECK_EQ(restored.checkpoint_slot, back);
}

TEST(a_tiered_snapshot_is_not_a_victim_and_not_a_slot_to_hand_back) {
    /* Two ways the checkpoint index can go wrong once a snapshot can be somewhere else, and both
     * of them are silent. Choosing a tiered snapshot as the victim when the SLOT pool is full
     * frees nothing and retires a usable snapshot; handing its -1 back from reserve_checkpoint
     * reads to the scheduler exactly like a refusal, and linear caching stops with nothing in
     * the log. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/2));

    auto prompt = tok_run(7200, 336);
    publish(f, 1, prompt, 320);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }

    int32_t s1 = -1, s2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, hs[64 / 16 - 1], &s1));
    f.pc.commit_checkpoint(hs[64 / 16 - 1]);
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, hs[128 / 16 - 1], &s2));
    f.pc.commit_checkpoint(hs[128 / 16 - 1]);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    CHECK_OK(f.pc.detach_snapshot(hs[64 / 16 - 1]));
    CHECK_EQ(f.kv.free_checkpoint_slots(), 1ll);

    /* A reservation at the tiered snapshot's own prefix gets a REAL slot, not its -1. */
    int32_t again = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, hs[64 / 16 - 1], &again));
    CHECK(again >= 0);
    CHECK_EQ(f.pc.snapshots_off_device(), 0ll);
}

TEST(the_tiers_losing_a_snapshot_erases_the_index_entry) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    auto prompt = tok_run(7300, 336);
    publish(f, 1, prompt, 320);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    const BlockHash key = hs[128 / 16 - 1];
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, key, &slot));
    f.pc.commit_checkpoint(key);
    CHECK_OK(f.pc.detach_snapshot(key));

    /* The snapshot arena overflowed, or a promotion could not place the state. Either way the
     * bytes are gone and an index entry that outlived them would be a checkpoint the cache goes
     * on offering that nothing can ever fill. */
    f.pc.drop_snapshot(key);
    CHECK_EQ(f.pc.snapshots_off_device(), 0ll);
    CHECK_EQ(f.pc.checkpoints(1).size(), 0u);
    CHECK_EQ(f.pc.lookup(prompt, {}).n_linear_tokens, 0);
    /* And no double free: the device slot went back at detach, not here. */
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4ll);
}

/* A FULL CHECKPOINT POOL DETACHES WHAT THE TIERS HOLD, and only that. The victim's slot goes back
 * either way; a victim with a finished copy off the card keeps its index entry, marked off-device,
 * so the session's next restore brings the state back instead of replaying every recurrent layer.
 * A victim with no copy is retired. */
TEST(a_full_checkpoint_pool_detaches_a_snapshot_the_tiers_hold_rather_than_losing_it) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/2));
    const int64_t ck = f.kv.checkpoint_bytes();
    REQUIRE(ck > 0);
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(64 << 20, 0, f.kv.plan(G_FULL)->bytes_per_block, ck, 8 * ck, 0),
                         nullptr));
    REQUIRE(t.ckpt_on());
    f.pc.set_tiers(&t);

    auto chain = [&](const std::vector<int32_t>& p) {
        std::vector<BlockHash> hs;
        BlockHash parent;
        for (int b = 0; b < 21; ++b) {
            parent = chain_block_hash(parent, p.data() + b * 16, 16, nullptr, 0, b * 16);
            hs.push_back(parent);
        }
        return hs;
    };
    const auto a = tok_run(7500, 336), b = tok_run(9500, 336);
    publish(f, 1, a, 320);
    const std::vector<BlockHash> ha = chain(a), hb = chain(b);
    int32_t s1 = -1, s2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, ha[64 / 16 - 1], &s1));
    f.pc.commit_checkpoint(ha[64 / 16 - 1]);
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, ha[128 / 16 - 1], &s2));
    f.pc.commit_checkpoint(ha[128 / 16 - 1]);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);
    copy_now(t);
    CHECK(t.snapshot_backed(ha[64 / 16 - 1]));

    /* Another session wants a slot; the least recently used snapshot that is not a tip is the
     * one at 64, and it has a copy. */
    int32_t s3 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(2, 64, hb[64 / 16 - 1], &s3));
    CHECK(s3 >= 0);
    CHECK_EQ(f.pc.snapshots_off_device(), 1ll);
    CHECK_EQ(f.pc.checkpoints(1).size(), 2u);      /* the session still knows both */
    CHECK(!t.snapshot_backed(ha[64 / 16 - 1]));
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);    /* nothing was lost */
    f.pc.set_tiers(nullptr);
}

/* A RESTORED SESSION'S STATE NEEDS A DEVICE SLOT, and the pool is full whenever as many sessions
 * have run since. The restore takes one from the least recently used snapshot the tiers hold a
 * copy of -- but never one a lookup handed out this step: that request reads the slot in the
 * coming forward, and the restore writes it before the forward runs. */
TEST(a_restore_takes_a_checkpoint_slot_only_from_a_backed_snapshot_nobody_was_handed) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/2));
    const int64_t ck = f.kv.checkpoint_bytes();
    REQUIRE(ck > 0);
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(64 << 20, 0, f.kv.plan(G_FULL)->bytes_per_block, ck, 8 * ck, 0),
                         nullptr));
    REQUIRE(t.ckpt_on());
    f.pc.set_tiers(&t);

    const auto a = tok_run(7500, 336);
    publish(f, 1, a, 320);
    std::vector<BlockHash> ha;
    BlockHash parent;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, a.data() + b * 16, 16, nullptr, 0, b * 16);
        ha.push_back(parent);
    }
    int32_t s1 = -1, s2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, ha[64 / 16 - 1], &s1));
    f.pc.commit_checkpoint(ha[64 / 16 - 1]);
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, ha[128 / 16 - 1], &s2));
    f.pc.commit_checkpoint(ha[128 / 16 - 1]);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);
    copy_now(t);
    CHECK(t.snapshot_backed(ha[128 / 16 - 1]));

    /* This step's admission is handed the one at 128. */
    f.pc.begin_step();
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(a, {}, &hit);
    REQUIRE_EQ(h.n_linear_tokens, 128);
    REQUIRE_EQ(h.checkpoint_slot, s2);

    CHECK(f.pc.free_checkpoint_for_restore());        /* the one at 64 */
    CHECK(!t.snapshot_backed(ha[64 / 16 - 1]));
    CHECK_EQ(f.kv.free_checkpoint_slots(), 1ll);
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);       /* detached, not lost */
    int32_t taken = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&taken));

    CHECK(!f.pc.free_checkpoint_for_restore());       /* 128 is this step's */
    f.pc.begin_step();
    CHECK(f.pc.free_checkpoint_for_restore());        /* and the next step's to take */
    CHECK_EQ(f.pc.snapshots_off_device(), 2ll);
    f.kv.free_checkpoint(taken);
    f.pc.set_tiers(nullptr);
}

TEST(the_tiers_are_told_the_cache_s_block_ids_and_not_the_requests) {
    /* TWO REQUESTS COMPUTE THE SAME PREFIX AND ONLY ONE SET OF BLOCKS IS REFERENCED. The second
     * request missed -- its lookup stopped early, which is exactly what happens while the tiers
     * hold the entry -- so it filled its own blocks for tokens the cache already has. insert()
     * keeps the entry it already holds and lets the second request's copy go.
     *
     * Told the second request's ids, the tiers would later gather out of blocks the pool had
     * since handed to somebody else -- putting a stranger's KV into the tier under this prefix's
     * key -- and release a reference nobody took. What that looks like from outside is a log full
     * of "double free of block N" and, on the path that promotes the entry back, fluent text with
     * one word changed: intermittent, plausible, and invisible to a single-session test. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1 << 20, 0, 1024), nullptr));
    f.pc.set_tiers(&t);

    auto prompt = tok_run(7400, 336);
    publish(f, 1, prompt, 320);
    std::vector<int32_t> first = f.kv.block_table(1, f.pc.cached_groups()[0]);
    CHECK(first.size() >= 20u);

    /* The second sequence gets its own blocks for the same tokens; the cache keeps the first's. */
    publish(f, 2, prompt, 320);
    std::vector<int32_t> second = f.kv.block_table(2, f.pc.cached_groups()[0]);
    CHECK(second[0] != first[0]);

    std::vector<TierJob> jobs;
    t.plan_writeback(512, &jobs);
    CHECK(jobs.size() >= 20u);
    /* Every job names a block the CACHE holds. */
    for (const TierJob& j : jobs) {
        bool from_cache = false, from_second = false;
        for (size_t i = 0; i < first.size(); ++i)  if (j.blocks[0] == first[i])  from_cache = true;
        for (size_t i = 0; i < second.size(); ++i) if (j.blocks[0] == second[i]) from_second = true;
        CHECK(from_cache);
        CHECK(!from_second);
    }
}

/* ================================================================== the pool in bytes
 *
 * WHY THE BYTE FIGURE EXISTS AT ALL. Blocks and tokens are both un-summable across KV groups: a
 * recurrent group has no token axis and hands out states rather than blocks, and two PAGED groups
 * share one token axis, so a token of context takes a slot in each and adding their capacities
 * reports a pool several times the size of the one the card holds. Bytes are disjoint extents of
 * VRAM and are the one measure that survives being added up -- which is what makes them the one
 * measure that can answer "how much of the card is the KV cache holding".
 */

TEST(kv_pool_bytes_are_the_plan_and_used_is_what_is_held) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true));

    const KVGroupPlan* pg = f.kv.plan(G_FULL);
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    CHECK(pg && ps);

    /* The capacity is the plan's own arithmetic and not a second derivation of it. */
    CHECK_EQ(f.kv.pool_bytes(G_FULL), pg->n_blocks * pg->bytes_per_block);
    CHECK_EQ(f.kv.pool_bytes(G_GDN),  ps->n_states * ps->bytes_per_state);
    /* The sentinel at handle 0 is neither paged nor stateful and must contribute nothing: it is
     * summed over with every real group, and a non-zero answer there would inflate every total. */
    CHECK_EQ(f.kv.pool_bytes(G_NULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_NULL), 0ll);
    /* And a handle no group was ever carved for. */
    CHECK_EQ(f.kv.pool_bytes(99), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(99), 0ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 32));                       /* two 16-token blocks */
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * pg->bytes_per_block);
    /* The recurrent half is one state and it is charged the moment the sequence exists, because
     * that is when the slot stops being available to anyone else. */
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), ps->bytes_per_state);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}

/* USED TRACKS THE FREE LIST AND NOT THE SEQUENCE TABLE, which is the whole reason it can be read
 * as "VRAM this server cannot hand to the next request". A finished turn's blocks are held by the
 * prefix cache with no sequence attached, and they are occupying the card for exactly as long. */
TEST(kv_bytes_in_use_counts_a_block_only_the_cache_still_holds) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const int64_t per = f.kv.plan(G_FULL)->bytes_per_block;

    publish(f, 1, tok_run(100, 32), 32);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.kv.free_sequence(1);                    /* the turn ends; the cache keeps its reference */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.pc.clear();                             /* and now nothing holds them */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
}

/* ================================================================== a dropped copy gives VRAM back
 *
 * THE LINK NEITHER HALF OF THE SUITE COVERS ON ITS OWN. IdleTiers decides that an entry's VRAM may
 * go and KVManager owns the free list, and the release that connects them lives in the cache --
 * PrefixCache::drop_clean, which is also where a block some running sequence still reads is held
 * back. A drop that moved the record and never released the blocks would pass every tier test in
 * this file and every KV test, and the symptom would be a cache that grows to the size of the pool
 * with the tier counters reporting that it was working. So these drive the whole path and ask the
 * pool rather than the policy.
 */
struct TierFixture {
    Fixture f;
    IdleTiers t;
    int64_t per = 0;
    int build() {
        int s = f.build(/*with_linear=*/false, /*ckpt_slots=*/0);
        if (s < 0) return s;
        s = t.configure(tier_config(1 << 20, 0, 1024), nullptr);
        if (s < 0) return s;
        f.pc.set_tiers(&t);
        per = f.kv.plan(G_FULL)->bytes_per_block;
        return RAD_OK;
    }
};

TEST(dropping_a_clean_entry_gives_its_vram_back_and_keeps_it_restorable) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty 16-token blocks */
    x.f.kv.free_sequence(1);                        /* finished: only the cache holds them */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);

    CHECK_EQ(copy_now(x.t).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.resident(), 0ll);
    CHECK_EQ(x.f.pc.size(), 20ll);                  /* every entry still indexed */
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);        /* nothing was lost */
    /* The lookup stops where VRAM stops, and the restore path sees the whole chain. */
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 0);
    CHECK_EQ(x.f.pc.restorable_from(prompt, {}).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 0);               /* and there is nothing left to give */
}

TEST(an_entry_a_running_sequence_reads_is_dropped_once_the_sequence_lets_go) {
    /* A BLOCK A SEQUENCE ALSO HOLDS WOULD FREE NOTHING, and its entry would then read off-device
     * while its bytes sat in VRAM under the sequence's table -- the next lookup would restore a
     * second copy beside them. */
    TierFixture x;
    CHECK_OK(x.build());
    publish(x.f, 1, tok_run(7600, 336), 320);       /* sequence 1 is still running */
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 21 * x.per);

    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* asked again because a sequence let go */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}

TEST(a_failed_copy_leaves_the_vram_exactly_where_it_was) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t, /*ok=*/false);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
}

TEST(pressure_on_the_pool_moves_a_clean_entry_and_drops_one_with_no_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);                                  /* a is clean */
    publish(x.f, 2, b, 320);
    x.f.kv.free_sequence(2);                        /* b has no copy yet */

    CHECK_EQ(x.f.pc.evict_lru(1000), 40);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);   /* kept, off the card */
    CHECK_EQ(x.f.pc.restorable_from(b, {}).size(), 0u);    /* gone */
    CHECK_EQ(x.f.pc.size(), 20ll);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 20ll);
    /* And the entries it kept are not on the list pressure walks: evicting again frees nothing
     * and loses nothing. */
    CHECK_EQ(x.f.pc.evict_lru(1000), 0);
    CHECK_EQ(x.f.pc.size(), 20ll);
}

TEST(a_prefix_computed_again_takes_the_new_blocks_as_its_device_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.t.arena().used_slots(), 20ll);

    /* A request that could not reach the tier copy computed the same tokens. */
    publish(x.f, 2, prompt, 320);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.resident(), 20ll);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);        /* the older copy went */
    CHECK(x.t.writeback_waiting());                 /* and the new blocks want one of their own */
    const std::vector<int32_t> bt = x.f.kv.block_table(2, G_FULL);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, bt[0]), 2);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
}

TEST(evicting_or_clearing_an_entry_on_its_way_back_releases_nothing) {
    /* AN ENTRY WITH A PROMOTION OUT ON IT STILL NAMES THE IDS IT LEFT WITH, and those belong to
     * whoever the pool gave them to since. */
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7700, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);

    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(!keys.empty());
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.ensure(2, 16));
    const int32_t taken = x.f.kv.block_table(2, G_FULL)[0];

    TierJob up;
    CHECK(x.t.plan_promote(keys[0], &up));          /* out, on its way back */
    x.f.pc.evict_lru(1000);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    x.f.pc.clear();
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    CHECK(x.t.doomed(keys[0]));                     /* forgotten, and finished off on report */
    CHECK_EQ(x.t.arena().used_slots(), 1ll);        /* the one the promotion is reading */
    x.t.report(up, false);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);
}

/* ================================================================== the cache-only count
 *
 * KVManager keeps the number of blocks only the cache holds current at every reference change, so
 * that reading it is free. A count maintained at six call sites is a count that drifts the first
 * time one of them is missed, so these check it after every operation against the figure taken
 * from scratch: a block at a count of one that no live sequence holds can only be the cache's.
 */
static int64_t cache_only_recount(Fixture& f, const std::vector<uint64_t>& live) {
    const int64_t nb = f.kv.plan(G_FULL)->n_blocks;
    std::vector<uint8_t> in_seq((size_t)nb, 0);
    for (uint64_t id : live)
        for (int32_t b : f.kv.block_table(id, G_FULL)) in_seq[(size_t)b] = 1;
    int64_t n = 0;
    for (int64_t b = 0; b < nb; ++b)
        if (f.kv.block_refcount(G_FULL, (int32_t)b) == 1 && !in_seq[(size_t)b]) ++n;
    return n;
}

TEST(the_cache_only_count_follows_every_reference_change) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x](uint64_t n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x % n; };
    std::vector<uint64_t> live;
    uint64_t next_id = 1;
    int64_t hits = 0, forks = 0, evicted = 0, bad = 0;
    for (int it = 0; it < 4000; ++it) {
        const uint64_t op = rnd(6);
        if (op <= 1 && live.size() < 8) {
            /* A prompt from one of six families: a prefix of a family's run is a hit on whatever
             * that family has published before. */
            const auto prompt = tok_run(1000 * (int32_t)rnd(6), 16 + (int)rnd(160));
            const uint64_t id = next_id++;
            CHECK_OK(f.kv.add_sequence(id));
            std::vector<std::vector<int32_t>> found;
            const CacheHit h = f.pc.lookup(prompt, {}, &found);
            if (h.n_attn_tokens > 0) {
                CHECK_OK(f.kv.adopt(id, G_FULL, found[0], h.n_attn_tokens));
                ++hits;
            }
            int st = f.kv.ensure(id, (int64_t)prompt.size());
            if (st == RAD_E_FULL) { evicted += f.pc.evict_lru(64); st = f.kv.ensure(id, (int64_t)prompt.size()); }
            if (st != RAD_OK) { f.kv.free_sequence(id); continue; }
            std::vector<std::vector<int32_t>> bt{ f.kv.block_table(id, G_FULL) };
            CHECK_OK(f.pc.insert(id, prompt, {}, bt, (int64_t)prompt.size()));
            live.push_back(id);
        } else if (op == 2 && !live.empty() && live.size() < 8) {
            const uint64_t id = next_id++;
            if (f.kv.fork(live[(size_t)rnd(live.size())], id) == RAD_OK) { live.push_back(id); ++forks; }
        } else if (op <= 4 && !live.empty()) {
            const size_t k = (size_t)rnd(live.size());
            f.kv.free_sequence(live[k]);
            live.erase(live.begin() + (ptrdiff_t)k);
        } else if (op == 5) {
            evicted += f.pc.evict_lru((int64_t)rnd(9));
        }
        if (f.pc.unshared_tokens() != 16 * cache_only_recount(f, live)) ++bad;
    }
    CHECK_EQ(bad, 0ll);
    /* Every kind of change happened, or the loop proved less than it says. */
    CHECK(hits > 100);
    CHECK(forks > 100);
    CHECK(evicted > 100);
    for (uint64_t id : live) f.kv.free_sequence(id);
    live.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 16 * f.pc.resident());
    f.pc.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(cache_only_recount(f, live), 0ll);
}

TEST(the_cache_only_count_follows_a_drop_and_a_restore) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty blocks, sequence 1 reading them */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* off the card: none of it counts */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);

    /* A restore reserves blocks and hands each reservation to the cache as its reference. */
    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(keys.size() == 20u);
    int64_t handed = 0;
    for (const BlockHash& k : keys) {
        int32_t b = -1;
        CHECK_OK(x.f.kv.reserve_block(G_FULL, &b));
        CHECK_EQ(x.f.pc.unshared_tokens(), 16 * handed);   /* a reservation is not the cache's */
        CHECK_OK(x.f.pc.rebind(k, { b }));
        ++handed;
    }
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    std::vector<std::vector<int32_t>> found;
    const CacheHit h = x.f.pc.lookup(prompt, {}, &found);
    REQUIRE(h.n_attn_tokens == 320);
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.adopt(2, G_FULL, found[0], h.n_attn_tokens));
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);
    x.f.pc.clear();
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}


/* ================================================================== the tier names
 *
 * THESE ARE READ BY A PERSON AND BY A SCRAPER, on the sessions view and in the SSD tier's report,
 * and the value of a name is that it is stable and distinct. A "?" where a tier belongs means the
 * enum grew a value this switch does not know -- which is exactly when a reader most needs to be
 * told, rather than shown the name of a neighbouring tier.
 */
TEST(every_tier_and_consumer_has_its_own_stable_name) {
    CHECK_EQ(std::string(kv_tier_name(KVTier::Device)), std::string("device"));
    CHECK_EQ(std::string(kv_tier_name(KVTier::Host)), std::string("host"));
    CHECK_EQ(std::string(kv_tier_name(KVTier::Disk)), std::string("disk"));
    /* A value outside the enum says so instead of picking one. */
    CHECK_EQ(std::string(kv_tier_name((KVTier)99)), std::string("?"));

    const char* pc = ssd_consumer_name(SSDConsumer::PrefixCache);
    const char* wt = ssd_consumer_name(SSDConsumer::WeightTier);
    CHECK(pc && *pc);
    CHECK(wt && *wt);
    /* THE TWO CONSUMERS SHARE ONE BUCKET, so the report's whole job is telling them apart:
       two rows under one name would read as one consumer using twice the budget. */
    CHECK(std::strcmp(pc, wt) != 0);
}

/* ONE BUCKET, PROCESS WIDE, and that is the entire reason it lives in this component rather than
 * inside either consumer: the weight tier and the prefix tier draw on the same disk. Two objects
 * would each honour the configured rate and the device would see twice it. */
TEST(the_ssd_bucket_is_one_shared_object) {
    CHECK(&rad_ssd_bucket() == &rad_ssd_bucket());
    /* Unconfigured it is off, and off means "do not rate limit" rather than "refuse": a tier
       whose limiter was never given a rate has to keep working. */
    const std::string r = rad_ssd_bucket().report();
    CHECK(!r.empty());
}

/* ================================================================== the disk clock, and the loan */

/* THE CLOCK IS THE LAST USE, and a restore is one. An entry that came back and left again has
 * been idle since it was last read, not since it was first published -- otherwise a conversation
 * in daily use would be pushed to disk on the schedule of its first turn. */
TEST(a_restored_disk_entry_keeps_its_disk_copy_so_its_slot_can_go_again) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1024, 1 << 20, 1024), &d.disk));   /* one slot */
    t.note_used(bh(1), {10}, 1 * SEC);
    copy_now(t);
    CHECK_EQ(disk_now(t).size(), 1u);
    CHECK(drop_now(t, bh(1)));
    CHECK_EQ(t.make_room(1), 1ll);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Disk);

    REQUIRE_EQ(fetch_now(t, {bh(1)}).size(), 1u);   /* into a host slot first */
    TierJob up;
    REQUIRE(t.plan_promote(bh(1), &up));
    REQUIRE(up.host_slot >= 0);
    up.blocks = {11};
    up.started = true;
    t.report(up, true, 1000 * SEC);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);
    CHECK(!t.disk_waiting());                   /* not written a second time */

    /* The slot it was read into goes again for nothing, and the VRAM copy stays clean through the
     * disk. */
    CHECK_EQ(t.make_room(1), 1ll);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK(t.droppable(bh(1)));
    CHECK(drop_now(t, bh(1)));
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Disk);
    CHECK_EQ(t.stats().to_disk, 1ll);
}

/* THE DISK STORE GIVES UP ITS LEAST RECENTLY USED ENTRY, and a read is a use. Finding that entry
 * by walking every slot on every write would cost millions of slots at the sizes it is given, on
 * the thread that has to keep up with the host tier's copies. */
TEST(a_full_disk_store_gives_up_its_least_recently_used_entry) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 4, 1));
    std::vector<char> buf(1024, 0), got(1024, 0);
    auto put = [&](uint64_t k) {
        std::memset(buf.data(), (int)k, buf.size());
        return d.disk.put_block(bh(k), nullptr, 0, buf.data(), 1024);
    };
    auto has = [&](uint64_t k) {
        return d.disk.get_block(bh(k), nullptr, 0, got.data(), 1024) == RAD_OK &&
               got[0] == (char)k;
    };
    for (uint64_t k = 1; k <= 4; ++k) CHECK_OK(put(k));
    CHECK(has(1));                                /* 1 is now the most recently used */
    CHECK_OK(put(3));                             /* a rewrite takes no new slot */
    CHECK_EQ(d.disk.stats().evictions, 0ll);
    CHECK_OK(put(5));                             /* evicts 2, the least recently used */
    CHECK_EQ(d.disk.stats().evictions, 1ll);
    CHECK(!has(2));
    CHECK_OK(put(6));                             /* then 4 */
    CHECK(!has(4));
    CHECK(has(1));
    CHECK(has(3));
    CHECK(has(5));
    CHECK(has(6));
}

/* A CHAIN IS READ AS ONE CALL, many reads in flight at once, and each block answers for itself: a
 * key the store does not hold and a slot that is not the key asked for are misses, the poisoned
 * slot is given up, and neither costs the blocks around it. */
TEST(the_disk_store_reads_a_chain_at_once_and_each_block_answers_for_itself) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 0, 64, 0));
    std::vector<std::vector<char>> in(40, std::vector<char>(1024));
    for (size_t k = 0; k < in.size(); ++k) {
        for (size_t i = 0; i < 1024; ++i) in[k][i] = (char)(k * 31 + i * 7);
        if (k == 17) continue;                    /* never written */
        if (k == 23) {                            /* written as a different shape of block */
            const std::vector<int32_t> toks = tok_run(5, 16);
            CHECK_OK(d.disk.put_block(bh(k), toks.data(), 16, in[k].data(), 1024));
            continue;
        }
        CHECK_OK(d.disk.put_block(bh(k), nullptr, 0, in[k].data(), 1024));
    }
    std::vector<BlockHash> keys;
    std::vector<std::vector<char>> out(in.size(), std::vector<char>(1024, 0));
    std::vector<void*> dst;
    for (size_t k = 0; k < in.size(); ++k) { keys.push_back(bh(k)); dst.push_back(out[k].data()); }
    std::vector<char> ok(in.size(), 9);
    CHECK_OK(d.disk.get_blocks(keys.data(), (int64_t)keys.size(), dst.data(), 1024, ok.data()));
    for (size_t k = 0; k < in.size(); ++k) {
        const bool want = k != 17 && k != 23;
        CHECK_EQ((int)ok[k], want ? 1 : 0);
        if (want) CHECK(out[k] == in[k]);
    }
    CHECK_EQ(d.disk.stats().hits, 38ll);
    CHECK_EQ(d.disk.stats().verify_misses, 1ll);
    /* The slot that failed its check is gone, so the next read of it is a plain miss. */
    CHECK_EQ(d.disk.get_block(bh(23), nullptr, 0, out[23].data(), 1024), RAD_E_NOTFOUND);
    CHECK_EQ(d.disk.stats().verify_misses, 1ll);
    /* A payload size the store was not opened for is refused before any read. */
    CHECK_EQ(d.disk.get_blocks(keys.data(), 1, dst.data(), 512, ok.data()), RAD_E_SHAPE);
    CHECK_EQ((int)ok[0], 0);
}

/* A READ THAT FAILS LOSES THE ENTRY. The record said the disk held bytes it does not; the slot it
 * was reading into goes back, and the cache is told the key is gone so a lookup stops matching it.
 * The restore that asked then finds a shorter chain -- which is the prompt's hit from then on. */
TEST(a_disk_read_that_fails_drops_the_entry_and_gives_its_slot_back) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 0, 8, 0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4 * 1024, 1 << 20, 1024), &d.disk));
    std::vector<BlockHash> dropped;
    t.set_drop_sink([&](const BlockHash& k) { dropped.push_back(k); });
    for (uint64_t k = 1; k <= 2; ++k) t.note_used(bh(k), {(int32_t)(10 + k)}, k * SEC);
    copy_now(t);
    CHECK_EQ(disk_now(t).size(), 2u);
    for (uint64_t k = 1; k <= 2; ++k) CHECK(drop_now(t, bh(k)));
    CHECK_EQ(t.make_room(2), 2ll);
    CHECK_EQ(t.arena().used_slots(), 0ll);

    const std::vector<TierJob> rd = fetch_now(t, {bh(1), bh(2)}, /*ok=*/false);
    CHECK_EQ(rd.size(), 2u);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK(!t.resident_off_device(bh(1)));
    CHECK(!t.resident_off_device(bh(2)));
    CHECK_EQ(dropped.size(), 2u);
    CHECK_EQ(t.stats().failures, 2ll);
    CHECK_EQ(t.stats().fetched, 0ll);
    /* Nothing left to read, so nothing to wait for. */
    std::vector<TierJob> again;
    CHECK(!t.plan_fetch({bh(1), bh(2)}, &again, 1));

    /* A read the IO thread never began is not a failure: the entry is still on disk. */
    t.note_used(bh(3), {13}, 3 * SEC);
    copy_now(t);
    CHECK_EQ(disk_now(t).size(), 1u);
    CHECK(drop_now(t, bh(3)));
    CHECK_EQ(t.make_room(1), 1ll);
    std::vector<TierJob> r3;
    CHECK(t.plan_fetch({bh(3)}, &r3, 1));
    REQUIRE_EQ(r3.size(), 1u);
    t.report(r3[0], false);
    CHECK_EQ(t.tier_of(bh(3)), KVTier::Disk);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK_EQ(t.stats().failures, 2ll);
    std::vector<TierJob> r3b;
    CHECK(t.plan_fetch({bh(3)}, &r3b, 1));        /* and it can be asked for again */
    CHECK_EQ(r3b.size(), 1u);
}

TEST(a_host_copy_goes_on_to_disk_at_once_and_keeps_its_slot) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1 << 20, 1024), &d.disk));
    t.note_used(bh(1), {10}, 1 * SEC);
    CHECK(!t.disk_waiting());                   /* nothing in host memory yet */
    const std::vector<TierJob> out = copy_now(t);
    REQUIRE_EQ(out.size(), 1u);
    CHECK(t.disk_waiting());

    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].kind == TierJob::Kind::ToDisk);
    CHECK_EQ(jobs[0].host_slot, out[0].host_slot);

    /* A WRITE DOES NOT HOLD THE ENTRY. Its VRAM can go and a hit can bring it straight back out of
     * the slot the write is reading -- a restore waiting on a disk write would be a re-prefill. */
    CHECK(drop_now(t, bh(1)));
    TierJob up;
    REQUIRE(t.plan_promote(bh(1), &up));
    CHECK_EQ(up.host_slot, out[0].host_slot);
    up.blocks = {11};
    up.started = true;
    t.report(up, true);
    t.report(jobs[0], true);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);
    CHECK_EQ(t.stats().to_disk, 1ll);
    CHECK_EQ(t.arena().used_slots(), 1ll);
    IdleTiers::Occupancy occ;
    t.sessions(&occ);
    CHECK_EQ(occ.disk_bytes, 1024);
}

TEST(a_full_arena_gives_up_what_the_disk_holds_for_nothing) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1 << 20, 1024), &d.disk));   /* two slots */
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    CHECK_EQ(disk_now(t).size(), 2u);
    CHECK(drop_now(t, bh(1)));
    CHECK(drop_now(t, bh(2)));

    /* The third copy takes the oldest host entry's slot: it is a disk entry now, nothing was
     * written to make it one, and nothing was lost. */
    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Disk);
    CHECK_EQ(t.tier_of(bh(2)), KVTier::Host);
    CHECK(t.droppable(bh(3)));
    CHECK(lost.empty());
    CHECK_EQ(t.stats().host_freed, 1ll);
    CHECK_EQ(t.stats().host_drops, 0ll);
    CHECK(t.resident_off_device(bh(1)));
}

TEST(a_clean_vram_entry_gives_up_its_host_copy_once_the_disk_holds_it) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1 << 20, 1024), &d.disk));
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    copy_now(t);
    disk_now(t);

    /* Both are still in VRAM and hold the arena between them; the older one's host copy goes. */
    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Device);
    CHECK_EQ(t.stats().host_freed, 1ll);
    /* ...and it stays clean through its disk copy: not copied again, and free to drop. */
    CHECK(!t.writeback_waiting());
    CHECK(t.droppable(bh(1)));
    CHECK(drop_now(t, bh(1)));
    CHECK_EQ(t.tier_of(bh(1)), KVTier::Disk);
}

TEST(a_write_back_with_no_room_waits_for_a_disk_copy_to_land) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1 << 20, 1024), &d.disk));
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    CHECK(drop_now(t, bh(1)));
    CHECK(drop_now(t, bh(2)));

    /* Neither host entry is on disk yet, so neither slot can go. */
    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);
    CHECK(!t.writeback_waiting());
    CHECK_EQ(disk_now(t).size(), 2u);
    CHECK(t.writeback_waiting());
    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK(t.droppable(bh(3)));
}

TEST(a_slot_given_up_while_its_disk_write_is_out_is_freed_when_the_write_reports) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 8, 4));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1 << 20, 1024), &d.disk));

    /* Forgotten mid-write: the slot outlives the record until the write is done reading it. */
    t.note_used(bh(1), {11}, 1 * SEC);
    copy_now(t);
    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    t.forget(bh(1));
    CHECK_EQ(t.arena().used_slots(), 1ll);
    t.report(jobs[0], true);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK_EQ(t.stats().to_disk, 0ll);

    /* Recomputed mid-write: the new VRAM copy is not what the write is writing, so the entry does
     * not count as on disk -- it is dirty, and wants a copy of its own. */
    t.note_used(bh(2), {12}, 2 * SEC);
    copy_now(t);
    jobs.clear();
    t.plan(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    t.rebound(bh(2), {22}, 3 * SEC);
    CHECK_EQ(t.arena().used_slots(), 1ll);
    t.report(jobs[0], true);
    CHECK_EQ(t.arena().used_slots(), 0ll);
    CHECK(!t.droppable(bh(2)));
    CHECK(t.writeback_waiting());
}

TEST(a_snapshot_the_disk_holds_gives_its_slot_up_before_one_only_the_host_holds) {
    TmpDisk d;
    CHECK_OK(d.open(1024, 2048, 16, 16));
    IdleTiers t;
    CHECK_OK(t.configure(ck_config(1 << 20, 1 << 20, 1024, 2048, 2 * 2048, 16), &d.disk));
    REQUIRE_EQ(t.ckpt_arena().capacity(), 2);
    int lost_ck = 0;
    t.set_snapshot_drop_sink([&lost_ck](const BlockHash&) { ++lost_ck; });

    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    t.note_snapshot(bh(1), 7, 2048);
    t.note_snapshot(bh(2), 8, 4096);
    copy_now(t);
    CHECK_EQ(t.stats().ck_to_host, 2ll);
    for (uint64_t k : {1, 2}) { CHECK(drop_now(t, bh(k))); CHECK(detach_now(t, bh(k))); }

    /* Only the second snapshot reaches disk. */
    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    for (TierJob& j : jobs) { if (j.key == bh(1)) j.ck_ok = false; t.report(j, true); }
    CHECK_EQ(t.stats().ck_to_disk, 1ll);

    /* The earliest position would lose -- except that one is the only copy of its state, and the
     * other costs nothing to give up. */
    t.note_used(bh(3), {13}, 3 * SEC);
    t.note_snapshot(bh(3), 9, 6144);
    copy_now(t);
    CHECK_EQ(t.stats().ck_to_host, 3ll);
    CHECK_EQ(lost_ck, 0);
    CHECK_EQ(t.stats().ck_drops, 0ll);

    /* The second session's state is now only on disk, and reading it back needs a slot. A read
     * never takes another session's only copy -- nor the copy that lets a snapshot still in its
     * device slot be detached, until the disk holds it too. With no slot the read is not planned
     * and the session is restored without it. */
    std::vector<TierJob> rd;
    CHECK(!t.plan_fetch({bh(2)}, &rd, 1));
    CHECK(rd.empty());
    CHECK(drop_now(t, bh(3)));
    CHECK(t.snapshot_backed(bh(3)));
    CHECK(!t.plan_fetch({bh(2)}, &rd, 1));
    CHECK(rd.empty());
    CHECK(t.snapshot_backed(bh(3)));
    disk_now(t);
    CHECK(t.plan_fetch({bh(2)}, &rd, 1));
    REQUIRE_EQ(rd.size(), 1u);
    CHECK_EQ(rd[0].host_slot, -1);                /* its blocks are in host memory already */
    CHECK(rd[0].ck_host >= 0);
    CHECK_EQ(rd[0].ck_pos, 4096ll);
    CHECK(t.snapshot_backed(bh(3)));              /* through its disk copy now */
    rd[0].started = true;
    t.report(rd[0], true);
    CHECK_EQ(t.stats().ck_fetched, 1ll);
    TierJob up;
    CHECK(t.plan_promote(bh(2), &up));
    CHECK(up.ck_want);
    CHECK_EQ(up.ck_host, rd[0].ck_host);
    CHECK_EQ(lost_ck, 0);
}

/* A RESTORE IS SIZED BEFORE IT STARTS, so it recalls the loan for the whole chain at once. One
 * recall a block would drain the slab a buffer at a time and stop at whatever the cache kept in
 * hand. */
TEST(a_restore_reserves_its_whole_chain_against_one_recall) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    int calls = 0;
    e.kv.set_reclaim([&](int64_t) { ++calls; return (int)RAD_OK; });

    const int64_t want = (idle + all) / 2;
    std::vector<int32_t> got;
    CHECK_EQ(e.kv.reserve_blocks(G_FULL, want, &got), want);
    CHECK_EQ((int64_t)got.size(), want);
    CHECK_EQ(calls, 1);
    for (int32_t b : got) CHECK(b < e.kv.live_blocks(G_FULL));

    /* Past the carve it takes what there is and says how much. */
    std::vector<int32_t> more;
    const int64_t n = e.kv.reserve_blocks(G_FULL, all, &more);
    CHECK_EQ(n, all - want);
    CHECK_EQ((int64_t)more.size(), n);
}


/* ================================================================== the transfer, end to end
 *
 * EVERYTHING ABOVE IS BOOKKEEPING, and this is the one piece that can put another conversation's
 * context into a restored entry: the layer-major addressing, the staging pitch, a batch cut into
 * staging-sized pieces, arena slots that are not consecutive, and under tensor parallelism a rank's
 * window of a slot every rank shares. So a chain is copied out through the real TierExec, its VRAM
 * given up and written over, restored into DIFFERENT blocks, and every byte of every layer compared
 * with what was there before -- on the host backend, whose streams are real queues on threads, so
 * a missing order between the copies and the row kernels would show up here as wrong bytes.
 */
static std::string rows_plugin() {
    const char* d = std::getenv("RAD_TESTPLUGIN_HOME");
    if (!d) d = std::getenv("RADIANCE_HOME");
    return std::string(d ? d : "radiance_home") + "/testrows/radtest_rows.so";
}

/* The byte a pool holds at (rank, layer, block, i) before anything moves. */
/* A full avalanche, so that neighbouring blocks, layers and bytes all differ: a multiply-only hash
 * moves a small input difference into the high bits and leaves the byte taken here unchanged,
 * which is a pattern every wrong offset reproduces. */
static unsigned char pat(int rank, int64_t layer, int32_t block, int64_t i) {
    uint64_t z = ((uint64_t)rank << 60) ^ ((uint64_t)layer << 48) ^ ((uint64_t)block << 20) ^
                 (uint64_t)i;
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (unsigned char)(z ^ (z >> 31));
}

static void fill(Fixture& f, int rank) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        const int64_t frag = p->layer_bytes_per_block();
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b = 0; b < (int32_t)p->n_blocks; ++b) {
                unsigned char* row = (unsigned char*)p->base + l * p->layer_stride() + b * frag;
                for (int64_t i = 0; i < frag; ++i) row[i] = pat(rank, g * 16 + l, b, i);
            }
    }
}

static void scribble(Fixture& f) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        for (int64_t l = 0; l < p->n_layers; ++l)
            std::memset((char*)p->base + l * p->layer_stride(), 0xEE, (size_t)p->layer_stride());
    }
}

/* How many entries' blocks in `now` hold, in every cached group and layer, exactly what the
 * blocks in `was` held before the move. */
static int64_t same_bytes(Fixture& f, int rank, const std::vector<std::vector<int32_t>>& was,
                          const std::vector<std::vector<int32_t>>& now) {
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    int64_t good = 0;
    for (size_t k = 0; k < was[0].size(); ++k) {
        bool ok = true;
        for (size_t gi = 0; gi < cg.size() && ok; ++gi) {
            const KVGroupPlan* p = f.kv.plan(cg[gi]);
            const int64_t frag = p->layer_bytes_per_block();
            for (int64_t l = 0; l < p->n_layers && ok; ++l) {
                const unsigned char* row =
                    (const unsigned char*)p->base + l * p->layer_stride() + now[gi][k] * frag;
                for (int64_t i = 0; i < frag && ok; ++i)
                    ok = row[i] == pat(rank, cg[gi] * 16 + l, was[gi][k], i);
            }
        }
        good += ok;
    }
    return good;
}

/* Two paged groups of different widths, the shape a production model has: the attention cache
 * beside a narrower per-block key cache. One entry is a block in each, and a restore that confused
 * their offsets inside a staging entry would put one group's bytes in the other. */
static std::vector<KVGroupInfo> two_paged_groups() {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo a;
    a.name = "attn";
    a.decl.kind = RAD_KV_FULL;
    a.decl.dtype = RAD_BF16;
    a.decl.n_head_kv = 2;
    a.decl.head_dim = 8;
    a.block_size = 16;
    a.layers = { 0, 1 };
    g.push_back(a);
    KVGroupInfo k;
    k.name = "bkey";
    k.decl.kind = RAD_KV_FULL;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 1;
    k.decl.head_dim = 4;
    k.block_size = 16;
    k.layers = { 2, 3, 4 };
    g.push_back(k);
    return g;
}

static void run_transfer(int world, bool fragment) {
    /* THE HOST BACKEND ONLY. The stand-in kernels copy rows with the CPU, which on the host backend
     * is where "device" memory is; on a card the staging buffer is VRAM and the production kernels
     * are the ones to run. */
    if (!device_is_host()) {
        std::printf("  SKIP  the tier transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());

    std::vector<std::unique_ptr<Fixture>> fx;
    for (int r = 0; r < world; ++r) {
        fx.push_back(std::make_unique<Fixture>());
        fx.back()->cfg.checkpoint_slots = 0;
        REQUIRE_EQ(fx.back()->build_groups(two_paged_groups()), RAD_OK);
    }
    Fixture& f = *fx[0];
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    REQUIRE_EQ(cg.size(), 2u);
    int64_t entry = 0;
    for (int32_t g : cg) entry += f.kv.plan(g)->bytes_per_block;

    /* FORTY SLOTS. Fragmented, every other one is taken and the copy goes out as twenty runs of
     * one; whole, it goes out as one run of twenty, which is the case where a rank's pitch through
     * a slot every rank shares decides where its rows land. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(40 * entry * world, 0, entry * world), nullptr));
    REQUIRE_EQ(t.arena().capacity(), 40);
    if (fragment) {
        for (int i = 0; i < 40; ++i) CHECK_EQ(t.arena().alloc(), i);
        for (int i = 1; i < 40; i += 2) t.arena().free(i);
    }
    const int64_t held_before = t.arena().used_slots();
    f.pc.set_tiers(&t);

    std::vector<RadStream> compute((size_t)world, nullptr);
    std::vector<std::unique_ptr<TierExec>> x;
    for (int r = 0; r < world; ++r) {
        REQUIRE_EQ(rad_stream_create(&compute[(size_t)r], 0), RAD_OK);
        x.push_back(std::make_unique<TierExec>());
        /* A three-entry staging buffer, so twenty entries cross it in seven pieces. */
        REQUIRE_EQ(x.back()->configure(&reg, &fx[(size_t)r]->kv, &t, nullptr, cg, /*max_batch=*/3,
                                       /*max_copy=*/64, /*max_chain=*/64, /*leader=*/r == 0,
                                       /*device=*/0, /*slot_off=*/r * entry, /*ck_slot_off=*/0,
                                       /*shared=*/nullptr), RAD_OK);
    }

    /* THE GROUPS' IDS ARE MADE TO DIFFER. They allocate in step, so an entry is normally the same
     * id in each -- and an index row read for the wrong group would then be right by accident. */
    int32_t skew = -1;
    CHECK_OK(f.kv.reserve_block(cg[1], &skew));

    const auto prompt = tok_run(4100, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    std::vector<std::vector<int32_t>> hit;
    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> was = hit;
    CHECK(was[0][0] != was[1][0]);
    for (int r = 0; r < world; ++r) fill(*fx[(size_t)r], r);

    /* ---- out ---- */
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 20u);
    for (int r = 0; r < world; ++r)
        CHECK_OK(x[(size_t)r]->issue_copy(jobs, compute[(size_t)r]));
    for (int r = 0; r < world; ++r) {
        int st = 0;
        for (int spin = 0; spin < 20000 && (st = x[(size_t)r]->copy_state()) == 0; ++spin)
            usleep(100);
        CHECK_EQ(st, 1);
    }
    x[0]->report_pass(jobs, std::vector<char>(jobs.size(), 1), 0);
    CHECK_EQ(t.stats().to_host, 20);

    /* ---- the VRAM goes, and somebody else writes over it ---- */
    CHECK_EQ(f.pc.drop_clean(), 20);
    for (auto& fp : fx) scribble(*fp);
    /* A new sequence takes the lowest blocks, so the chain cannot come back where it was. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 5 * 16));

    /* ---- back ---- */
    const std::vector<BlockHash> keys = f.pc.restorable_from(prompt, {});
    REQUIRE_EQ(keys.size(), 20u);
    std::vector<TierJob> up;
    for (const BlockHash& k : keys) {
        TierJob j;
        REQUIRE(x[0]->plan_promote(k, &j));
        up.push_back(j);
    }
    std::vector<char> ok(up.size(), 1);
    for (int r = 0; r < world; ++r) {
        std::vector<char> mine;
        CHECK_OK(x[(size_t)r]->issue_restore(up, compute[(size_t)r], &mine));
        for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
    }
    for (char c : ok) CHECK(c);
    x[0]->report_pass(up, ok, 0);
    for (const TierJob& j : up) CHECK_OK(f.pc.rebind(j.key, j.blocks));
    /* THE STEP THAT READS THEM WAITS ON THE DEVICE, not the host: draining the compute stream is
     * what reading them would do. */
    for (int r = 0; r < world; ++r) CHECK_OK(rad_stream_sync(compute[(size_t)r]));

    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> now = hit;
    int64_t moved = 0;
    for (size_t k = 0; k < was[0].size(); ++k) moved += was[0][k] != now[0][k];
    CHECK(moved > 0);
    for (int r = 0; r < world; ++r)
        CHECK_EQ(same_bytes(*fx[(size_t)r], r, was, now), 20);
    /* AND THE COPY STAYED: the entries are clean the moment they land. */
    CHECK_EQ(t.arena().used_slots(), held_before + 20);
    for (const BlockHash& k : keys) CHECK(t.droppable(k));

    for (auto& xp : x) xp->close();
    for (RadStream s : compute) rad_stream_destroy(s);
    f.pc.set_tiers(nullptr);
    (void)cg;
}

/* A CHAIN IS READ OFF DISK AS FAR AS THE ARENA REACHES, AND RESTORED AS FAR AS THE POOL DOES. The
 * read runs before the restore, into slots nothing may take back until the restore has used them --
 * here the arena has room for ten of the chain's twenty entries. The restore then stops where the
 * leader's pool runs short, and a follower finds the same point from the block ids it is handed. A
 * follower that went on past it would copy into blocks that are not the cache's, and its failure
 * would take the whole chain with it: every entry it had started is dropped, and the conversation
 * comes back as a full re-prefill. */
static void run_disk_cut(int world) {
    if (!device_is_host()) {
        std::printf("  SKIP  the tier transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());

    std::vector<std::unique_ptr<Fixture>> fx;
    for (int r = 0; r < world; ++r) {
        fx.push_back(std::make_unique<Fixture>());
        fx.back()->cfg.checkpoint_slots = 0;
        REQUIRE_EQ(fx.back()->build_groups(two_paged_groups()), RAD_OK);
    }
    Fixture& f = *fx[0];
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    int64_t entry = 0;
    for (int32_t g : cg) entry += f.kv.plan(g)->bytes_per_block;

    TmpDisk d;
    REQUIRE_EQ(d.open(entry * world, 0, 64, 0), RAD_OK);
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(40 * entry * world, 1 << 30, entry * world), &d.disk));
    REQUIRE(t.disk_on());
    f.pc.set_tiers(&t);

    std::vector<RadStream> compute((size_t)world, nullptr);
    std::vector<std::unique_ptr<TierExec>> x;
    for (int r = 0; r < world; ++r) {
        REQUIRE_EQ(rad_stream_create(&compute[(size_t)r], 0), RAD_OK);
        x.push_back(std::make_unique<TierExec>());
        REQUIRE_EQ(x.back()->configure(&reg, &fx[(size_t)r]->kv, &t, &d.disk, cg, /*max_batch=*/3,
                                       /*max_copy=*/64, /*max_chain=*/64, /*leader=*/r == 0,
                                       /*device=*/0, /*slot_off=*/r * entry, /*ck_slot_off=*/0,
                                       /*shared=*/nullptr), RAD_OK);
    }

    const auto prompt = tok_run(4100, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    std::vector<std::vector<int32_t>> hit;
    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> was = hit;
    for (int r = 0; r < world; ++r) fill(*fx[(size_t)r], r);

    /* ---- out: to host, on to disk, out of VRAM, and out of the arena ---- */
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 20u);
    for (int r = 0; r < world; ++r) CHECK_OK(x[(size_t)r]->issue_copy(jobs, compute[(size_t)r]));
    for (int r = 0; r < world; ++r) {
        int st = 0;
        for (int spin = 0; spin < 20000 && (st = x[(size_t)r]->copy_state()) == 0; ++spin)
            usleep(100);
        CHECK_EQ(st, 1);
    }
    x[0]->report_pass(jobs, std::vector<char>(jobs.size(), 1), 0);
    std::vector<TierJob> dj;
    t.plan(64, &dj);
    REQUIRE_EQ(dj.size(), 20u);
    std::vector<char> dok;
    x[0]->run_disk(dj, &dok);
    x[0]->report_pass(dj, dok, 0);
    CHECK_EQ(t.stats().to_disk, 20);
    CHECK_EQ(f.pc.drop_clean(), 20);
    CHECK_EQ(t.make_room(20), 20);
    CHECK_EQ(t.arena().used_slots(), 0);
    for (auto& fp : fx) scribble(*fp);
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 5 * 16));

    /* Somebody else holds thirty of the forty slots, where no entry can give them up. */
    std::vector<int32_t> other;
    for (int i = 0; i < 30; ++i) other.push_back(t.arena().alloc());

    std::vector<BlockHash> chain;
    REQUIRE_EQ(f.pc.restorable_from(prompt, {}, &chain).size(), 20u);
    REQUIRE_EQ(chain.size(), 20u);

    /* The leader reads the whole slot, every rank's shard of it; the followers copy out of it. */
    auto fetch = [&](size_t expect) {
        std::vector<TierJob> rd;
        CHECK_EQ(t.plan_fetch(chain, &rd, 1), expect > 0);
        CHECK_EQ(rd.size(), expect);
        if (rd.empty()) return;
        /* Asked again while the read is out, the chain waits on it rather than reading it twice. */
        std::vector<TierJob> again;
        CHECK(t.plan_fetch(chain, &again, 1));
        CHECK(again.empty());
        std::vector<char> ok;
        x[0]->run_fetch(rd, &ok);
        for (char c : ok) CHECK(c);
        x[0]->report_pass(rd, ok, 0);
    };

    auto restore = [&](size_t expect_keys, size_t expect_up) -> std::vector<char> {
        const std::vector<BlockHash> keys = f.pc.restorable_from(prompt, {});
        CHECK_EQ(keys.size(), expect_keys);
        std::vector<TierJob> up;
        for (const BlockHash& k : keys) {
            TierJob j;
            if (!x[0]->plan_promote(k, &j)) break;      /* the first entry still on disk */
            up.push_back(j);
        }
        CHECK_EQ(up.size(), expect_up);
        std::vector<char> ok(up.size(), 1);
        for (int r = 0; r < world; ++r) {
            std::vector<char> mine;
            CHECK_OK(x[(size_t)r]->issue_restore(up, compute[(size_t)r], &mine));
            for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
        }
        x[0]->report_pass(up, ok, 0);
        for (size_t i = 0; i < up.size(); ++i)
            if (ok[i]) CHECK_OK(f.pc.rebind(up[i].key, up[i].blocks));
        for (int r = 0; r < world; ++r) CHECK_OK(rad_stream_sync(compute[(size_t)r]));
        return ok;
    };

    /* ---- read, as far as the arena reaches ---- */
    fetch(10);
    CHECK_EQ(t.stats().fetched, 10);
    /* What was read is held for the restore that asked: it is neither room for the rest of its own
     * chain nor for anything else, though the disk holds every entry of it. */
    fetch(0);
    CHECK_EQ(t.make_room(10), 0);
    for (size_t i = 0; i < chain.size(); ++i)
        CHECK(t.tier_of(chain[i]) == (i < 10 ? KVTier::Host : KVTier::Disk));

    /* ---- back, as far as the pool reaches: five blocks of each group are free ---- */
    std::vector<std::vector<int32_t>> hog(cg.size());
    for (size_t k = 0; k < cg.size(); ++k) {
        const int64_t n = f.kv.free_blocks(cg[k]) - 5;
        REQUIRE_EQ(f.kv.reserve_blocks(cg[k], n, &hog[k]), n);
    }
    std::vector<char> ok = restore(20, 10);
    REQUIRE_EQ(ok.size(), 10u);
    for (size_t i = 0; i < ok.size(); ++i) CHECK_EQ((int)ok[i], i < 5 ? 1 : 0);
    CHECK_EQ(t.stats().failures, 0);
    CHECK_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 80);
    for (size_t k = 0; k < cg.size(); ++k)
        for (int32_t b : hog[k]) f.kv.release_block(cg[k], b);
    /* The rest were not touched: still read or still on disk, and still restorable. */
    const std::vector<BlockHash> rest = f.pc.restorable_from(prompt, {});
    REQUIRE_EQ(rest.size(), 15u);
    for (size_t i = 0; i < rest.size(); ++i)
        CHECK(t.tier_of(rest[i]) == (i < 5 ? KVTier::Host : KVTier::Disk));

    /* ---- and the rest, once the arena has room again ---- */
    for (int32_t s : other) t.arena().free(s);
    fetch(10);
    ok = restore(15, 15);
    REQUIRE_EQ(ok.size(), 15u);
    for (char c : ok) CHECK(c);
    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    for (int r = 0; r < world; ++r) CHECK_EQ(same_bytes(*fx[(size_t)r], r, was, hit), 20);
    CHECK_EQ(t.stats().fetched, 20);
    CHECK_EQ(t.stats().failures, 0);

    for (auto& xp : x) xp->close();
    for (RadStream s : compute) rad_stream_destroy(s);
    f.pc.set_tiers(nullptr);
}

TEST(a_chain_read_off_disk_is_restored_as_far_as_the_pool_reaches_on_every_rank) {
    run_disk_cut(1);
    run_disk_cut(2);
}

TEST(a_conversation_restored_from_host_is_byte_identical_in_new_blocks) {
    run_transfer(1, /*fragment=*/false);
    run_transfer(1, /*fragment=*/true);
}
TEST(each_rank_restores_its_own_window_of_a_shared_slot) {
    run_transfer(2, /*fragment=*/false);
    run_transfer(2, /*fragment=*/true);
}

/* A SPECULATING DEPLOYMENT KEEPS TWO PHYSICAL SLOTS A SEQUENCE, and the free list counts logical
 * ones. Subtracting one from the other reads an idle pool as half full. */
TEST(kv_bytes_in_use_count_a_speculating_state_once_per_sequence) {
    Fixture f;
    f.cfg.n_spec = 2;
    CHECK_OK(f.build(/*with_linear=*/true));
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    REQUIRE(ps != nullptr);
    CHECK_EQ(ps->state_copies, 2ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 2 * ps->bytes_per_state);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}

/* ================================================================== tree-bound linear state */

/* A TREE VERIFY READS COPY 0 AND ITS COMMIT FOLDS THE WINNER INTO COPY 0, so a group whose every
 * layer is served that way needs no scratch bank. A group merely declared that way, or bound on
 * some layers only, can still reach the anchor kernel and keeps both. The row is a launchable
 * fake: tree_state_bound asks whether a band can launch, not what it computes. */
static int tree_fake_launch(const RadArgs*, RadStream) { return RAD_OK; }
static RadKernelInfo g_tree_ki;
static KernelRow     g_tree_row;

/* hybrid_groups() with the three ops tree_state_bound requires on every GDN layer. `served` false
 * leaves every band without a row (declared, never resolved); `unserved_commit` names the one
 * layer whose winner commit resolved to nothing. The flag is set as Engine::declare sets it. */
static Program tree_program(bool tree_ops, bool served = true, int unserved_commit = -1) {
    g_tree_ki = RadKernelInfo{};
    g_tree_ki.name = "tree_fake";
    g_tree_ki.launch = tree_fake_launch;
    g_tree_row = KernelRow{ &g_tree_ki, "fake", 0, 0 };
    Program p;
    p.ops.emplace_back();                       /* the sentinel every handle table carries */
    p.buffers.emplace_back();
    p.kv_groups = hybrid_groups(/*with_linear=*/true);
    if (tree_ops)
        for (int layer : p.kv_groups[G_GDN].layers)
            for (const char* name : { "gdn_conv_tree", "gdn_recurrent_tree", "gdn_tree_commit" }) {
                OpInfo op;
                op.op = name;
                op.base.set_i("layer", layer);
                op.base.set_i("state_group", G_GDN);
                Band b;
                b.hi = 16;
                const bool commit = !std::strcmp(name, "gdn_tree_commit");
                if (served && !(commit && layer == unserved_commit))
                    b.dom[RAD_DOMAIN_DEVICE].row = &g_tree_row;
                op.bands.push_back(b);
                p.ops.push_back(op);
            }
    for (size_t g = 0; g < p.kv_groups.size(); ++g)
        p.kv_groups[g].tree_bound = tree_linear_bound(p, g);
    return p;
}

static int build_speculating(Fixture& f, const Program& p) {
    f.cfg.n_spec = 7;
    f.cfg.checkpoint_slots = 4;
    return f.build_groups(p.kv_groups);
}

/* Same --vram-kv-mib: the bank leaves the fixed costs and the paged split takes it page for page.
 * Nothing here is lent anywhere -- the pool does not grow, its paged half does. */
TEST(a_fully_bound_tree_keeps_one_state_bank_and_pages_the_other_in_the_same_pool) {
    Fixture tree, base;
    const Program tp = tree_program(true), bp = tree_program(false);
    CHECK(tp.kv_groups[G_GDN].tree_bound);
    CHECK(!tp.kv_groups[G_FULL].tree_bound);
    CHECK(!bp.kv_groups[G_GDN].tree_bound);
    REQUIRE_OK(build_speculating(tree, tp));
    REQUIRE_OK(build_speculating(base, bp));
    const KVGroupPlan* t = tree.kv.plan(G_GDN);
    const KVGroupPlan* b = base.kv.plan(G_GDN);
    REQUIRE(t != nullptr && b != nullptr);
    const int64_t seqs = tree.cfg.max_seqs, bps = t->bytes_per_state;

    CHECK_EQ(t->state_copies, 1ll);
    CHECK_EQ(t->n_states, seqs);
    CHECK_EQ(tree.kv.pool_bytes_carved(G_GDN), seqs * bps);
    CHECK_EQ(tree.kv.state_index_width(G_GDN), 1ll);
    CHECK_EQ(b->state_copies, 2ll);
    CHECK_EQ(b->n_states, 2 * seqs);
    CHECK_EQ(base.kv.pool_bytes_carved(G_GDN), 2 * seqs * bps);
    CHECK_EQ(base.kv.state_index_width(G_GDN), 6ll);

    CHECK_EQ(base.kv.fixed_bytes() - tree.kv.fixed_bytes(), seqs * bps);
    const int64_t page = tree.kv.plan(G_FULL)->bytes_per_block;
    CHECK_EQ(tree.kv.total_blocks(G_FULL) - base.kv.total_blocks(G_FULL), seqs * bps / page);
    CHECK_EQ(tree.kv.checkpoint_bytes(), base.kv.checkpoint_bytes());   /* copy 0 either way */
}

TEST(a_one_bank_tree_state_names_clears_and_reuses_only_copy_zero) {
    Fixture f;
    const Program p = tree_program(true);
    REQUIRE(p.kv_groups[G_GDN].tree_bound);
    REQUIRE_OK(build_speculating(f, p));
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    REQUIRE(ps != nullptr);
    const int64_t seqs = f.cfg.max_seqs;
    auto gdn_clears = [&f]() {
        std::vector<int32_t> out;
        for (const StateClear& c : f.kv.state_clears()) if (c.group == G_GDN) out.push_back(c.slot);
        return out;
    };

    f.kv.clear_state_clears();
    std::vector<int32_t> slot((size_t)seqs + 1, -1);
    std::vector<char> seen((size_t)seqs, 0);
    for (int64_t id = 1; id <= seqs; ++id) {
        REQUIRE_OK(f.kv.add_sequence((uint64_t)id));
        REQUIRE_OK(f.kv.ensure((uint64_t)id, 16));
        const int32_t s = f.kv.state_slot((uint64_t)id, G_GDN);
        REQUIRE(s >= 0 && s < seqs);
        CHECK(!seen[(size_t)s]);
        seen[(size_t)s] = 1;
        slot[(size_t)id] = s;
        int32_t row[6] = { -7, -7, -7, -7, -7, -7 };
        CHECK_EQ(f.kv.state_index_row((uint64_t)id, G_GDN, row, 6), 1);
        CHECK_EQ(row[0], s);
        CHECK_EQ(row[1], -7);                   /* one column: nothing past it is written */
    }
    /* One clear a sequence, of its own slot: there is no scratch copy to zero. */
    std::vector<int32_t> cl = gdn_clears();
    std::sort(cl.begin(), cl.end());
    REQUIRE_EQ((int64_t)cl.size(), seqs);
    for (int64_t i = 0; i < seqs; ++i) CHECK_EQ(cl[(size_t)i], (int32_t)i);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), seqs * ps->bytes_per_state);
    CHECK_EQ(f.kv.free_states(G_GDN), 0ll);

    /* Full at max_seqs: the bank that went away is not handed out as more sequences. */
    REQUIRE_OK(f.kv.add_sequence(100));
    CHECK_EQ(f.kv.ensure(100, 16), RAD_E_FULL);
    f.kv.free_sequence(100);

    /* A finished sequence's slot is the next one's, cleared once and named alone. */
    f.kv.clear_state_clears();
    f.kv.free_sequence(3);
    CHECK_EQ(f.kv.free_states(G_GDN), 1ll);
    REQUIRE_OK(f.kv.add_sequence(200));
    REQUIRE_OK(f.kv.ensure(200, 16));
    CHECK_EQ(f.kv.state_slot(200, G_GDN), slot[3]);
    cl = gdn_clears();
    REQUIRE_EQ(cl.size(), (size_t)1);
    CHECK_EQ(cl[0], slot[3]);
    int32_t row[1] = { -7 };
    CHECK_EQ(f.kv.state_index_row(200, G_GDN, row, 1), 1);
    CHECK_EQ(row[0], slot[3]);
    CHECK_EQ(f.kv.state_index_row(3, G_GDN, row, 1), 1);
    CHECK_EQ(row[0], -1);                       /* the freed sequence names nothing */
}

TEST(a_partial_declared_only_or_absent_tree_binding_keeps_both_banks) {
    const Program cases[] = {
        tree_program(true, /*served=*/true, /*unserved_commit=*/4),
        tree_program(true, /*served=*/false),
        tree_program(false),
    };
    for (const Program& p : cases) {
        CHECK(!p.kv_groups[G_GDN].tree_bound);
        Fixture f;
        REQUIRE_OK(build_speculating(f, p));
        const KVGroupPlan* ps = f.kv.plan(G_GDN);
        REQUIRE(ps != nullptr);
        const int64_t seqs = f.cfg.max_seqs;
        CHECK_EQ(ps->state_copies, 2ll);
        CHECK_EQ(ps->n_states, 2 * seqs);
        CHECK_EQ(f.kv.state_index_width(G_GDN), 6ll);

        f.kv.clear_state_clears();
        REQUIRE_OK(f.kv.add_sequence(1));
        REQUIRE_OK(f.kv.ensure(1, 16));
        const int32_t s = f.kv.state_slot(1, G_GDN);
        REQUIRE(s >= 0 && s < seqs);
        int32_t row[6] = {};
        CHECK_EQ(f.kv.state_index_row(1, G_GDN, row, 6), 6);
        for (int j = 0; j < 6; ++j) CHECK_EQ(row[j], (int32_t)(s + (j < 3 ? 0 : seqs)));
        std::vector<int32_t> cl;
        for (const StateClear& c : f.kv.state_clears()) if (c.group == G_GDN) cl.push_back(c.slot);
        REQUIRE_EQ(cl.size(), (size_t)2);
        CHECK_EQ(cl[0], s);
        CHECK_EQ(cl[1], (int32_t)(s + seqs));
        CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 2 * ps->bytes_per_state);
    }

    /* The QSA tail is bound by its persistent projection, not by served ops: tree_state_bound
     * says yes and the one-bank flag still says no -- the manager's own tail rule sizes it. */
    Program q = tree_program(false);
    q.kv_groups[G_GDN].name = "kv_qsa_tail";
    for (int layer : q.kv_groups[G_GDN].layers) {
        BufferInfo bi;
        bi.name = "blk." + std::to_string(layer) + ".qsa_tree_qk";
        bi.decl.kind = RAD_BUF_PERSIST;
        q.buffers.push_back(bi);
    }
    CHECK(tree_state_bound(q, G_GDN));
    CHECK(!tree_linear_bound(q, G_GDN));
}
