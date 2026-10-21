/* The all-reduce shims, and the init/fini that make them possible.
 *
 * THIS IS WHY init/fini EXIST IN THE ABI (spec §2.1). libr4d's all-reduces handshake through peer
 * signal buffers and an IPC scratch region that has to be allocated on every rank, published to
 * every other rank, and made peer-accessible BEFORE the first launch. None of that can happen on
 * the step path -- it allocates, it synchronises, and it needs every rank present.
 *
 * THE EXCHANGE IS NOT IPC HERE, AND THAT IS THE POINT. A one-process-per-rank layout has no
 * choice: hipIpcGetMemHandle on each rank, a collective gather of the 64-byte handles,
 * hipIpcOpenMemHandle on the other side. Radiance runs ONE PROCESS with one thread per rank sharing
 * an address space (spec §1). A device pointer allocated on rank 0's card is already a valid
 * address on rank 1's card once hipDeviceEnablePeerAccess has run -- that is what unified virtual
 * addressing means within a process -- so the "handle exchange" degenerates to writing a pointer
 * into a shared array and waiting on a barrier. The single-process path (`r4d_ar_ipc_enable_peer`)
 * IS the runtime path.
 *
 * What that buys, concretely: no 64-byte handle marshalling, no hipIpcOpenMemHandle (which maps a
 * second VA range for the same physical pages and has to be closed in the right order at
 * shutdown), and no serialisation layer between ranks at all -- which is most of what a Python
 * engine spends its multiprocessing budget on.
 *
 * THE BARRIER ASSUMES EVERY RANK RUNS THE SAME DECLARE. It does: rad_arch_declare runs once per
 * rank with the same architecture plugin and the same model, so the Nth all-reduce instance on
 * rank 0 is the Nth on every other rank. The rendezvous is therefore keyed on nothing but arrival
 * order, and a rank that declares a different graph deadlocks at startup with a 60-second timeout
 * and a named error rather than corrupting a later step.
 */

#include "r4d_args.h"
#include "r4d.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <vector>
#include <chrono>

static_assert(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS == 20971520,
              "r4d_rows.cpp restates this bound; libr4d changed it");
static_assert(R4D_MAX_RANKS == 8, "the 8-rank entry point names exactly seven peers");

/* ------------------------------------------------------------------ throw conversion
 *
 * The all-reduce family is the half of libr4d that THROWS instead of returning. Errors are values
 * here, so every call is wrapped. The shim pre-checks everything
 * the entry point throws about, so a throw that still escapes is a library or device fault rather
 * than a bad call -- which is why it becomes RAD_E_DEVICE and prints once with the text. */
static int r4d_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return RAD_E_DEVICE;
}

#define R4D_TRY(who, ...)                                                       \
    do {                                                                        \
        try { __VA_ARGS__; }                                                    \
        catch (const std::exception& e_) { return r4d_caught(who, e_.what()); } \
        catch (...) { return r4d_caught(who, "unknown exception"); }            \
    } while (0)

/* ------------------------------------------------------------------ the rendezvous */

namespace {

struct Rendezvous {
    std::mutex              m;
    std::condition_variable cv;
    int  arrived = 0;
    int  epoch   = 0;
    long scratch[R4D_MAX_RANKS] = { 0 };
    long flags  [R4D_MAX_RANKS] = { 0 };
    long seq    [R4D_MAX_RANKS] = { 0 };
    int  dev    [R4D_MAX_RANKS] = { 0 };
    /* Whether that rank's scratch and flags came out of the FINE-GRAINED allocation, published
     * with the pointers above so every rank can decide the same thing about a launch's drain. */
    int  fine   [R4D_MAX_RANKS] = { 0 };
    int  failed = 0;
};

Rendezvous g_rv;

/* Sense-reversing barrier on a monotone epoch, so a rank that wakes and immediately re-enters for
 * the next instance cannot be counted twice. The timeout is not a recovery path -- a startup
 * deadlock is fatal either way -- it is there so the failure is a named error at declare instead
 * of a hung process with no output. */
int rv_barrier(int world) {
    std::unique_lock<std::mutex> lk(g_rv.m);
    const int e = g_rv.epoch;
    if (++g_rv.arrived == world) {
        g_rv.arrived = 0;
        g_rv.epoch++;
        g_rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = g_rv.cv.wait_for(lk, std::chrono::seconds(60),
                                     [&] { return g_rv.epoch != e; });
    return ok ? RAD_OK : RAD_E_STATE;
}

/* One resolved instance's view of the world: its own buffers and its peers', the latter in
 * ASCENDING GLOBAL RANK WITH THIS RANK REMOVED, which is the order libr4d's wide entry points
 * take their pointer arguments in. */
struct ArInst {
    int  rank = 0, world = 0, dev = 0;
    long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;
    long peer_scratch[R4D_MAX_PEERS] = { 0 };
    long peer_flags  [R4D_MAX_PEERS] = { 0 };
    long slot_bytes = 0;      /* per-slot capacity, 16-byte aligned                  */
    long slot16     = 0;      /* the same in 16-byte words, which is what most take  */
    long max_numel  = 0;      /* the declared bound the sizing was computed from     */
    int  esize      = 2;
    int  dtype      = 0;      /* libr4d code: 0 bf16, 1 fp16, 2 fp32                 */
    int  max_blocks = 0;      /* what the flag array was sized for -- a hard bound    */
    /* What a LAUNCH is allowed to ask for, which is a much smaller number and a different kind of
     * number. max_blocks is a capacity; this is a tuning. Every block drives its own flag
     * handshake, so block count buys parallel pushes up to the point the link saturates and after
     * that buys nothing but round trips: at a 2048-token prefill chunk the capacity rule asks for
     * 936 blocks and the tuning rule asks for 24. */
    int  nb_cap     = 0;
    /* Fine-grained scratch and flags on every rank permit drain=3. Otherwise the fused
     * exact-wire launches retain their previous full system fence (drain=1). */
    int  all_fine   = 0;
};

/* A parameter that may have been declared as a RANGE: init is handed the frozen geometry for one
 * band, and the band's UPPER bound is what a buffer has to be sized for. */
long long pgeti_hi(const RadParam* p, int n_p, const char* k, long long dflt) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return p[i].ihi;
    }
    return dflt;
}

long align16(long v) { return (v + 15) & ~15L; }

/* Build the world. `max_numel` is the largest message any launch on this instance will carry.
 *
 * ONE SIZING RULE FOR THE WHOLE FAMILY, which is also what libr4d's own driver did ("one set
 * serves every kernel in the family"): a slot holds a full message. The one-shots need exactly
 * that; the two-shot's two regions are 2 * (message / world_size), which is smaller for every
 * world >= 2; and the tiered-int8 wire's two images are 2 * ceil(shard/32) * 38 bytes, smaller
 * again. Sizing for the maximum costs the difference in VRAM and removes a per-row branch that
 * would otherwise have to reconstruct which kernel this instance is -- RadInitFn is not told.
 * That is the one thing missing from the hook: init knows the geometry but not the row.
 *
 * THE COST, NAMED: a slot holds a full message, so at a 4096-token prefill chunk on a 5120-wide
 * residual that is 40 MiB a slot and 80 MiB for the pair. ONE set serves every op that resolved
 * to the same (rank, world, dtype, bound) -- see the cache below, which is what keeps a 64-layer
 * model from building 128 of these. */
int build(ArInst* in, int rank, int world, long long max_numel, int dtype_code) {
    if (world < 2 || world > R4D_MAX_RANKS) return RAD_E_SHAPE;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    if (max_numel <= 0) return RAD_E_INVAL;

    in->rank = rank;
    in->world = world;
    in->dtype = dtype_code;
    in->esize = r4d_ar_esize(dtype_code);
    in->max_numel = (long)max_numel;
    in->slot_bytes = align16((long)max_numel * in->esize);
    in->slot16 = in->slot_bytes / 16;

    int maxb = 0, nb_design = 0, max_peers = 0;
    if (world == 2) {
        maxb = r4d_ar_max_blocks();
    } else {
        r4d_ar_wide_dims(&maxb, &nb_design, &max_peers);
        if (world - 1 > max_peers) return RAD_E_SHAPE;
    }
    if (maxb <= 0) return RAD_E_DEVICE;
    in->max_blocks = maxb;
    /* The launch ceilings are the library driver's (radiance_allreduce.py): 24 at two
     * ranks, and at width the library's own design count, which is what that file asks for by
     * name. The compressed 2-rank wire runs to 48 and takes it at its own call site -- it moves
     * three eighths of the bytes, so it saturates the link with more blocks, not fewer. */
    in->nb_cap = (world == 2) ? 24 : (nb_design > 0 ? nb_design : maxb);
    if (in->nb_cap > maxb) in->nb_cap = maxb;

    /* Slots: one per (parity, source rank) at width; at world 2 parity alone names the single
     * remote writer, so two. Flags: one word per (block, source rank) at width, one per block at
     * world 2. Sequence counters: a handshake word and a parity word per block of capacity
     * (r4d_ar_seq.h), device-resident so a captured graph replays. */
    const int  nslots     = (world == 2) ? 2 : 2 * world;
    const long flag_words = (long)maxb * ((world == 2) ? 1 : world);

    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    long sc = 0, fl = 0, sq = 0, lp = 0;
    int  dev = 0;

    /* FINE-GRAINED: the scratch is written by the peer and read here without going through L2, so
     * a coarse-grained allocation would leave the peer's bytes in a cache this rank never
     * invalidates. libr4d's driver falls back to coarse when fine fails; so does this, because a
     * deterministic fallback stays symmetric across ranks (same environment, same hardware). */
    int fine_sc = 1, fine_fl = 1, fine_sq = 1;
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        in->dev = dev;
        sc = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 1, handle, &hlen);
        if (!sc) { fine_sc = 0; sc = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 0, handle, &hlen); }
        fl = r4d_ar_ipc_alloc(flag_words * 4, 1, handle, &hlen);
        if (!fl) { fine_fl = 0; fl = r4d_ar_ipc_alloc(flag_words * 4, 0, handle, &hlen); }
        sq = r4d_ar_ipc_alloc(2L * maxb * 4, 1, handle, &hlen);
        if (!sq) { fine_sq = 0; sq = r4d_ar_ipc_alloc(2L * maxb * 4, 0, handle, &hlen); });
    if (!sc || !fl || !sq) return RAD_E_NOMEM;

    /* SAY IT, because the fallback above changes what the KERNEL's fence settings mean. The
     * launches run acq=0 -- no acquire fence -- on the argument that an uncached read is already
     * fresh, and that argument holds for FINE-GRAINED scratch and for nothing else: on a coarse
     * allocation the peer's bytes can sit in an L2 this rank never invalidates, and the reduce
     * then folds a stale slot. That failure is not a crash and not a wrong shape. It is a sum
     * that is occasionally a little wrong, which at greedy decode is a completion that differs
     * between runs of the same prompt. Silence here would make that unfindable. */
    if (!fine_sc || !fine_fl || !fine_sq)
        std::fprintf(stderr, "[r4d] WARNING: all-reduce scratch is COARSE-GRAINED "
                             "(scratch %s, flags %s, seq %s). The kernels run without an acquire "
                             "fence, which is only sound on fine-grained memory -- expect "
                             "occasional stale peer data.\n",
                     fine_sc ? "fine" : "COARSE", fine_fl ? "fine" : "COARSE",
                     fine_sq ? "fine" : "COARSE");

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce has to fold exactly
     * the bytes it sent, not a re-derivation of them. The sizing is libr4d's driver's, transcribed
     * -- payload at 6 bits an element is 3/8 of the bf16 message and the bf16 scale per group of
     * 64 is 1/64 of it, so half the message plus a thirty-second plus a page is generous by
     * construction and cheap enough not to tighten. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            lp = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 32 + 4096, 0,
                                  handle, &hlen));
    }

    in->my_scratch = sc;
    in->my_flags   = fl;
    in->my_seq     = sq;
    in->locpack    = lp;

    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        g_rv.scratch[rank] = sc;
        g_rv.flags[rank]   = fl;
        g_rv.seq[rank]     = sq;
        g_rv.dev[rank]     = dev;
        /* Publish allocation type with the pointers; the next barrier makes both visible. */
        g_rv.fine[rank]    = (fine_sc && fine_fl);
    }
    int rc = rv_barrier(world);
    if (rc != RAD_OK) return rc;

    /* The second barrier below prevents the next instance from overwriting this epoch. */
    in->all_fine = 1;
    for (int r = 0; r < world; ++r)
        if (!g_rv.fine[r]) in->all_fine = 0;

    /* Peer access is a property of THIS rank's context, so every rank enables its own. It has to
     * happen after the publish barrier only because that is where the peer device ordinals are
     * known -- a host can expose more HIP devices than there are ranks (an integrated part among
     * them), so the ordinal is read from the peer rather than assumed to equal the rank. */
    int j = 0;
    for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        if (!g_rv.scratch[r] || !g_rv.flags[r]) return RAD_E_STATE;
        in->peer_scratch[j] = g_rv.scratch[r];
        in->peer_flags[j]   = g_rv.flags[r];
        ++j;
        R4D_TRY("ar_ipc_enable_peer", r4d_ar_ipc_enable_peer((long)g_rv.dev[r]));
    }

    /* Second barrier: no rank may launch until every rank has enabled peer access, or the first
     * message would be pushed into a mapping the receiver has not opened. */
    return rv_barrier(world);
}

/* ------------------------------------------------------------------ one world, shared
 *
 * ONE INSTANCE PER (rank, world, dtype, message bound), NOT ONE PER OP. The core calls init once
 * per resolved instance -- one op, one band, one domain -- and a 64-layer tensor-parallel model
 * declares 128 all-reduces with identical parameters. One peer scratch per instance would be
 * ruinous, because the scratch is sized by the LARGEST message the band can carry: at a 2048-token
 * chunk on a 5120-wide residual that is 21 MiB a slot, two slots, plus the rotated wire's packed
 * copy, so about 53 MiB per instance and 6.7 GiB a card for a graph that needs 53 MiB. It is
 * allocated straight from hipMalloc rather than out of a radiance pool, so it is invisible to the
 * VRAM budget and shows up instead as the KV cache refusing to fit, or as a large prefill chunk
 * refusing to start at all.
 *
 * SHARING IS SAFE AND IS NOT AN OPTIMISATION OF THE HANDSHAKE. The slots are double-buffered by
 * the parity of the LAUNCH on this instance and every all-reduce on a rank rides one stream in
 * declared order, so all-reduce k+2 -- the next user of k's slot -- cannot launch until k+1
 * completed, which required the peer's k+1 flag, which the peer sets only after its own k+1
 * kernel started, which is after its k kernel finished reading that slot. The same argument is
 * why two slots suffice. It needs the parity to be the launch's and not a block's: the ops that
 * share an instance run different block counts, so per-block counts drift apart, and a block's
 * own parity would let launch k+1 push into bytes a different block of launch k is still reading
 * (r4d_ar_seq.h).
 *
 * The rendezvous stays symmetric because every rank declares the same graph in the same order
 * (rad_arch_declare runs once per rank against one plugin): the Nth all-reduce misses the cache
 * on every rank and the (N+1)th hits it on every rank, so a barrier is either entered by
 * everybody or by nobody. */
struct CacheKey {
    int       rank, world, dtype;
    long long numel;
    bool operator==(const CacheKey& o) const {
        return rank == o.rank && world == o.world && dtype == o.dtype && numel == o.numel;
    }
};

struct CacheSlot {
    CacheKey key;
    ArInst*  in   = nullptr;
    int      refs = 0;
};

std::mutex             g_cache_mu;
std::vector<CacheSlot> g_cache;

int acquire(const CacheKey& k, ArInst** out) {
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (CacheSlot& s : g_cache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    /* Built OUTSIDE the lock: build() rendezvouses with the other ranks and holding a mutex
     * across that barrier is a deadlock with the peer's own lookup. One declare thread per rank
     * and a rank-keyed entry means no two builders race for the same key. */
    ArInst* in = new (std::nothrow) ArInst();
    if (!in) return RAD_E_NOMEM;
    int rc = build(in, k.rank, k.world, k.numel, k.dtype);
    if (rc != RAD_OK) { delete in; return rc; }
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        g_cache.push_back(CacheSlot{ k, in, 1 });
    }
    *out = in;
    return RAD_OK;
}

void release(ArInst* in) {
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (size_t i = 0; i < g_cache.size(); ++i) {
            if (g_cache[i].in != in) continue;
            if (--g_cache[i].refs > 0) return;
            g_cache.erase(g_cache.begin() + (long)i);
            break;
        }
    }
    /* Only this rank's own allocations. A peer pointer is the peer's to free, and freeing it here
     * would be a double free the moment two ranks tear down in either order. */
    int orig_dev = -1;
    if (hipGetDevice(&orig_dev) == hipSuccess && orig_dev != in->dev) {
        (void)hipSetDevice(in->dev);
    }
    r4d_ar_ipc_free(in->my_scratch);
    r4d_ar_ipc_free(in->my_flags);
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    if (orig_dev >= 0 && orig_dev != in->dev) {
        (void)hipSetDevice(orig_dev);
    }
    delete in;
}

}  /* namespace */

/* ------------------------------------------------------------------ init / fini */

extern "C" int r4d_rad_ar_init(const RadParam* p, int n_p, int rank, int world,
                               void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    /* world_size is both an ABI argument and a declared parameter, and they must agree: the
     * parameter is what the constraint matched on and the argument is what the engine will
     * actually run, so a disagreement means the graph and the process do not describe the same
     * deployment. Refusing here is cheaper than finding out at the first handshake. */
    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;

    const int dt = r4d_ar_dtype_code(r4d_pgets(p, n_p, "dtype"));
    if (dt < 0) return RAD_E_DTYPE;

    const long long numel = pgeti_hi(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    int rc = acquire(CacheKey{ rank, world, dt, numel }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

/* The fused all-reduce + norm declares `M`(range) and `n` rather than `numel`: its message is the
 * post-projection activation. Same world, same buffers, different arithmetic to get the bound. */
static int ar_mn_acquire(const RadParam* p, int n_p, int rank, int world, void** out) {
    const long long M = pgeti_hi(p, n_p, "M", 0);
    const long long K = pgeti_hi(p, n_p, "n", 0);
    if (M <= 0 || K <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    int rc = acquire(CacheKey{ rank, world, 0 /* bf16 */, M * K }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_ln_init(const RadParam* p, int n_p, int rank, int world,
                                  void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 2) return RAD_E_SHAPE;   /* the handshake is one peer flag per block */

    return ar_mn_acquire(p, n_p, rank, world, out);
}

/* The four-rank ar_hc_write and ar_gather_hc_write. The key is the family's -- {rank, 4, bf16,
 * M_hi * n} -- so the instance is sized for the most rows the op was declared for, and it is the
 * same instance as a standalone all_reduce band's only where the two bounds agree. Either way is
 * safe on one stream: every wide kernel takes its flag value from the instance's seq counters,
 * advances them once per all-reduce and writes only the (parity, source) slot that value names
 * (r4d_ar_wide.h), so ops sharing an instance interleave in any order every rank issues them in,
 * and ops on separate instances touch nothing of each other's. Exact wire only: the rows constrain
 * `wire` to 0, and a `wire` 6 that reaches this init anyway is refused. */
extern "C" int r4d_rad_ar_hc_wide_init(const RadParam* p, int n_p, int rank, int world,
                                       void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 4) return RAD_E_SHAPE;   /* the peer bundle is compiled for four */
    if (pgeti_hi(p, n_p, "wire", 0) != 0) return RAD_E_UNSUPPORTED;

    return ar_mn_acquire(p, n_p, rank, world, out);
}

extern "C" void r4d_rad_ar_fini(void* instance) {
    /* One fini per successful init, and the instance is shared across every op that resolved to
     * the same world: the last release frees it. */
    release((ArInst*)instance);
}

/* ------------------------------------------------------------------ launch helpers */

enum { AR_X = 0, AR_Y };

struct ArCall {
    const ArInst* in;
    long inp, out;
    long numel;
    int  nblocks, nthreads, drain, acq, pub;
};

/* Block count. libr4d takes it as an argument because PCIe saturates with very few blocks and
 * everything above that only helps the local reduce, so the right number scales with the message
 * and then stops. 1400 sixteen-byte words per block, floor 4, ceiling 24, is the driver's
 * rule; a tuned `nblocks` that names a fixed count overrides it. */
static int nb_for(long words16, int max_nb) {
    long nb = words16 / 1400;
    if (nb < 4) nb = 4;
    if (nb > max_nb) nb = max_nb;
    if (nb > words16) nb = words16 > 0 ? words16 : 1;
    return (int)nb;
}

/* Shared front half of every all-reduce launch: find the instance, check the message against what
 * init sized for, resolve the tuned point. Every kernel of the family runs in place (`y` absent or
 * equal to `x`); each says why at its entry. */
static int ar_setup(const RadArgs* a, ArCall* c, int shard_nb) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    const RadTensor* x = r4d_opt(a, AR_X);
    if (!x) return RAD_E_INVAL;
    if (!r4d_contig(x)) return RAD_E_STRIDE;

    const RadTensor* y = r4d_opt(a, AR_Y);
    if (y) {
        if (!r4d_contig(y)) return RAD_E_STRIDE;
        if (r4d_numel(y) != r4d_numel(x)) return RAD_E_SHAPE;
    }

    const long n = (long)r4d_numel(x);
    if (n <= 0) return RAD_E_SHAPE;
    if (n > in->max_numel) return RAD_E_SHAPE;    /* the peer scratch was sized for max_numel */

    /* The push is a 16-byte vector store and libr4d throws on a message that is not a whole
     * number of them; checked here so the throw never happens. */
    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16)) return RAD_E_ALIGN;
    if (y && !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const R4dArCfg* cfg = &cfgv;

    c->in    = in;
    c->inp   = r4d_addr(x);
    c->out   = y ? r4d_addr(y) : r4d_addr(x);
    c->numel = n;
    c->nthreads = cfg->nthreads;
    c->drain    = cfg->drain;
    c->acq      = cfg->acq;
    c->pub      = cfg->pub;

    const long words16 = (long)n * in->esize / 16;
    const long basis   = shard_nb ? words16 / in->world : words16;
    /* The two-shots move SHARDS, so they take both the smaller basis and the tighter ceiling: the
     * full-message rule over-launches the whole mid band. */
    int cap = in->nb_cap;
    if (shard_nb && cap > 24) cap = 24;
    c->nblocks = cfg->nblocks > 0 ? cfg->nblocks : nb_for(basis, cap);
    if (c->nblocks > in->max_blocks) c->nblocks = in->max_blocks;
    return RAD_OK;
}

/* ------------------------------------------------------------------ 2-rank all-gather */
/* NOT ar_setup: every all-reduce there requires numel(y) == numel(x), and an all-gather's output
 * is world_size times its input. The instance, the scratch and the flags are the all-reduce's --
 * a slot holds a full message and this message is the INPUT, so r4d_rad_ar_init sizes it right
 * from the same `numel` parameter. */
extern "C" int r4d_rad_ar_gather_2rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    if (in->world != 2) return RAD_E_SHAPE;       /* the handshake is one peer flag, not a tree */

    const RadTensor* x = r4d_opt(a, AR_X);
    const RadTensor* y = r4d_opt(a, AR_Y);
    if (!x || !y) return RAD_E_INVAL;             /* the gather has no in-place form */
    if (!r4d_contig(x) || !r4d_contig(y)) return RAD_E_STRIDE;
    if (y->data == x->data) return RAD_E_INVAL;

    const long n = (long)r4d_numel(x);
    if (n <= 0 || n > in->max_numel) return RAD_E_SHAPE;
    /* "y is [world_size * numel] with rank r's contribution at offset r * numel" (docs/OPS.md).
     * A y of any other extent is a caller that means something this kernel does not do. */
    if ((long long)r4d_numel(y) != (long long)n * in->world) return RAD_E_SHAPE;

    /* The row width, if the caller asked for the row-interleaved placement. It has to divide the
     * message and be a whole number of 16-byte words, both of which libr4d rechecks. */
    const long long row = r4d_pgeti_or(a->p, a->n_p, "row", 0);
    if (row < 0 || (row > 0 && (n % row || (row * in->esize) % 16))) return RAD_E_SHAPE;

    /* The 2-rank gather moves 16-byte vectors plus up to three 4-byte tail words. */
    if (((long long)n * in->esize) % 4) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const R4dArCfg* cfg = &cfgv;
    const long words16 = (long)n * in->esize / 16;
    int nblocks = cfg->nblocks > 0 ? cfg->nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    R4D_TRY("ar_gather_2rank",
        r4d_ar_gather_2rank(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                            in->my_seq, in->slot16, r4d_addr(x), r4d_addr(y), n, in->dtype,
                            in->rank, (long)row, (long)(uintptr_t)s, nblocks, cfg->nthreads,
                            cfg->drain, cfg->acq));
    return RAD_OK;
}

/* ------------------------------------------------------------------ TP=2 logit gathering & argmax
 * Helpers for FullDrafter split vocabulary (V/2 = 124160 columns per GPU) on dual R9700. */

/**
 * Gathers split vocabulary logits from Rank 0 and Rank 1 into full vocabulary logits.
 * local_logits: [M, 124160] (M rows of 124160 floats)
 * full_logits:  [M, 248320] (M rows of 248320 floats, Rank 0 followed by Rank 1)
 */
extern "C" int r4d_rad_ar_gather_logits_tp2(
    const void* local_logits,
    void* full_logits,
    int M,
    int rank,
    void* instance,
    RadStream s
) {
    if (!local_logits || !full_logits || M <= 0 || (rank != 0 && rank != 1) || !instance) return RAD_E_INVAL;
    const ArInst* in = (const ArInst*)instance;
    if (in->world != 2) return RAD_E_INVAL;

    const long shard_vocab = 124160;
    const long n = (long)M * shard_vocab;
    const long esize = sizeof(float);
    constexpr int dtype_fp32 = 2; // libr4d code: 0 bf16, 1 fp16, 2 fp32

    const long words16 = n * esize / 16;
    int nblocks = nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    R4D_TRY("ar_gather_2rank_logits",
        r4d_ar_gather_2rank(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                            in->my_seq, in->slot16, (long)(uintptr_t)local_logits,
                            (long)(uintptr_t)full_logits, n, dtype_fp32,
                            rank, (long)shard_vocab, (long)(uintptr_t)s, nblocks, 256,
                            0, 0));
    return RAD_OK;
}

__global__ void r4d_cross_gpu_argmax_tp2_kernel(
    const uint32_t* __restrict__ cand_rank0,
    const float*    __restrict__ val_rank0,
    const uint32_t* __restrict__ cand_rank1,
    const float*    __restrict__ val_rank1,
    uint32_t*       __restrict__ out_tokens,
    int M
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) {
        const float v0 = val_rank0[m];
        const float v1 = val_rank1[m];
        const uint32_t id0 = cand_rank0[m];
        const uint32_t id1 = cand_rank1[m];
        if (v1 > v0 || (v1 == v0 && id1 < id0)) {
            out_tokens[m] = id1;
        } else {
            out_tokens[m] = id0;
        }
    }
}

/**
 * Cross-GPU argmax over dual-GPU split logits shards under TP=2.
 * Gathers candidate pairs (id, val) from Rank 0 and Rank 1 and selects the winning token ID.
 * Supports both device buffers (via GPU kernel) and host buffers.
 */
extern "C" int r4d_rad_ar_cross_gpu_argmax_tp2(
    const uint32_t* cand_rank0,    /* [M] winning candidate ID on Rank 0 */
    const float* val_rank0,        /* [M] max logit on Rank 0 */
    const uint32_t* cand_rank1,    /* [M] winning candidate ID on Rank 1 */
    const float* val_rank1,        /* [M] max logit on Rank 1 */
    uint32_t* out_tokens,          /* [M] winning token IDs */
    int M
) {
    if (!cand_rank0 || !val_rank0 || !cand_rank1 || !val_rank1 || !out_tokens || M <= 0) {
        return RAD_E_INVAL;
    }

    hipPointerAttribute_t attr{};
    bool is_device = false;
    if (hipPointerGetAttributes(&attr, cand_rank0) == hipSuccess && attr.type == hipMemoryTypeDevice) {
        is_device = true;
    } else {
        (void)hipGetLastError();
    }

    if (is_device) {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        r4d_cross_gpu_argmax_tp2_kernel<<<blocks, threads>>>(
            cand_rank0, val_rank0, cand_rank1, val_rank1, out_tokens, M
        );
        hipError_t err = hipGetLastError();
        return (err == hipSuccess) ? RAD_OK : RAD_E_DEVICE;
    }

    for (int m = 0; m < M; ++m) {
        const float v0 = val_rank0[m];
        const float v1 = val_rank1[m];
        const uint32_t id0 = cand_rank0[m];
        const uint32_t id1 = cand_rank1[m];
        if (v1 > v0 || (v1 == v0 && id1 < id0)) {
            out_tokens[m] = id1;
        } else {
            out_tokens[m] = id0;
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ wide (3 / 4 / 8 rank) all-gather
 * The gather the 2-rank one cannot be at width: its handshake is one peer flag per block, so the
 * wide form takes the wide family's peer arrays and r4d_ar_handshake instead (r4d_ar_wide.h), and
 * it rides the SAME instance, scratch, flags and counters as the wide one-shots above -- a slot
 * holds a full message here exactly as it does there, and this message is the INPUT. The two
 * launchers live in r4d_ar_gather_Nrank.hip, which is the only family unit whose caller is inside
 * the plugin, so they are declared here rather than in r4d.h. */
extern "C" void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long inp, long out, long n_elem,
                                    long dtype, long rank, long row_elems, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub);
extern "C" void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                    long peer_scratch6, long peer_flags0, long peer_flags1,
                                    long peer_flags2, long peer_flags3, long peer_flags4,
                                    long peer_flags5, long peer_flags6, long my_scratch,
                                    long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                    long out, long n_elem, long dtype, long rank, long row_elems,
                                    long stream, long nblocks, long nthreads, long drain, long acq,
                                    long pub);

/* Every check the 2-rank gather makes, and the same `row` rule: an all-gather's output is
 * world_size times its input, so this is not ar_setup either. */
static int ar_gather_wide_setup(const RadArgs* a, int ws, const ArInst** out_in, ArCall* c,
                                long long* row_out) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    if (in->world != ws) return RAD_E_SHAPE;      /* the peer bundle is compiled for this width */

    const RadTensor* x = r4d_opt(a, AR_X);
    const RadTensor* y = r4d_opt(a, AR_Y);
    if (!x || !y) return RAD_E_INVAL;             /* the gather has no in-place form */
    if (!r4d_contig(x) || !r4d_contig(y)) return RAD_E_STRIDE;
    if (y->data == x->data) return RAD_E_INVAL;

    const long n = (long)r4d_numel(x);
    if (n <= 0 || n > in->max_numel) return RAD_E_SHAPE;
    /* "y is [world_size * numel] with rank r's contribution at offset r * numel" (docs/OPS.md).
     * A y of any other extent is a caller that means something this kernel does not do. */
    if ((long long)r4d_numel(y) != (long long)n * in->world) return RAD_E_SHAPE;

    /* The row width, if the caller asked for the row-interleaved placement. It has to divide the
     * message and be a whole number of 16-byte words, both of which libr4d rechecks. */
    const long long row = r4d_pgeti_or(a->p, a->n_p, "row", 0);
    if (row < 0 || (row > 0 && (n % row || (row * in->esize) % 16))) return RAD_E_SHAPE;

    /* 16-byte vector stores, and libr4d throws on a message that is not a whole number of them. */
    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)n * in->esize / 16;
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    *out_in = in;
    c->in = in;
    c->inp = r4d_addr(x);
    c->out = r4d_addr(y);
    c->numel = n;
    c->nblocks = nblocks;
    c->nthreads = cfgv.nthreads;
    c->drain = cfgv.drain;
    c->acq = cfgv.acq;
    c->pub = cfgv.pub;
    *row_out = row;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_3rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0;
    int rc = ar_gather_wide_setup(a, 3, &in, &c, &row);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_3rank",
        r4d_ar_gather_3rank(in->peer_scratch[0], in->peer_scratch[1],
                            in->peer_flags[0], in->peer_flags[1],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0;
    int rc = ar_gather_wide_setup(a, 4, &in, &c, &row);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_4rank",
        r4d_ar_gather_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_8rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0;
    int rc = ar_gather_wide_setup(a, 8, &in, &c, &row);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_8rank",
        r4d_ar_gather_8rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                            in->peer_scratch[6],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                            in->peer_flags[6],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub));
    return RAD_OK;
}

/* ------------------------------------------------------------------ 2-rank one-shot */

extern "C" int r4d_rad_ar_oneshot_2rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    R4D_TRY("ar_oneshot_2rank_exact",
        r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                   c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                   c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq));
    return RAD_OK;
}

/* Where the compressed wire stops paying. A decode step's all-reduce is eight rows of the residual
 * stream, 80 KiB, and the exact kernel already moves that at about 92% of what the link delivers
 * one way -- the message is on the wire, not in latency, and there is nothing for a 2.4x smaller
 * payload to recover against the rotate, the pack and the unpack. A prefill chunk is forty times
 * that and squarely bandwidth-bound. So the row is chosen once at declare and the WIRE is chosen
 * per launch. */
enum { R4D_AR_WHT6_MIN_BYTES = 128 * 1024 };

/* The compressed path's own ceiling: it moves three eighths of the bytes, so it wants more blocks
 * to saturate the same link, not fewer. */
enum { R4D_AR_WHT6_NB_CAP = 48 };

/* THE ROTATED WIRE'S FUSED WRITES RUN 512 THREADS A BLOCK, not the axis's 1024. Their waves are
 * latency-bound on the reads behind the message -- the routed rows of the gather, the streams of
 * the write -- and at a 2048-row chunk half the waves a block, twice as many blocks resident,
 * is 5-9% faster per call (gather 491 us against 536, write 317 against 329) where 96 or 192
 * blocks of 1024 do not help. */
enum { R4D_AR_WHT6_FUSED_THREADS = 512 };

extern "C" int r4d_rad_ar_oneshot_2rank_wht6(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;      /* the rotated wire is bf16/fp16 only */
    if (!c.in->locpack) return RAD_E_STATE;

    int group = 0, bits = 0, chunk = 0;
    r4d_ar_wht6_dims(&group, &bits, &chunk);

    /* THE SIZE GATE. Below the floor -- or on a message the rotation's group does not tile -- this
     * row serves the message with the EXACT kernel instead. That is safe and it is not a
     * workaround: the two kernels share this instance's scratch, flags and sequence counters and
     * publish in one flag value space, so a stream that mixes them is protocol-safe, which is the
     * property libr4d's driver relies on to make the same choice per call. A caller that needs the
     * rotated wire unconditionally asks for a floor of 0. */
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    const long long msg_b   = (long long)c.numel * c.in->esize;
    if (group <= 0 || c.numel % group || msg_b < floor_b) {
        R4D_TRY("ar_oneshot_2rank_exact",
            r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                       c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                       c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                       (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq));
        return RAD_OK;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const R4dArCfg* cfg = &cfgv;
    if (cfg && cfg->nblocks <= 0) {
        int nb = nb_for((long)msg_b / 16, R4D_AR_WHT6_NB_CAP);
        if (nb > c.in->max_blocks) nb = c.in->max_blocks;
        c.nblocks = nb;
    }

    R4D_TRY("ar_oneshot_2rank_wht6",
        r4d_ar_oneshot_2rank_wht6(c.in->peer_scratch[0], c.in->my_scratch,
                                  c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                  c.in->locpack, c.in->slot_bytes, c.in->slot_bytes / 2,
                                  c.inp, c.out, c.numel, c.in->dtype,
                                  (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq));
    return RAD_OK;
}

/* ------------------------------------------------------------------ wide (3 / 4 / 8 rank)
 *
 * The peer arguments are ascending global rank with the current rank removed, which is exactly
 * the order build() filled peer_scratch/peer_flags in. */

/* The one-shots run in place: the thread that stores word k of the output is the one that read
 * word k of the input, for its push and again as its own term of the fold, and peers read only
 * their own scratch (r4d_ar_oneshot_Nrank_exact.hip). */
extern "C" int r4d_rad_ar_oneshot_3rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_3rank_exact",
        r4d_ar_oneshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],
                                   in->peer_flags[0], in->peer_flags[1],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_4rank_exact",
        r4d_ar_oneshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_8rank_exact",
        r4d_ar_oneshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

/* The two-shots shard the message, so their block count scales with the SHARD -- a full-message
 * count over-launches the whole mid band.
 *
 * IN PLACE IS SAFE (r4d_ar_twoshot_Nrank_exact.hip). A thread takes one set of word offsets k
 * within a shard and uses it in all three phases: the push reads word k of every peer's shard out
 * of the input, the reduce reads word k of its own shard and stores the sum over it, and the
 * assemble writes word k of every peer's shard. Each input word is therefore overwritten only by
 * the thread that already read it, later in its own program order, and no peer reads this rank's
 * input -- peers read their own scratch. The selector hands the two-shot the messages above the
 * one-shot's size bound (r4d_rows.cpp, cArTsExact4). */
extern "C" int r4d_rad_ar_twoshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    /* The message must shard evenly over the ranks in whole 16-byte words, and 2*shard must fit
     * the slot -- both are throws in the entry point. */
    const long words16 = c.numel * c.in->esize / 16;
    if (words16 % c.in->world) return RAD_E_SHAPE;
    if (2 * (c.numel * c.in->esize) / c.in->world > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_4rank_exact",
        r4d_ar_twoshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    const long words16 = c.numel * c.in->esize / 16;
    if (words16 % c.in->world) return RAD_E_SHAPE;
    if (2 * (c.numel * c.in->esize) / c.in->world > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_8rank_exact",
        r4d_ar_twoshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

/* The tiered-int8 wire. Its slot stride is in BYTES, not 16-byte words: the wire is 38 bytes per
 * group of 32 and so is not 16-byte aligned at all.
 *
 * It is the only lossy row at four ranks, so a lossy request resolves it at every message size. As
 * in r4d_rad_ar_oneshot_2rank_wht6, a message under `min_bytes` (or one the groups do not tile)
 * goes to the exact one-shot. The default floor is the two-rank wire's: every block declares
 * `min_bytes` from --tp-wire-min-kb, so the default serves only a caller that leaves it out, and no
 * four-rank crossover has been measured to set another. In place is safe for the exact two-shot's
 * reason: a thread owns the same 8 elements of a shard in all three phases, reads them from the
 * input before it writes them, and no peer reads this rank's input. */
extern "C" int r4d_rad_ar_twoshot_4rank_ti8(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;

    int group = 0, group_bytes = 0;
    r4d_ar_ti8_dims(&group, &group_bytes);
    if (group <= 0) return RAD_E_DEVICE;
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    if ((long long)c.numel * c.in->esize < floor_b || c.numel % ((long)group * c.in->world))
        return r4d_rad_ar_oneshot_4rank_exact(a, s);
    /* The slot holds two wire images of one shard. */
    const long wire = (c.numel / c.in->world / group) * group_bytes;
    if (2 * wire > c.in->slot_bytes) return RAD_E_SCRATCH;
    /* nthreads must be a multiple of the encoder's group width -- a throw in the entry point. */
    if (c.nthreads % 32) return RAD_E_INVAL;

    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_4rank_ti8",
        r4d_ar_twoshot_4rank_ti8(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                 in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                 in->my_scratch, in->my_flags, in->my_seq, in->slot_bytes,
                                 c.inp, c.out, c.numel, in->dtype, in->rank,
                                 (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                 c.pub));
    return RAD_OK;
}

/* ------------------------------------------------------------------ the fused all-reduce + norm */

enum { AL_X = 0, AL_RESID, AL_W, AL_Q, AL_SCALE, AL_OUTBF16 };

extern "C" int r4d_rad_ar_ln_had_quant_i8(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (x->rank < 2 || q->rank < 2) return RAD_E_SHAPE;
    /* One workgroup owns whole rows and walks them with a vector load: nothing here is pitched. */
    if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(w) || !r4d_contig(sc))
        return RAD_E_STRIDE;
    if (rs && !r4d_contig(rs)) return RAD_E_STRIDE;
    if (ob && !r4d_contig(ob)) return RAD_E_STRIDE;

    const long long K = x->shape[x->rank - 1];
    long long M = 1;
    for (uint32_t i = 0; i + 1 < x->rank; ++i) M *= x->shape[i];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;

    const long long had  = r4d_geti_or(a, "group", 0);
    const int       wire = (int)r4d_geti_or(a, "wire", 0);
    const float     eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float     wadd = (float)r4d_getf(a, "wadd", 0.0);

    /* Everything the entry point throws about, checked here so it does not have to. */
    if (K % 16) return RAD_E_SHAPE;
    if (had < 2 || (had & (had - 1))) return RAD_E_INVAL;
    if (had != 128 && had != 256 && had != 512) return RAD_E_SHAPE;
    if (K % had) return RAD_E_SHAPE;
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        int stage_bits = 0, chunk_elems = 0, chunk_bytes = 0, scale_bytes = 0;
        r4d_ar_ln_wire_dims(&stage_bits, &chunk_elems, &chunk_bytes, &scale_bytes);
        if (chunk_elems <= 0 || K % chunk_elems) return RAD_E_SHAPE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const R4dArCfg* cfg = &cfgv;
    int nthreads = cfg->nthreads;
    if (nthreads != 128 && nthreads != 256 && nthreads != 512 && nthreads != 1024)
        return RAD_E_INVAL;
    /* `nblocks` is CLAMPED TO M inside the kernel because a block owns whole rows; clamping here
     * too keeps the number this shim reports and the number the kernel runs the same. */
    int nblocks = cfg->nblocks > 0 ? cfg->nblocks : nb_for(M * K * 2 / 16, in->max_blocks);
    if (nblocks > (int)M) nblocks = (int)M;
    /* THE LDS FLOOR ON THE BLOCK COUNT. The compressed wire stages every row a block owns, so a
     * count chosen for the push alone hands a block more rows than LDS holds -- at K = 5120 that
     * is anything past two -- and the launch is refused from M = 9 up. Raising the count to the
     * kernel's own floor serves those shapes; a shape no count below the flag capacity serves is
     * refused here by name instead of thrown on in the entry point. Block count changes no bytes:
     * a block owns whole rows and every reduction closes over one. */
    const int nb_min = r4d_ar_ln_blocks_min((int)M, (int)K, (int)had, nthreads, wire);
    if (nb_min < 1 || nb_min > in->max_blocks) return RAD_E_SHAPE;
    if (nblocks < nb_min) nblocks = nb_min;

    R4D_TRY("ar_ln_had_quant_i8",
        r4d_ar_ln_had_quant_i8(in->peer_scratch[0], in->my_scratch, in->peer_flags[0],
                               in->my_flags, in->my_seq, in->slot16,
                               r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob),
                               r4d_addr(q), r4d_addr(sc),
                               (int)M, (int)K, (int)had, eps, wadd,
                               (long)(uintptr_t)s, nblocks, nthreads, cfg->drain, cfg->acq,
                               wire));
    return RAD_OK;
}

/* The fp8 twin: deliberately the same shim as the int8 form with the Hadamard rotation taken out.
 *
 * `x` IS INOUT HERE, which is the one operand difference from the int8 form: that kernel keeps the
 * reduced row in LDS and never writes it, while this one stores it back exactly as the standalone
 * all-reduce does -- so a reader of `x` that is not the norm still finds the bytes the unfused
 * pair would have left, and the buffer planner has to know `x` is written. That holds on both
 * wires.
 *
 * `wire` is read and passed through, 0 for the exact bf16 payload and 6 for the rotated 6-bit one.
 * The architecture asks the same question the standalone `all_reduce` row answers -- an install
 * that forbids a lossy wire, a message the rotation's group does not tile, or a message under
 * --tp-wire-min-kb -- and both ranks evaluate it on the same T, so neither can take a different
 * branch. Wire 6 additionally needs a chunk not to straddle a row; that is a shape this shim
 * refuses rather than a decision it makes. */
extern "C" int r4d_rad_ar_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (!r4d_contig(w)) return RAD_E_STRIDE;

    /* Rank 2 everywhere, and for rmsnorm_quant_fp8's reason: a norm has a gain per channel, so
     * "everything after dim 0 is the row" would put one gain across several heads. */
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return RAD_E_SHAPE;
    if (ob && ob->rank != 2) return RAD_E_SHAPE;
    const long long M = x->shape[0], K = x->shape[1];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    /* A row must be a whole number of 16B words, because a block pushes its rows as words. */
    if (K % 8) return RAD_E_SHAPE;

    const long long grp = r4d_quant_act_fp8_group();
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_geti_or(a, "group", grp) != grp) return RAD_E_SHAPE;
    if (r4d_numel(w) != K) return RAD_E_SHAPE;
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (q->shape[0] != M || q->shape[1] != K) return RAD_E_SHAPE;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return RAD_E_SHAPE;
    if (ob && (ob->shape[0] != M || ob->shape[1] != K)) return RAD_E_SHAPE;
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return RAD_E_SHAPE;
    /* x IS THE MESSAGE, so its rows must be where the push says they are: tight. The other
     * operands may be pitched and the entry point takes their pitches. */
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = ob ? r4d_row_pitch(ob) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return RAD_E_STRIDE;
    if (xp != K) return RAD_E_STRIDE;

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    /* The gain's width is read off the operand, exactly as rmsnorm_quant_fp8's shim reads it. */
    const int w_f32 = w->dtype == RAD_F32;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        const int ce = r4d_ar_rmsnorm_quant_fp8_chunk_elems();
        if (ce <= 0 || K % ce) return RAD_E_SHAPE;
        /* The side buffer holding this rank's own packed half. Allocated for every bf16 instance,
         * so its absence is a broken init rather than a shape this row declines. */
        if (!in->locpack) return RAD_E_STATE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const R4dArCfg* cfg = &cfgv;
    /* THE WIDTH IS NOT A TUNING KNOB ON THIS ROW: passing the tuned `nthreads` through here is a
     * latent output change. It decides which columns a thread accumulates into the sum of squares,
     * so the fused kernel and the standalone `rmsnorm_quant_fp8` it stands in for must pick the
     * same one -- and the standalone asks r4d_act_threads(M, n), whose answer DEPENDS ON M: 1024 to
     * 32 rows and 512 above. Every point in the all-reduce space pins a width (the default is
     * 1024), so at M = 64 -- eight concurrent sequences at DFlash2 depth 7 -- a pinned width would
     * sum a row in a different order from the kernel it replaces and move the generated text. The
     * divergence is invisible on the exact wire alone, because that band tops out at one sequence.
     *
     * Zero means "the norm's width" and the entry point resolves it from M. The tuned
     * block count, drain and acq are still honoured: those change no bytes. */
    const int nthreads = 0;
    /* ONE BLOCK A ROW, which is the NORM's shape and not the all-reduce's, and the two halves
     * genuinely want different numbers. The push saturates the fabric at four blocks and every
     * block past that pays its own flag round trip over it (r4d_ar_ln_had_quant_i8.hip carries the
     * block sweep); the norm's sum of squares closes over a whole row, so a block holding two rows
     * does them one after the other and the second waits for the first. The norm wins that
     * argument by a wide margin: at nb_for()'s 4 blocks the fused kernel gives back more in
     * serialised rows than the saved dispatch is worth. M is eight at one concurrent sequence and
     * sixty-four at eight, so one block a row is a block count the fabric is comfortable with
     * either way, and on wire 6 it also hands every block a whole row's worth of chunks to rotate.
     * A tuned `nblocks` may name a smaller number; the LDS the compressed wire needs then grows
     * with the rows a block owns, and the entry point refuses what will not fit.
     *
     * `nb_cap` IS THE WRONG CEILING HERE. It is 24 at world 2 and it is the PUSH's number: past
     * four blocks the fabric is saturated and every block after that is another flag round trip.
     * The norm's number is M. Below 24 rows they do not conflict and above them they do -- at
     * M = 64 a capped form runs 24 blocks of three rows and does those three IN SEQUENCE, against
     * the standalone norm's 64 blocks of one, and a row's norm is two barriers and a handful of
     * elements a thread.
     *
     * So: one block a row, under a cap of 256. The wire prefers a low cap -- its own knee is near
     * 64 blocks -- but a prefill chunk of 2048 rows under a cap of 64 hands every block
     * THIRTY-TWO rows to do in sequence, and those serialised norms cost several times what the
     * wire gains. 256 is the knee of the combined cost and 512 is inside its noise. Nothing at
     * decode is affected: M is at most 64 there (eight sequences at DFlash2 depth 7), so
     * `nblocks = min(M, cap)` is simply M.
     *
     * BLOCK COUNT CHANGES NO BYTES either way: a block owns whole rows, the wire's chunk of 512
     * tiles a row, and every reduction closes over one row, so no element's arithmetic depends on
     * which block held it.
     *
     * The caps are not reachable at decode sizes; they are here because a block count is a hard
     * bound on the flag array and a tuned `nblocks` may name any number it likes. */
    int nblocks = cfg->nblocks > 0 ? cfg->nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    constexpr int kNbCap = 256;
    if (nblocks > kNbCap) nblocks = kNbCap;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;

    const int e = r4d_ar_rmsnorm_quant_fp8(
        in->peer_scratch[0], in->my_scratch,
        in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob), r4d_addr(q),
        r4d_addr(sc), (int)M, (int)K, eps, wadd, w_f32,
        (int)rp, (int)yp, (int)qp, (int)sp,
        (long)(uintptr_t)s, nblocks, nthreads, cfg->drain, cfg->acq, wire,
        in->locpack, in->slot_bytes / 2);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* THE ALL-REDUCE WITH THE GATED RESIDUAL'S WRITE IN IT: `all_reduce` of `y` and then `hc_write`
 * of the reduced `y` into `h`, one kernel. A block whose output feeds a hyper-connection write
 * skips its own collective when this runs, and the write carries it.
 *
 * `y` IS THE MESSAGE and is pushed as 16B words, so it must be tight -- the same rule the fused
 * norm states. The streams may be pitched; each is written as whole 16B words, which n and the
 * pitch being multiples of eight guarantee.
 *
 * `wire` is 0 for the exact bf16 payload and 6 for the rotated 6-bit one, and it is the op's and
 * not the message's: the architecture declares one op a wire and issues the one ar_wire_is_exact
 * picks for the step, which both ranks decide from the same T. So a wire-6 op is 6 at every size,
 * with no floor of its own. */
enum { AH_Y = 0, AH_INJ, AH_H };

/* ---- ar_gather_hc_write: moe_gather, then ar_hc_write, in one kernel. Operands:
 *   [0] y_expert  [1] expert_w  [2] sorted_tok  [3] shared?  [4] shared_gate?  [5] y  [6] inj  [7] h
 * The first five are moe_gather's and the last three ar_hc_write's; see those two. `wire` is
 * ar_hc_write's; wire 6 builds its slot table in LDS at every size and takes no scratch. */
enum { AG_YE = 0, AG_EW, AG_SORTED, AG_SH, AG_SG, AG_Y, AG_INJ, AG_H };

/* The operand and shape rules ar_gather_hc_write states at every width: moe_gather's five and
 * ar_hc_write's three, and the inverse table's scratch. */
struct GatherHcWriteArgs {
    const RadTensor *ye = nullptr, *ew = nullptr, *so = nullptr, *sh = nullptr, *sg = nullptr;
    const RadTensor *y = nullptr, *inj = nullptr, *h = nullptr;
    long long M = 0, n = 0, hc = 0, top_k = 0, irow = 0, hrow = 0;
    long long ye_ld = 0, ew_ld = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0, wire = 0;
    bool has_inv = false;
};

static int ar_gather_hc_write_args(const RadArgs* a, const ArInst* in, GatherHcWriteArgs* o) {
    const RadTensor* ye  = r4d_opt(a, AG_YE);
    const RadTensor* ew  = r4d_opt(a, AG_EW);
    const RadTensor* so  = r4d_opt(a, AG_SORTED);
    const RadTensor* sh  = r4d_opt(a, AG_SH);
    const RadTensor* sg  = r4d_opt(a, AG_SG);
    const RadTensor* y   = r4d_opt(a, AG_Y);
    const RadTensor* inj = r4d_opt(a, AG_INJ);
    const RadTensor* h   = r4d_opt(a, AG_H);
    if (!ye || !ew || !so || !y || !inj || !h) return RAD_E_INVAL;
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8 || top_k < 1) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    /* moe_gather's side: the same pitches and the same shared-arm pair. */
    const long long T = M * top_k;
    long long Te = 0, Mw = 0;
    const long long ye_ld = r4d_rows_pitch(ye, n, &Te);
    const long long ew_ld = r4d_rows_pitch(ew, top_k, &Mw);
    if (ye_ld < 0 || ew_ld < 0 || Te < T || Mw < M) return RAD_E_STRIDE;
    if (so->dtype != RAD_I32 || !r4d_contig(so) || r4d_numel(so) < T) return RAD_E_SHAPE;
    if (ye->dtype != RAD_BF16 || ew->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (ye_ld % 8 || !r4d_aligned(ye->data, 16)) return RAD_E_ALIGN;
    long long sh_ld = 0, sg_ld = 0, shrows = 0, sgrows = 0;
    int sig = 0;
    if (sh) {
        if (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16) return RAD_E_DTYPE;
        sh_ld = r4d_rows_pitch(sh, n, &shrows);
        sg_ld = r4d_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return RAD_E_STRIDE;
        const char* const act = r4d_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))  sig = 1;
        else if (act && std::strcmp(act, "none")) return RAD_E_UNSUPPORTED;
    }

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->ye = ye; o->ew = ew; o->so = so; o->sh = sh; o->sg = sg;
    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->top_k = top_k; o->irow = irow; o->hrow = hrow;
    o->ye_ld = ye_ld; o->ew_ld = ew_ld; o->sh_ld = sh_ld; o->sg_ld = sg_ld;
    o->sig = sig; o->wire = wire;
    o->has_inv = a->scratch && a->scratch_bytes >= T * (long long)sizeof(int);
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    const RadTensor* const ye = g.ye;
    const RadTensor* const ew = g.ew;
    const RadTensor* const so = g.so;
    const RadTensor* const sh = g.sh;
    const RadTensor* const sg = g.sg;
    const RadTensor* const y = g.y;
    const RadTensor* const inj = g.inj;
    const RadTensor* const h = g.h;
    const long long M = g.M, n = g.n, hc = g.hc, top_k = g.top_k, irow = g.irow, hrow = g.hrow;
    const long long ye_ld = g.ye_ld, ew_ld = g.ew_ld, sh_ld = g.sh_ld, sg_ld = g.sg_ld;
    const int sig = g.sig, wire = g.wire;
    const bool has_inv = g.has_inv;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long)M, (long)n, (long)hc, (long)irow, (long)hrow, r4d_addr(ye),
            r4d_addr(ew), r4d_addr(so), sh ? r4d_addr(sh) : 0L, sh ? r4d_addr(sg) : 0L,
            (long)ye_ld, (long)ew_ld, (long)sh_ld, (long)sg_ld, (long)top_k, (long)sig,
            (long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS, cfgv.drain, cfgv.acq);
        return rc == 0 ? RAD_OK : RAD_E_SHAPE;
    }
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    /* Preserve the old full fence if any peer allocation used the coarse fallback. */
    const int drain = in->all_fine ? cfgv.drain : 1;
    const int rc = r4d_ar_gather_hc_write(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h), (long)M, (long)n, (long)hc,
        (long)irow, (long)hrow, r4d_addr(ye), r4d_addr(ew), r4d_addr(so),
        sh ? r4d_addr(sh) : 0L, sh ? r4d_addr(sg) : 0L, (long)ye_ld, (long)ew_ld, (long)sh_ld,
        (long)sg_ld, (long)top_k, (long)sig, (long)(uintptr_t)s, nblocks, cfgv.nthreads,
        drain, cfgv.acq, has_inv ? (long)(uintptr_t)a->scratch : 0L);
    return rc == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* AT FOUR RANKS, EXACT WIRE ONLY: r4d_rad_ar_hc_write_4rank with the gather in front, on an
 * instance from the same init (r4d_rad_ar_hc_wide_init). The architecture issues it only on a step
 * whose write takes the exact fused form (ar_exact_write_takes, arch/common/rad_fp8.h) and whose
 * gather is one pass's.
 * The block count is that write's rule, widened by the launcher until a block's slot table fits in
 * LDS -- or, where that would take more blocks than the rule asks for and the scratch is there,
 * kept and served from the inverse table, as at two ranks. */
extern "C" int r4d_rad_ar_gather_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    if (g.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(g.M * g.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    R4D_TRY("ar_gather_hc_write_4rank",
        r4d_ar_gather_hc_write_4rank(
            in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
            r4d_addr(g.y), r4d_addr(g.inj), r4d_addr(g.h), (long)g.M, (long)g.n, (long)g.hc,
            (long)g.irow, (long)g.hrow, r4d_addr(g.ye), r4d_addr(g.ew), r4d_addr(g.so),
            g.sh ? r4d_addr(g.sh) : 0L, g.sh ? r4d_addr(g.sg) : 0L, (long)g.ye_ld,
            (long)g.ew_ld, (long)g.sh_ld, (long)g.sg_ld, (long)g.top_k, (long)g.sig,
            (long)in->rank, (long)(uintptr_t)s, nblocks, cfgv.nthreads, cfgv.drain, cfgv.acq,
            cfgv.pub, g.has_inv ? (long)(uintptr_t)a->scratch : 0L));
    return RAD_OK;
}

/* The inverse slot table the exact wire's prefill form reads: one int a (token, slot) at the
 * band's M. The rotated wire's table is in LDS. */
extern "C" int64_t r4d_rad_ar_gather_hc_write_scratch(const RadArgs* a) {
    if (r4d_geti_or(a, "wire", 0) == 6) return 0;
    const long long M = r4d_geti_or(a, "M", 0), k = r4d_geti_or(a, "top_k", 0);
    return M > 0 && k > 0 ? M * k * (int64_t)sizeof(int) : 0;
}

/* The operand and shape rules ar_hc_write states at every width. */
struct HcWriteArgs {
    const RadTensor* y = nullptr;
    const RadTensor* inj = nullptr;
    const RadTensor* h = nullptr;
    long long M = 0, n = 0, hc = 0, irow = 0, hrow = 0;
    int wire = 0;
};

static int ar_hc_write_args(const RadArgs* a, const ArInst* in, HcWriteArgs* o) {
    const RadTensor* y   = r4d_opt(a, AH_Y);
    const RadTensor* inj = r4d_opt(a, AH_INJ);
    const RadTensor* h   = r4d_opt(a, AH_H);
    if (!y || !inj || !h) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->irow = irow; o->hrow = hrow;
    o->wire = wire;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    const RadTensor* const y = w.y;
    const RadTensor* const inj = w.inj;
    const RadTensor* const h = w.h;
    const long long M = w.M, n = w.n, hc = w.hc, irow = w.irow, hrow = w.hrow;
    const int wire = w.wire;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long)M, (long)n, (long)hc, (long)irow, (long)hrow, 0L, 0L, 0L, 0L, 0L,
            0L, 0L, 0L, 0L, 0L, 0L, (long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS,
            cfgv.drain, cfgv.acq);
        return rc == 0 ? RAD_OK : RAD_E_SHAPE;
    }
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    /* Preserve the old full fence if any peer allocation used the coarse fallback. */
    const int drain = in->all_fine ? cfgv.drain : 1;
    R4D_TRY("ar_hc_write",
        r4d_ar_hc_write(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                        in->my_seq, in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h),
                        (long)M, (long)n, (long)hc, (long)irow, (long)hrow, (long)(uintptr_t)s,
                        nblocks, cfgv.nthreads, drain, cfgv.acq));
    return RAD_OK;
}

/* AT FOUR RANKS, EXACT WIRE ONLY: the wide one-shot with the write as its epilogue. The four-rank
 * lossy wire is the ti8 two-shot and has no fused form: the row constrains `wire` to 0, and a
 * `wire` 6 that reaches this entry anyway is refused. Every message up to the instance's bound is
 * served correctly here as a one-shot. Which steps take it is the caller's choice -- the
 * architecture issues it on decode and verify steps (ar_wide_write_exact, arch/common/rad_fp8.h) --
 * and a message the standalone all_reduce's bands would have sent as a two-shot costs more traffic
 * here for the same bytes.
 *
 * The block count is the plain one-shot's rule (ar_setup's nb_for over the whole message under
 * nb_cap); a block's words, flags and slot are the one-shot's, which is what lets the two share an
 * instance where their keys agree. */
extern "C" int r4d_rad_ar_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    if (w.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(w.M * w.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    R4D_TRY("ar_hc_write_4rank",
        r4d_ar_hc_write_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                              in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                              in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                              r4d_addr(w.y), r4d_addr(w.inj), r4d_addr(w.h),
                              (long)w.M, (long)w.n, (long)w.hc, (long)w.irow, (long)w.hrow,
                              (long)in->rank, (long)(uintptr_t)s, nblocks, cfgv.nthreads,
                              cfgv.drain, cfgv.acq, cfgv.pub));
    return RAD_OK;
}
