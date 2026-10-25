#include "hip/hip_runtime.h"
#include "rad_plugin.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

/* ---- the context switch: callee-saved registers and the stack pointer (x86-64 SysV) */
extern "C" void emu_switch(void** save_sp, void* new_sp);
asm(R"(
.text
.globl emu_switch
.type emu_switch,@function
emu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
)");

namespace {
constexpr size_t kStack = 512 * 1024;
/* A park: the test holds block `block` at its `nth` cross-lane exchange (the wave barrier of the
 * nth __shfl/__syncwarp/__builtin_amdgcn_* it enters) while the rest of the grid runs, then lets
 * it go. That places a cross-block interleaving exactly where a case wants it -- the window
 * between a load and the check after it -- instead of hoping the block order gives it. */
struct EmuPark {
    int block = 0;
    unsigned nth = 0;
    bool hit = false;      /* the block reached it */
    bool released = false; /* and the test has let it go */
    int wave = 0;
};
struct Fiber {
    void* sp = nullptr;
    /* left uninitialised so the OS commits only the pages a fiber touches: a 1024-thread block
     * would otherwise zero half a GiB */
    std::unique_ptr<char[]> stack;
    EmuIdx tid{};
    bool done = false;
    unsigned bar_gen = 0;      /* block barrier generation this fiber waits on */
    unsigned wbar_gen = 0;
    int waiting = 0;           /* 0 run, 1 block barrier, 2 wave barrier */
};
struct Sched {
    std::vector<Fiber> f;
    void* main_sp = nullptr;
    int cur = -1, nthreads = 0;
    EmuIdx bid{}, bdim{}, gdim{};
    void (*body)(void*) = nullptr;
    void* ctx = nullptr;
    /* block barrier; or_acc[g % 3] collects __syncthreads_or's predicates at generation g */
    int arrive = 0; unsigned gen = 0;
    int or_acc[3] = {};
    /* wave barriers and exchange slots, double-buffered by parity */
    int warrive[64] = {}; unsigned wgen[64] = {};
    int slot[64][2][32] = {};
    int tag[64][2][32] = {};   /* which op, and its uniform operand, each lane posted */
    unsigned xparity[1024] = {};
    /* the park a test armed for this block, and how many cross-lane exchanges the block has
     * entered (the nth of them is where a park lands) */
    EmuPark* park = nullptr;
    unsigned barriers = 0;
    bool parked = false;
} S;

/* A block left standing at a park. Everything below is the whole grid's scratch, one block at a
 * time by construction: the blocks that run while it stands at its park leave theirs there, and one
 * set of stacks per thread is likewise reused. So the parked block's own goes out with the park and
 * back with the release, and its fibers resume on the frames they were interrupted in. */
struct Held {
    std::array<unsigned, 3> b{};
    EmuIdx bid{};
    unsigned nt = 0;
    int arrive = 0; unsigned gen = 0; int or_acc[3] = {};
    int warrive[64] = {}; unsigned wgen[64] = {};
    int slot[64][2][32] = {}; int tag[64][2][32] = {};
    unsigned xparity[1024] = {};
    unsigned barriers = 0;
    std::vector<void*> sp;                  /* where each fiber stood, [sp, top) being its frames */
    std::vector<int> waiting;
    std::vector<std::vector<char>> frames;
};

void yield() { Fiber& me = S.f[S.cur]; emu_switch(&me.sp, S.main_sp); }

extern "C" void emu_trampoline() {
    S.body(S.ctx);
    S.f[S.cur].done = true;
    yield();
    std::abort();
}
/* the entry a fresh stack "returns" into, keeping the SysV alignment at call boundaries */
asm(R"(
.text
.globl emu_entry
.type emu_entry,@function
emu_entry:
    andq $-16, %rsp
    call emu_trampoline
    ud2
)");
extern "C" void emu_entry();
}  // namespace

EmuIdx& emu_thread_idx() { return S.f[S.cur].tid; }
EmuIdx& emu_block_idx() { return S.bid; }
EmuIdx& emu_block_dim() { return S.bdim; }
EmuIdx& emu_grid_dim() { return S.gdim; }

void __syncthreads() {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.gen;
    /* Clearing the next generation's slot here, as the barrier completes, keeps it ahead of every
     * write to it (threads arrive at g + 1 only after g completes) and behind every read of the
     * slot it shares a residue with (g - 2's, read before arriving at g - 1). */
    if (++S.arrive == S.nthreads) { S.arrive = 0; S.or_acc[(g + 1) % 3] = 0; ++S.gen; return; }
    me.waiting = 1;
    while (S.gen == g) yield();
    me.waiting = 0;
}
int __syncthreads_or(int p) {
    const unsigned g = S.gen;
    if (p) S.or_acc[g % 3] = 1;
    __syncthreads();
    return S.or_acc[g % 3];
}

static void wave_barrier(int w) {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.wgen[w];
    if (me.tid.x == 0) {
        ++S.barriers;
        if (S.park && !S.park->hit && S.barriers == S.park->nth) {
            S.park->hit = true;
            S.park->wave = w;
        }
    }
    if (++S.warrive[w] == 32) {
        /* The armed park's barrier does not complete: the whole wave waits, the scheduler leaves
         * this block and runs the rest of the grid, and the barrier completes on release. */
        if (!(S.park && S.park->hit && !S.park->released)) {
            S.warrive[w] = 0;
            ++S.wgen[w];
            return;
        }
        S.parked = true;
    }
    me.waiting = 2;
    while (S.wgen[w] == g) yield();
    me.waiting = 0;
}

/* Every lane posts `v` to the wave's exchange slots and passes the wave barrier; the slots of this
 * exchange are returned for the caller to read. `tag` names the op and its wave-uniform operand:
 * lanes that met at different ops (a cross-lane op called under divergence, each side reaching
 * one) are reported, not combined. */
enum { kOpBpermute = 1 << 24, kOpReadlane = 2 << 24, kOpBallot = 3 << 24, kOpDpp = 4 << 24,
       kOpPermlanex16 = 5 << 24 };
static const int* exchange(int v, int* lane, int tag) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5;
    const unsigned par = S.xparity[t]++ & 1u;
    *lane = t & 31;
    S.slot[w][par][*lane] = v;
    S.tag[w][par][*lane] = tag;
    wave_barrier(w);
    for (int i = 0; i < 32; ++i)
        if (S.tag[w][par][i] != tag) {
            std::fprintf(stderr, "emu: DIVERGENCE in block (%u,%u,%u), wave %d: lanes %d and %d met at different "
                         "cross-lane ops (0x%x, 0x%x)\n", S.bid.x, S.bid.y, S.bid.z, w, *lane, i, tag,
                         S.tag[w][par][i]);
            std::abort();
        }
    return S.slot[w][par];
}
int emu_bpermute(int addr, int v) {
    int l;
    return exchange(v, &l, kOpBpermute)[(addr >> 2) & 31];
}
/* v_readlane_b32 takes a wave-uniform lane, so a lane-varying one is reported */
int emu_readlane(int v, int lane) {
    int l;
    return exchange(v, &l, kOpReadlane | (lane & 31))[lane & 31];
}
int emu_cvt_pk_fp8(float a, float b, int old, bool hi) {
    const unsigned pk = (unsigned)rad_f32_to_fp8e4m3(a) | ((unsigned)rad_f32_to_fp8e4m3(b) << 8);
    const unsigned o = (unsigned)old;
    return (int)(hi ? ((o & 0xffffu) | (pk << 16)) : ((o & 0xffff0000u) | pk));
}
/* One 16x16x16 WMMA: each lane posts its A and B elements, then reads the whole tile back. `ga`
 * and `gb` are the exchange slots of one element type, [wave][parity][lane][element]. */
template <class T, class V, class Cvt>
static emu_v8f emu_wmma(T (&ga)[64][2][32][8], T (&gb)[64][2][32][8], V a, V b, emu_v8f c,
                        Cvt cvt) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    for (int e = 0; e < 8; ++e) { ga[w][par][l][e] = a[e]; gb[w][par][l][e] = b[e]; }
    wave_barrier(w);
    const int N = l & 15, h = l >> 4;
    emu_v8f d;
    for (int e = 0; e < 8; ++e) {
        const int M = 8 * h + e;
        float acc = c[e];
        for (int k = 0; k < 16; ++k)
            acc += cvt(ga[w][par][M + 16 * (k >> 3)][k & 7]) * cvt(gb[w][par][N + 16 * (k >> 3)][k & 7]);
        d[e] = acc;
    }
    return d;
}
static short g_wa[64][2][32][8], g_wb[64][2][32][8];
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c) {
    return emu_wmma(g_wa, g_wb, a, b, c, [](short v) { return rad_bf16_to_f32((uint16_t)v); });
}
static _Float16 g_ha[64][2][32][8], g_hb[64][2][32][8];
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c) {
    return emu_wmma(g_ha, g_hb, a, b, c, [](_Float16 v) { return (float)v; });
}
namespace {
struct Bytes8 {
    uint8_t v[8];
    uint8_t operator[](int e) const { return v[e]; }
};
}
static uint8_t g_qa[64][2][32][8], g_qb[64][2][32][8];
emu_v8f emu_wmma_fp8(emu_v2i a, emu_v2i b, emu_v8f c) {
    Bytes8 pa, pb;
    std::memcpy(pa.v, &a, 8);
    std::memcpy(pb.v, &b, 8);
    return emu_wmma(g_qa, g_qb, pa, pb, c, [](uint8_t v) { return rad_fp8e4m3_to_f32(v); });
}
emu_v8i emu_wmma_iu8(bool, emu_v2i, bool, emu_v2i, emu_v8i, bool) {
    std::fprintf(stderr, "emu: v_wmma_i32_16x16x16_iu8 is not emulated\n");
    std::abort();
}
unsigned emu_ballot(bool p) {
    int l;
    const int* s = exchange(p ? 1 : 0, &l, kOpBallot);
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) m |= (unsigned)(s[i] & 1) << i;
    return m;
}
int emu_update_dpp(int old, int src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl) {
    int l;
    const int* s = exchange(src, &l, kOpDpp | (ctrl & 0xffff));
    const int r = l & 15, k = ctrl & 15;
    int from;
    if (ctrl > 0x100 && ctrl < 0x110)      from = r + k < 16 ? l + k : -1;     /* row_shl */
    else if (ctrl > 0x110 && ctrl < 0x120) from = r - k >= 0 ? l - k : -1;     /* row_shr */
    else if (ctrl >= 0x160 && ctrl < 0x170) from = (l & 16) | (r ^ k);         /* row_xmask */
    else { std::fprintf(stderr, "emu: dpp_ctrl 0x%x is not emulated\n", ctrl); std::abort(); }
    if ((row_mask & 3) != 3 || (bank_mask & 15) != 15) {
        std::fprintf(stderr, "emu: dpp row/bank masks 0x%x/0x%x are not emulated\n", row_mask, bank_mask);
        std::abort();
    }
    return from < 0 ? (bound_ctrl ? 0 : old) : s[from];
}
int emu_permlanex16(int, int src, unsigned sel_lo, unsigned sel_hi, bool, bool) {
    int l;
    const int* s = exchange(src, &l, kOpPermlanex16);
    const int r = l & 15;
    const unsigned sel = (r < 8 ? sel_lo >> (4 * r) : sel_hi >> (4 * (r - 8))) & 15u;
    return s[(16 - (l & 16)) | (int)sel];
}
emu_v8s emu_load_tr_b128(const void*) {
    std::fprintf(stderr, "emu: global_load_tr_b128 is not emulated\n");
    std::abort();
}

/* Dynamic LDS: a mapping whose last page is inaccessible, the buffer placed against it with its
 * start 16-byte aligned, refilled with a poison pattern for every block. */
static char* g_lds_map = nullptr;
static size_t g_lds_map_len = 0;
static char* g_lds = nullptr;
void* emu_dyn_lds() { return g_lds; }
static void lds_reserve(size_t bytes) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t len = (bytes + 15 + page - 1) / page * page + page;
    if (len > g_lds_map_len) {
        if (g_lds_map) munmap(g_lds_map, g_lds_map_len);
        void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED || mprotect((char*)m + len - page, page, PROT_NONE) != 0) {
            std::fprintf(stderr, "emu: no dynamic LDS mapping\n");
            std::abort();
        }
        g_lds_map = (char*)m;
        g_lds_map_len = len;
    }
    const uintptr_t end = (uintptr_t)(g_lds_map + g_lds_map_len - page);
    g_lds = (char*)((end - bytes) & ~(uintptr_t)15);
}

int emu_grids = 0;
int emu_last_block = 0;
const char* emu_last_kernel = "";

void emu_syncwarp() { wave_barrier((int)(S.f[S.cur].tid.x >> 5)); }

/* r4d_p2p_copy2d is the AR unit's (r4d_ar_oneshot_2rank_exact.hip), which the emulator does not
 * build; among the units it does build r4d_model_bf16.hip's cast op is its only caller, and no
 * emulated case reaches it. A row copy is exact here for the same reason every other copy is: the
 * "device" buffers are host memory. Kept so a unit that references it links. */
extern "C" int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width,
                             long height, long stream) {
    (void)stream;
    for (long r = 0; r < height; ++r)
        std::memcpy((void*)(uintptr_t)(dst + r * dpitch),
                    (const void*)(uintptr_t)(src + r * spitch), (size_t)width);
    return 0;
}

/* The parks a test armed, in the order it armed them. */
static std::vector<EmuPark> g_parks;
int emu_park(int block, unsigned nth) {
    g_parks.push_back(EmuPark{block, nth, false, false, 0});
    return (int)g_parks.size() - 1;
}
int emu_park_hit(int park) {
    return park >= 0 && park < (int)g_parks.size() && g_parks[(size_t)park].hit;
}
void emu_park_clear() { g_parks.clear(); }

void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx, size_t dyn_lds) {
    ++emu_grids;
    const int n = (int)(block.x * block.y * block.z);
    emu_last_block = n;
    if (block.y != 1 || block.z != 1 || n % 32 || n > 1024) { std::fprintf(stderr, "emu: block shape\n"); std::abort(); }
    if (dyn_lds > 64 * 1024) {
        std::fprintf(stderr, "emu: %zu bytes of dynamic LDS, past the 64 KiB a workgroup has\n", dyn_lds);
        std::abort();
    }
    lds_reserve(dyn_lds);
    S.body = body; S.ctx = ctx; S.nthreads = n;
    S.bdim = EmuIdx{block.x, block.y, block.z};
    S.gdim = EmuIdx{grid.x, grid.y, grid.z};
    S.f.resize(n);
    for (auto& f : S.f) if (!f.stack) f.stack.reset(new char[kStack]);

    auto park_of = [&](unsigned bx, unsigned by, unsigned bz) -> EmuPark* {
        const int idx = (int)((bz * grid.y + by) * grid.x + bx);
        for (auto& p : g_parks)
            if (p.block == idx) return &p;
        return nullptr;
    };
    auto begin = [&](unsigned bx, unsigned by, unsigned bz) {
        S.bid = EmuIdx{bx, by, bz};
        std::memset(g_lds, 0xA5, dyn_lds);
        S.arrive = 0; S.gen = 0; S.barriers = 0; S.parked = false;
        for (int& a : S.or_acc) a = 0;
        for (int w = 0; w < 64; ++w) { S.warrive[w] = 0; S.wgen[w] = 0; }
        for (int t = 0; t < n; ++t) {
            Fiber& f = S.f[t];
            f.done = false; f.waiting = 0; f.tid = EmuIdx{(unsigned)t, 0, 0};
            S.xparity[t] = 0;
            /* a fresh stack: six callee-saved slots, then the entry as the return address */
            void** top = (void**)(f.stack.get() + kStack - 64);
            *--top = (void*)&emu_entry;
            for (int r = 0; r < 6; ++r) *--top = nullptr;
            f.sp = top;
        }
    };
    auto run = [&](unsigned bx, unsigned by, unsigned bz) -> bool {   /* true: it parked */
        int live = 0;
        for (auto& f : S.f) live += !f.done;
        while (live > 0) {
            const int live_before = live;
            const unsigned g0 = S.gen; unsigned wg0[64]; for (int w = 0; w < 64; ++w) wg0[w] = S.wgen[w];
            int arr0 = S.arrive; int warr0[64]; for (int w = 0; w < 64; ++w) warr0[w] = S.warrive[w];
            live = 0;
            for (int t = 0; t < n; ++t) {
                if (S.f[t].done) continue;
                S.cur = t;
                emu_switch(&S.main_sp, S.f[t].sp);
                S.cur = -1;
                if (!S.f[t].done) ++live;
            }
            bool progressed = S.gen != g0 || S.arrive != arr0 || live != live_before;
            for (int w = 0; w < 64; ++w) progressed |= (S.wgen[w] != wg0[w]) || (S.warrive[w] != warr0[w]);
            if (live > 0 && !progressed) {
                if (S.parked) return true;     /* every lane of the wave at the armed park */
                int done = 0, wb = 0, bb = 0;
                for (auto& f : S.f) { done += f.done; wb += f.waiting == 2; bb += f.waiting == 1; }
                std::fprintf(stderr, "emu: DEADLOCK in block (%u,%u,%u): %d threads done, %d at a block "
                             "barrier, %d at a wave barrier -- a barrier not reached uniformly\n",
                             bx, by, bz, done, bb, wb);
                std::abort();
            }
        }
        return false;
    };

    /* A block a test armed a park on runs first and is left standing at that barrier: the grid's
     * other blocks then run to completion in dispatch order, and only then does the parked block
     * go on. One interleaving, one order, whichever day it runs on. */
    auto take = [&](unsigned bx, unsigned by, unsigned bz) {
        Held w;
        w.b = {bx, by, bz};
        w.bid = S.bid;
        w.arrive = S.arrive; w.gen = S.gen;
        std::memcpy(w.or_acc, S.or_acc, sizeof w.or_acc);
        std::memcpy(w.warrive, S.warrive, sizeof w.warrive);
        std::memcpy(w.wgen, S.wgen, sizeof w.wgen);
        std::memcpy(w.slot, S.slot, sizeof w.slot);
        std::memcpy(w.tag, S.tag, sizeof w.tag);
        std::memcpy(w.xparity, S.xparity, sizeof w.xparity);
        w.barriers = S.barriers;
        /* The stacks are one set of kStack per thread, and the blocks that run next take them
         * over, so the part of them this block is using goes with the park. Only [sp, top): a
         * kernel's frames here are kilobytes. */
        w.nt = S.nthreads;
        w.sp.resize(w.nt); w.waiting.resize(w.nt); w.frames.resize(w.nt);
        for (unsigned t = 0; t < w.nt; ++t) {
            Fiber& f = S.f[t];
            int* top = (int*)(f.stack.get() + kStack);
            w.sp[t] = f.sp;
            w.waiting[t] = f.waiting;
            w.frames[t].assign((char*)f.sp, (char*)top);
        }
        return w;
    };
    auto put = [&](const Held& w) {
        S.bid = w.bid;
        S.arrive = w.arrive; S.gen = w.gen;
        std::memcpy(S.or_acc, w.or_acc, sizeof S.or_acc);
        std::memcpy(S.warrive, w.warrive, sizeof S.warrive);
        std::memcpy(S.wgen, w.wgen, sizeof S.wgen);
        std::memcpy(S.slot, w.slot, sizeof S.slot);
        std::memcpy(S.tag, w.tag, sizeof S.tag);
        std::memcpy(S.xparity, w.xparity, sizeof S.xparity);
        S.barriers = w.barriers;
        for (unsigned t = 0; t < w.nt; ++t) {
            Fiber& f = S.f[t];
            char* top = f.stack.get() + kStack;
            std::memcpy(top - w.frames[t].size(), w.frames[t].data(), w.frames[t].size());
            f.sp = w.sp[t];
            f.waiting = w.waiting[t];
            f.done = false;
        }
    };
    std::vector<Held> held;
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        S.park = park_of(bx, by, bz);
        begin(bx, by, bz);
        if (run(bx, by, bz)) held.push_back(take(bx, by, bz));
    }
    for (size_t h = 0; h < held.size(); ++h) {
        const std::array<unsigned, 3> b = held[h].b;   /* held may grow below */
        EmuPark* p = park_of(b[0], b[1], b[2]);
        if (!p || !p->hit) {
            std::fprintf(stderr, "emu: block (%u,%u,%u) parked and never reached the armed barrier %u\n",
                         b[0], b[1], b[2], p ? p->nth : 0u);
            std::abort();
        }
        put(held[h]);
        p->released = true;
        S.warrive[p->wave] = 0;      /* the barrier this block stood at now completes */
        ++S.wgen[p->wave];
        S.park = p;
        S.parked = false;
        if (run(b[0], b[1], b[2])) held.push_back(take(b[0], b[1], b[2]));
    }
    S.park = nullptr;
}

/* ---- where a pointer lives (hipPointerGetAttributes)
 *
 * The emulator's allocations are host memory, so a plugin asking the driver about its own buffers
 * gets "host" -- the PLE cache's host-read accounting, for one. A test that wants the other branch
 * marks a range as device-resident; the range list is process-wide and cleared by emu_clear_devices.
 */
namespace {
struct EmuDevRange { const char* p; size_t n; };
std::vector<EmuDevRange> g_dev_ranges;
std::vector<EmuDevRange> g_unknown_ranges;   /* ranges the driver cannot classify */
}  // namespace

void emu_mark_device(const void* p, size_t n) {
    if (p && n) g_dev_ranges.push_back({(const char*)p, n});
}

void emu_mark_unknown(const void* p, size_t n) {
    if (p && n) g_unknown_ranges.push_back({(const char*)p, n});
}

void emu_clear_devices() { g_dev_ranges.clear(); g_unknown_ranges.clear(); }

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* a, const void* p) {
    if (!a || !p) return 1;
    *a = hipPointerAttribute_t{};
    const char* q = (const char*)p;
    for (const EmuDevRange& r : g_unknown_ranges)
        if (q >= r.p && q < r.p + r.n) return 1;   /* what the driver says about a locked pool */
    for (const EmuDevRange& r : g_dev_ranges)
        if (q >= r.p && q < r.p + r.n) {
            a->type = hipMemoryTypeDevice;
            a->devicePointer = (void*)p;
            return 0;
        }
    a->type = hipMemoryTypeHost;
    a->hostPointer = (void*)p;
    return 0;
}
