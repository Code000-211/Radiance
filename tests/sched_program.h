/* sched_program.h -- the minimal declared program the scheduler tests run against: a paged full
 * group beside a linear and a conv one, the one op whose schema names the chunk parameter, and the
 * derived buffers each group needs. Everything below the scheduler is a fake (sched_fakes.h).
 *
 * Two test binaries include this header, and only this shape: sched_draft_test.cpp, which drives
 * the scheduler directly, and engine_history_test.cpp, which hands the same program to a real
 * Engine's scheduler to reach the engine's collect and finish. sched_test.cpp keeps its own local
 * copy; the shape is what makes a mid-prompt chunk's plan, and a second copy here would drift.
 *
 * The caller owns `owned` (the calloc'd derived buffers) and frees it after the scheduler is
 * finished; the scheduler holds pointers into `prog`, so `prog` must outlive it. */
#pragma once

#include "sched_fakes.h"

namespace sched_fake {

namespace {

const RadParamSpec kGdnPs[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "chunk", RAD_P_INT, RAD_DERIVED,  RAD_PROLE_SEQ_CHUNK },
};
const RadOpSchema kGdnSchema = { "gdn_chunk_scan", kGdnPs, 2, nullptr, 0, "test" };
RadConstraint kGdnC[] = { RAD_CDIV("chunk", 64) };
RadKernelInfo kGdnKi;
KernelRow     kGdnRow;

inline void add_derived(Program& prog, std::vector<void*>& owned, const char* name,
                        const char* derive, int64_t n) {
    BufferInfo b;
    b.name = name;
    b.decl.dtype = RAD_I32;
    b.decl.rank = 1;
    b.decl.shape[0] = n;
    b.decl.kind = RAD_BUF_DERIVED;
    b.decl.derive = derive;
    b.bytes = n * 4;
    void* p = std::calloc(1, (size_t)b.bytes);
    owned.push_back(p);
    b.ptr = p;
    prog.buffers.push_back(b);
}

}  /* namespace */

inline void build_program(Program& prog, Config& cfg, FakeKV& kv, FakePC& pc,
                          std::vector<void*>& owned, int64_t blocks = 4096,
                          int64_t states = 64, int64_t max_tok = 512,
                          int64_t max_ctx = 4096) {
    kGdnKi = RadKernelInfo{};
    kGdnKi.name = "gdn_fake";
    kGdnKi.op = "gdn_chunked";
    kGdnKi.constraints = kGdnC;
    kGdnKi.n_constraints = 1;
    kGdnRow = KernelRow{ &kGdnKi, "fake", 0, 0 };

    prog.meta.n_ctx_train = 4096;
    prog.meta.n_vocab = 512;

    /* index 0 is the sentinel every handle table in Program carries */
    prog.ops.emplace_back();
    prog.buffers.emplace_back();

    KVGroupInfo attn;
    attn.name = "attn";
    attn.decl.kind = RAD_KV_FULL;
    attn.block_size = 16;
    prog.kv_groups.push_back(attn);

    KVGroupInfo lin;
    lin.name = "gdn";
    lin.decl.kind = RAD_KV_LINEAR;
    lin.decl.state_dim[0] = 128;
    lin.decl.state_dim[1] = 128;
    prog.kv_groups.push_back(lin);

    KVGroupInfo conv;
    conv.name = "conv";
    conv.decl.kind = RAD_KV_CONV;
    conv.decl.conv_width = 4;
    prog.kv_groups.push_back(conv);

    OpInfo op;
    op.op = "gdn_chunk_scan";
    op.schema = &kGdnSchema;
    Band b;
    b.hi = 4096;
    b.dom[RAD_DOMAIN_DEVICE].row = &kGdnRow;
    b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
    op.bands.push_back(b);
    prog.ops.push_back(op);

    add_derived(prog, owned, "seq_start", "seq_start_index", 257);
    add_derived(prog, owned, "chunk_bounds", "state_chunk_bounds", 4097);
    add_derived(prog, owned, "conv_idx", "conv_state_index", 256);

    cfg.max_tok = max_tok;
    cfg.max_seqs = 32;
    cfg.max_ctx = max_ctx;
    cfg.checkpoint_interval = 256;
    cfg.prefix_cache = true;
    /* Config::n_spec defaults to -1 (`auto`), which engine_bringup resolves; a fixture that
     * skips bringup must set it, and DraftController::configure refuses a negative depth. */
    cfg.n_spec = 0;

    kv.setup(prog.kv_groups, cfg.max_ctx, blocks, states);
    pc.kv = &kv;
    pc.groups.clear();
    for (size_t i = 0; i < prog.kv_groups.size(); ++i)
        if (kv.geom.g[i].paged) pc.groups.push_back((int32_t)i);
}

}  /* namespace sched_fake */
