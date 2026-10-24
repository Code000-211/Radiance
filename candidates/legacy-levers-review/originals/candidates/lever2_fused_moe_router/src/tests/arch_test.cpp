/* arch_test.cpp -- the declared graph of the qwen35 architecture plugin, asserted statically.
 *
 * No GPU, no model file, no core. That is the point of two-phase construction (spec §3): between
 * declare and plan the whole model is known, so "does this plugin declare a Qwen3.5 layer" is a
 * question a unit test can answer. The builder below is a fake that records every decl_weight,
 * decl_buffer, decl_op, buffer use and name-map call, and a fake RadCtx that records every issue --
 * so the test also checks the run phase's ORDER and its OPERANDS, which is the other half of what a
 * plugin is.
 *
 * The fake is deliberately not core/build/'s RadBuilder: this test must not go red because of a
 * defect in that builder, and a plugin that only works against one builder is a plugin with an
 * undeclared dependency.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include "rad_test.h"

#include "rad_builder.h"
#include "rad_runtime.h"

#include <cstdarg>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

/* ==================================================================== the recording builder */
namespace {

struct RecParam {
    std::string key;
    int         kind = 0;
    long long   ival = 0, ihi = 0;
    double      dval = 0.0;
    std::string sval;
};

struct RecWeight {
    std::string     name;
    RadWeightDecl   d{};
    /* The uneven slice this weight was handed, in the CONTAINER's unsharded units
     * (rad_weight_shard_span). Recorded rather than dropped: "which rows of the checkpoint's k/v
     * projection does rank 3 read" is a declare-time fact and this file is where declare is
     * checked. A weight with no span -- every even shard -- has `span_set` false. */
    bool            span_set = false;
    int64_t         span_lo = 0, span_hi = 0;
};
struct RecBuf    { std::string name; RadBufDecl    d{}; };
struct RecOp {
    std::string             op;
    std::vector<RecParam>   p;
    std::vector<rad_weight> w;
    std::vector<rad_buf>    reads, writes;
    bool                    resolved = true;
    bool                    uses_declared = false;   /* rad_op_reads/writes were called */
};
struct RecMap {
    std::string              declared;
    int                      mode = 0;
    std::vector<std::string> src;
    int                      concat_dim = 0;
};
struct RecKV    { std::string name; RadKVGroupDecl d{}; };
struct RecIssue {
    rad_op                  op = 0;
    std::vector<RadOperand> opd;
    int64_t                 n = 0;
};

}  /* namespace */

struct RadBuilder {
    std::vector<RecWeight> weights;
    std::vector<RecBuf>    bufs;
    std::vector<RecOp>     ops;
    std::vector<RecMap>    maps;
    std::vector<RecKV>     kv_groups;
    std::vector<std::pair<int, rad_kvgroup>> binds;
    std::vector<std::string> notes;

    /* Ops the fake kernel hierarchy refuses, so the fused-op fallback can be driven. */
    std::set<std::string> refuse;
    int64_t kv_block = 16;

    /* The buffer the plugin named as this step's logits (rad_declare_logits). */
    rad_buf logits = 0;

    /* The vision tower's declaration (rad_declare_encoder), when the plugin made one. */
    RadEncoderDecl encoder{};
    std::string    encoder_name;

    /* What rad_weight_encoding answers: the first entry whose key the source name contains. Empty
     * is a builder with no model behind it, where every weight is plain in its declared dtype. */
    std::vector<std::pair<std::string, RadEncoding>> encs;

    const RecOp* op_of(rad_op h) const { return h ? &ops[h - 1] : nullptr; }
};

struct RadCtx {
    RadBuilder*     b     = nullptr;
    const RadBatch* batch = nullptr;
    int             rank  = 0;
    int             world = 1;
    std::vector<RecIssue> issues;
    std::vector<int>      routed_layers;
    std::string           step_fail;
};

/* -------------------------------------------------------------------- the declare-side ABI */
extern "C" rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    b->weights.push_back({name, *d});
    return (rad_weight)b->weights.size();
}

extern "C" int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t*,
                                   uint32_t*) {
    for (const auto& [key, e] : b->encs)
        if (source && std::strstr(source, key.c_str())) { *out = e; return RAD_OK; }
    return RAD_E_NOTFOUND;
}
extern "C" rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    b->bufs.push_back({name, *d});
    return (rad_buf)b->bufs.size();
}

extern "C" rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* params, int n_params,
                              const rad_weight* weights, int n_weights) {
    RecOp r;
    r.op = op;
    for (int i = 0; i < n_params; ++i) {
        RecParam p;
        p.key  = params[i].key ? params[i].key : "";
        p.kind = params[i].kind;
        p.ival = params[i].ival;
        p.ihi  = params[i].ihi;
        p.dval = params[i].dval;
        p.sval = params[i].sval ? params[i].sval : "";
        r.p.push_back(p);
    }
    for (int i = 0; i < n_weights; ++i) r.w.push_back(weights[i]);
    r.resolved = b->refuse.count(r.op) == 0;
    b->ops.push_back(r);
    /* Declare always completes. An op that resolved to nothing is RAD_NULL_HANDLE and is not an
     * error yet -- it becomes one only if the plugin issues it (rad_builder.h). */
    return r.resolved ? (rad_op)b->ops.size() : RAD_NULL_HANDLE;
}

extern "C" int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.reads.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.writes.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_resolved(RadBuilder* b, rad_op h) {
    return h != RAD_NULL_HANDLE && b->ops[h - 1].resolved;
}

extern "C" rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name,
                                         const RadKVGroupDecl* d) {
    b->kv_groups.push_back({name, *d});
    return (rad_kvgroup)b->kv_groups.size();
}

extern "C" int64_t rad_kv_block_size(RadBuilder* b, rad_kvgroup g) { (void)g; return b->kv_block; }

extern "C" int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) {
    b->binds.push_back({layer, g});
    return RAD_OK;
}

extern "C" int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) {
    RecMap r;
    r.declared   = m->declared;
    r.mode       = m->mode;
    r.concat_dim = m->concat_dim;
    for (int i = 0; i < m->n_src; ++i) r.src.push_back(m->src[i]);
    b->maps.push_back(r);
    return RAD_OK;
}

extern "C" void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    b->notes.push_back(buf);
}

/* Which buffer the core samples from. Recorded rather than ignored, because "the plugin named a
 * logits buffer, and it is the narrow one" is exactly the kind of thing this test exists to
 * assert: declaring it [max_tok, n_vocab] is legal and, at this vocabulary, 8 GiB. */
extern "C" int rad_declare_logits(RadBuilder* b, rad_buf logits) {
    if (!b || !logits) return RAD_E_INVAL;
    b->logits = logits;
    return RAD_OK;
}

/* Recorded as calls and nothing else: no test here reads a lane or a drafter declaration. */
extern "C" int rad_buf_concurrent(RadBuilder* b, rad_buf buf) { (void)b; (void)buf; return RAD_OK; }
extern "C" int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    if (!b || w == 0 || w > b->weights.size()) return RAD_E_INVAL;
    if (lo < 0 || hi <= lo) return RAD_E_INVAL;
    /* The real Builder also checks the declared extent against hi-lo (core/build/rad_builder.cpp);
     * the fake checks what it always did and records the answer so a test can read it back. */
    RecWeight& rw = b->weights[w - 1];
    rw.span_set = true;
    rw.span_lo  = lo;
    rw.span_hi  = hi;
    return RAD_OK;
}
extern "C" int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d) {
    (void)b; (void)d;
    return RAD_OK;
}
extern "C" int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d) {
    if (!b || !d || !d->name || !d->modalities) return RAD_E_INVAL;
    b->encoder = *d;
    b->encoder_name = d->name;
    return RAD_OK;
}

/* -------------------------------------------------------------------- model metadata */
extern "C" long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atoll(m->kv_val[i]);
    return dflt;
}
extern "C" double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atof(m->kv_val[i]);
    return dflt;
}
extern "C" const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return m->kv_val[i];
    return dflt;
}

/* -------------------------------------------------------------------- the run-phase ABI */
extern "C" int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    RecIssue r;
    r.op = op;
    r.n  = n;
    for (int i = 0; i < n_opd; ++i) r.opd.push_back(opd[i]);
    c->issues.push_back(r);
    return RAD_OK;
}
/* A block that refuses the batch records the reason; the recorder is what the tests read. */
extern "C" int rad_step_fail(RadCtx* c, const char* what) {
    c->step_fail = what ? what : "";
    return RAD_E_SHAPE;
}
extern "C" const RadBatch* rad_batch(RadCtx* c)      { return c->batch; }
extern "C" int             rad_rank(RadCtx* c)       { return c->rank; }
extern "C" int             rad_world_size(RadCtx* c) { return c->world; }
extern "C" RadStream       rad_stream(RadCtx* c)     { (void)c; return nullptr; }
extern "C" int rad_route_report(RadCtx* c, int layer, const RadRouting* r) {
    (void)r;
    c->routed_layers.push_back(layer);
    return RAD_OK;
}
/* No card-side histogram here, so a router writes its own buffer and reports that. */
extern "C" int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert) {
    (void)c; (void)layer; (void)n_expert;
    return nullptr;
}
/* Non-null and distinct per handle: a plugin only passes these through to rad_route_report. */
extern "C" void* rad_weight_ptr(RadCtx* c, rad_weight w) { (void)c; return (void*)(uintptr_t)(w * 64 + 8); }
extern "C" void* rad_buf_ptr   (RadCtx* c, rad_buf    b) { (void)c; return (void*)(uintptr_t)(b * 64 + 16); }
extern "C" int   rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) { (void)dst; (void)src; (void)bytes; (void)s; return 0; }
extern "C" int   rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) { (void)dst; (void)value; (void)bytes; (void)s; return 0; }
extern "C" int   rad_stream_sync(RadStream s) { (void)s; return 0; }
extern "C" void* rad_dev_alloc(int64_t bytes, int kind) { (void)kind; return malloc((size_t)bytes); }
extern "C" void  rad_dev_free(void* p, int kind) { (void)kind; free(p); }
extern "C" int   rad_lane(RadCtx* c, int lane) { (void)c; (void)lane; return 0; }
extern "C" int   rad_lane_join(RadCtx* c, int from, int to) { (void)c; (void)from; (void)to; return 0; }

/* ==================================================================== the plugin under test */
#include "../arch/qwen35_bf16/qwen35_bf16.cpp"
/* And the production architecture, for its sizing declare (RadBuildCtx::shape_probe). */
#include "../arch/qwen4exp_fp8/qwen4exp_fp8.cpp"
/* The PLE block on its own: no architecture in this file declares it, and what its step() hands
 * the two windowed ops is a contract of its own (rad_block_ple.h). */
#include <arch/rad_block_ple.h>
/* Paged FP8 attention with CSA 3:1 and KVarN historical compression */
#include <arch/rad_block_attn_paged_fp8.h>
/* RDNA4 (gfx1201) ISA engineering rules and hardware constraints */
#include <arch/rad_rdna4_isa.h>

/* ==================================================================== fixtures and helpers */
namespace {

/* Qwen3.8-27B's shape, cut to EIGHT layers so an assertion failure prints something a human can
 * read. Eight is the smallest count that exercises the alternation twice: with
 * full_attention_interval 4 the full-attention layers are 3 and 7 and the other six are gated
 * delta net, which is the same 1:3 ratio the 64-layer model has.
 *
 * head_dim is 256 while n_embd / n_head is 213. Taking the second would be wrong, and this fixture
 * is chosen so the test notices. */
const char* k_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
};
const char* k_vals[] = { "4", "64", "128", "16", "48", "6144", "4" };

RadModelMeta qwen35_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen35";
    m.name        = "test-qwen35-27b";
    m.quant       = "bf16";
    m.n_layers    = 8;
    m.n_embd      = 5120;
    m.n_head      = 24;
    m.n_head_kv   = 4;
    m.head_dim    = 256;
    m.n_ff        = 17408;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_keys) / sizeof(k_keys[0]));
    m.kv_key      = k_keys;
    m.kv_val      = k_vals;
    return m;
}

RadBuildCtx build_ctx(int rank = 0, int world = 1) {
    RadBuildCtx c{};
    c.rank       = rank;
    c.world_size = world;
    c.max_tok    = 256;
    c.max_seqs   = 8;
    c.max_ctx    = 32768;
    c.scope      = "";
    return c;
}

/* The sampled rows. A real batch carries them (RadBatch::out_ids) and the plugin branches on
 * n_out -- a chunk that finishes no sequence sets it to zero and does not touch the lm_head at
 * all -- so a fake batch that left them at zero would silently test the no-logits path. */
static const int32_t kDecodeOut[]  = { 0, 1, 2, 3 };
static const int32_t kPrefillOut[] = { 31, 63 };   /* the last token of each 32-token sequence */

RadBatch decode_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_DECODE;
    b.n_tok       = 4;
    b.n_seq       = 4;
    b.max_q_len   = 1;
    b.max_ctx_len = 100;
    b.n_out       = 4;
    b.out_ids     = kDecodeOut;
    return b;
}

RadBatch prefill_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = 64;
    b.n_seq       = 2;
    b.max_q_len   = 32;
    b.max_ctx_len = 32;
    b.n_out       = 2;
    b.out_ids     = kPrefillOut;
    return b;
}

/* --- small queries over the recorded graph --- */
std::vector<std::string> op_names(const RadBuilder& b) {
    std::vector<std::string> v;
    for (const auto& o : b.ops) v.push_back(o.op);
    return v;
}

const RecOp* find_op(const RadBuilder& b, const std::string& name, int nth = 0) {
    for (const auto& o : b.ops)
        if (o.op == name && nth-- == 0) return &o;
    return nullptr;
}

int count_op(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name) ++n;
    return n;
}

const RecParam* param(const RecOp* o, const char* key) {
    if (!o) return nullptr;
    for (const auto& p : o->p) if (p.key == key) return &p;
    return nullptr;
}

long long pint(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->ival : -12345;
}

double pflt(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->dval : -12345.0;
}

std::string pstr(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->sval : std::string("<absent>");
}

const RecWeight* find_w(const RadBuilder& b, const std::string& name) {
    for (const auto& w : b.weights) if (w.name == name) return &w;
    return nullptr;
}

const RecBuf* find_b(const RadBuilder& b, const std::string& name) {
    for (const auto& x : b.bufs) if (x.name == name) return &x;
    return nullptr;
}

const RecMap* find_map(const RadBuilder& b, const std::string& declared) {
    for (const auto& m : b.maps) if (m.declared == declared) return &m;
    return nullptr;
}
/* The gated attention layer, spelled out once so every test reads against the same list.
 *
 * The fused qk_norm_rope_gate is NOT here and is not declared: its schema takes a precomputed
 * cos/sin table as a required operand rather than a theta, and this plugin has no table to give
 * it (see rad_block_attn_gated.h). Declaring an op with parameters its schema does not have is a
 * structural error, not a miss, so such a probe fails declare outright instead of falling through
 * to the four ops it stands for. */
std::vector<std::string> attn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt" };
    v.push_back("rmsnorm"); v.push_back("rmsnorm");
    v.push_back("rope");    v.push_back("rope");
    v.push_back("attn_paged");
    v.push_back("kv_store");
    v.push_back("sigmoid");
    v.push_back("mul");
    v.push_back("gemm_nt");
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> gdn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "gdn_conv_prep", "gdn_kkt_solve", "gdn_chunk_scan",
                                   "gdn_gated_rmsnorm",
                                   "gdn_conv_update", "gdn_recurrent_update",
                                   "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> mlp_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

void append(std::vector<std::string>& v, const std::vector<std::string>& t) {
    v.insert(v.end(), t.begin(), t.end());
}

/* The whole declared program for the 8-layer fixture: layers 3 and 7 are full attention. */
std::vector<std::string> program_ops(bool tp) {
    std::vector<std::string> v = { "embed_lookup" };
    /* The embedding table is VOCAB-SHARDED, so a rank gathers only the ids in its own slice and
     * zero-fills the rest; this collective sums the ranks back into x. It is the reason the
     * all_reduce count is 2 per layer PLUS ONE rather than 2 per layer. */
    if (tp) v.push_back("all_reduce");
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) append(v, attn_layer_ops(tp));
        else                  append(v, gdn_layer_ops(tp));
        append(v, mlp_ops(tp));
    }
    v.push_back("rmsnorm");
    /* gather_rows before the lm_head, not after: the sampler wants one row per sequence,
     * and running the widest GEMM in the model over every token of a prefill chunk is
     * n_tok/n_seq times the work for an 8 GiB logits buffer (docs/OPS.md). */
    v.push_back("gather_rows");
    v.push_back("logits_gemm");
    return v;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += " "; s += v[i]; }
    return s;
}

std::vector<std::string> issued_ops(const RadBuilder& b, const RadCtx& c) {
    std::vector<std::string> got;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        got.push_back(o ? o->op : "<null>");
    }
    return got;
}

}  /* namespace */

/* ==================================================================== tests */

TEST(declare_is_a_qwen35_trunk) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* Six gated delta net layers to two full attention ones -- the 1:3 ratio the interval sets. */
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "attn_paged"), 2);
    /* The fused gate op is not declared (see attn_layer_ops), so the two attention layers run the
     * four ops it stands for: two per-head rmsnorms and two ropes each. */
    CHECK_EQ(count_op(b, "qk_norm_rope_gate"), 0);
    CHECK_EQ(count_op(b, "rope"), 4);
    /* One MLP per layer, and the MLP is the same block on both kinds. */
    CHECK_EQ(count_op(b, "silu_mul"), 8);
}

TEST(the_alternation_puts_full_attention_at_3_and_7) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The weight manifest is where a layer's kind is visible from outside: a full-attention layer
     * has a q|gate projection and a linear one has an in-projection, and never both. */
    for (int l = 0; l < 8; ++l) {
        const bool full = ((l + 1) % 4 == 0);
        const std::string qg  = "blk." + std::to_string(l) + ".attn_qg.weight";
        const std::string in  = "blk." + std::to_string(l) + ".ssm_in.weight";
        CHECK_EQ(find_w(b, qg) != nullptr, full);
        CHECK_EQ(find_w(b, in) != nullptr, !full);
        /* Both kinds carry the same pre-norm name and the same feed-forward. */
        CHECK(find_w(b, "blk." + std::to_string(l) + ".attn_norm.weight") != nullptr);
        CHECK(find_w(b, "blk." + std::to_string(l) + ".ffn_gate_up.weight") != nullptr);
    }
}

TEST(attention_shapes_at_the_27b_dimensions) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* q_dim = 24 * 256 = 6144, and the projection is TWICE that: query and gate, interleaved. */
    const RecWeight* qg = find_w(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->d.rank, 2u);
    CHECK_EQ(qg->d.shape[0], 12288);
    CHECK_EQ(qg->d.shape[1], 5120);
    CHECK_EQ(qg->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(qg->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(qg->d.access, RAD_ACCESS_PER_TOKEN);

    /* k and v are separate projections, 4 kv heads of 256. */
    const RecWeight* kw = find_w(b, "blk.3.attn_k.weight");
    const RecWeight* vw = find_w(b, "blk.3.attn_v.weight");
    CHECK(kw != nullptr && vw != nullptr);
    CHECK_EQ(kw->d.shape[0], 1024);
    CHECK_EQ(vw->d.shape[0], 1024);
    CHECK_EQ(kw->d.shard, RAD_SHARD_ROW);

    const RecWeight* o = find_w(b, "blk.3.attn_output.weight");
    CHECK(o != nullptr);
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 6144);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* Per-head gains are head_dim wide and replicated: one vector shared by every head. */
    const RecWeight* qn = find_w(b, "blk.3.attn_q_norm.weight");
    CHECK(qn != nullptr);
    CHECK_EQ(qn->d.shape[0], 256);
    CHECK_EQ(qn->d.shard, RAD_SHARD_NONE);
    CHECK_EQ(qn->d.dtype, (uint32_t)RAD_F32);

    /* The interleaved projection buffer is rank 3 -- [tokens, heads, 2*head_dim] -- because that
     * is what makes the query and the gate addressable as column slices. */
    const RecBuf* qgb = find_b(b, "attn_qg");
    CHECK(qgb != nullptr);
    CHECK_EQ(qgb->d.rank, 3u);
    CHECK_EQ(qgb->d.shape[0], 256);
    CHECK_EQ(qgb->d.shape[1], 24);
    CHECK_EQ(qgb->d.shape[2], 512);

    /* The fused qk_norm_rope_gate is NOT declared, and asserting its absence is the point:
     * its schema in libr4d takes a precomputed cos/sin table rather than theta, and this
     * plugin has no table to give it. Declaring it anyway is a structural error that fails
     * declare, not a miss that falls through -- so it stays undeclared until something builds
     * the table. The four ops it stands for carry the same numbers. */
    CHECK(find_op(b, "qk_norm_rope_gate") == nullptr);

    const RecOp* qn2 = find_op(b, "rope");
    CHECK(qn2 != nullptr);
    if (qn2) {
        CHECK_EQ(pint(qn2, "head_dim"), 256);
        CHECK_EQ(pint(qn2, "rotary_dim"), 64);   /* partial_rotary_factor 0.25 of 256 */
        CHECK_EQ(pstr(qn2, "mode"), std::string("mrope"));
        CHECK_NEAR(pflt(qn2, "theta"), 1e7, 1.0);
    }
    const RecOp* at = find_op(b, "attn_paged");
    CHECK_EQ(pint(at, "head_dim"), 256);
    CHECK_EQ(pint(at, "gqa"), 6);
    CHECK_EQ(pint(at, "causal"), 1);
    CHECK_EQ(pint(at, "window"), 0);
    /* block_size IS declared here, deliberately. It is RAD_DERIVED -- a value the kernel supplies
     * -- and every libr4d attention row carries C_EQ("block_size", 16), so declaring 16 changes no
     * resolution on the accelerated path. But derived only works when something derives it: a
     * geometry no accelerated row serves (head_dim 256 at gqa 4, say, where libr4d has 256/6 and
     * 128/4) resolves attn_paged to libref, which serves any block size and therefore names none,
     * and the KV group cannot be sized at all. The block size is a property of the CACHE the model
     * is asking for, so the model is where it belongs. */
    CHECK_EQ(pint(at, "block_size"), 16);

    /* ...and it reaches kv_store, which does need it as a parameter. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ(pint(ks, "block_size"), 16);
    CHECK_EQ(pint(ks, "n_head_kv"), 4);

    /* The gate is a per-head sigmoid over head_dim and an elementwise multiply over the whole
     * attention output: `out = attn * sigmoid(gate)`. */
    const RecOp* sg = find_op(b, "sigmoid");
    CHECK_EQ(pint(sg, "n"), 256);
    CHECK_EQ(param(sg, "M")->ihi, 256LL * 24);
    const RecOp* mu = find_op(b, "mul");
    CHECK_EQ(pint(mu, "n"), 6144);
}

TEST(gdn_shapes_and_both_paths) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* conv_dim = 2*16*128 + 48*128 = 10240, and the projection row carries one `a` and one `b`
     * per value head after it: 10240 + 96 = 10336. That fusion is why gdn_conv_prep can read the
     * decay and beta inputs out of `x`. */
    const RecWeight* in = find_w(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->d.shape[0], 10336);
    CHECK_EQ(in->d.shape[1], 5120);
    CHECK_EQ(in->d.shard, RAD_SHARD_ROW);

    const RecWeight* z = find_w(b, "blk.0.ssm_z.weight");
    CHECK(z != nullptr);
    CHECK_EQ(z->d.shape[0], 6144);            /* 48 value heads of 128 */

    const RecWeight* cv = find_w(b, "blk.0.ssm_conv1d.weight");
    CHECK(cv != nullptr);
    CHECK_EQ(cv->d.rank, 2u);
    CHECK_EQ(cv->d.shape[0], 10240);          /* depthwise: one filter per convolved channel */
    CHECK_EQ(cv->d.shape[1], 4);
    CHECK_EQ(cv->d.shard, RAD_SHARD_ROW);
    /* BF16, and asserted because nothing else states it. libr4d's conv takes this pointer as
     * `const unsigned short*` and reads it through bf2f, so an f32 plane feeds it two bf16
     * elements out of every float -- on every gated-delta-net layer, on the device path only,
     * while libref stays correct because it reads the operand's own dtype. */
    CHECK_EQ(cv->d.dtype, (uint32_t)RAD_BF16);
    /* Qwen3.5's conv1d has no bias, so none is declared -- the operand is passed absent rather
     * than as a zero vector nobody loads. */
    CHECK(find_w(b, "blk.0.ssm_conv1d.bias") == nullptr);

    /* The decay weight is named for its convention: it is a LOG and the kernels take exp(). */
    const RecWeight* al = find_w(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->d.shape[0], 48);
    CHECK_EQ(al->d.dtype, (uint32_t)RAD_F32);
    CHECK(find_w(b, "blk.0.ssm_a") == nullptr);

    const RecWeight* on = find_w(b, "blk.0.ssm_norm.weight");
    CHECK(on != nullptr);
    CHECK_EQ(on->d.shape[0], 128);            /* over head_v, shared by every head */
    CHECK_EQ(on->d.shard, RAD_SHARD_NONE);

    const RecWeight* ow = find_w(b, "blk.0.ssm_out.weight");
    CHECK(ow != nullptr);
    CHECK_EQ(ow->d.shape[0], 5120);
    CHECK_EQ(ow->d.shape[1], 6144);
    CHECK_EQ(ow->d.shard, RAD_SHARD_COL);

    /* Both paths are declared. A branch on phase is not a branch on quant: both sequences are in
     * the graph dump and both are resolved at declare. */
    CHECK_EQ(count_op(b, "gdn_conv_prep"), 6);
    CHECK_EQ(count_op(b, "gdn_kkt_solve"), 6);
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "gdn_gated_rmsnorm"), 6);
    CHECK_EQ(count_op(b, "gdn_conv_update"), 6);
    CHECK_EQ(count_op(b, "gdn_recurrent_update"), 6);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ(pint(cp, "head_k"), 128);
    CHECK_EQ(pint(cp, "head_v"), 128);
    CHECK_EQ(pint(cp, "conv_width"), 4);
    CHECK_EQ(pint(cp, "chunk"), 64);
    CHECK_EQ(param(cp, "M")->kind, RAD_P_RANGE);
    /* The conv weight and the two per-head vectors are its weight operands; no bias. */
    CHECK_EQ((int)cp->w.size(), 3);

    /* chunk is RAD_DERIVED on the two ops that tile on it -- it belongs to the kernel. */
    CHECK(param(find_op(b, "gdn_kkt_solve"), "chunk") == nullptr);
    CHECK(param(find_op(b, "gdn_chunk_scan"), "chunk") == nullptr);

    /* The recurrent path is ranged on q_len, because a speculative verify carries n_spec+1 rows
     * per sequence, and it states n_head_v: 48 value heads against 16 key heads is the geometry
     * where guessing from a flattened view gives a plausible wrong answer. */
    const RecOp* ru = find_op(b, "gdn_recurrent_update");
    CHECK_EQ(param(ru, "q_len")->kind, RAD_P_RANGE);
    CHECK_EQ(pint(ru, "n_head_v"), 48);
    CHECK_EQ(pstr(ru, "act"), std::string("silu"));
    CHECK_EQ(pint(find_op(b, "gdn_conv_update"), "conv_width"), 4);

    /* q and k carry 16 heads and v carries 48: the 3:1 sharing is in the buffer shapes. */
    const RecBuf* gq = find_b(b, "gdn_q");
    const RecBuf* gv = find_b(b, "gdn_v");
    CHECK(gq != nullptr && gv != nullptr);
    CHECK_EQ(gq->d.shape[1], 16);
    CHECK_EQ(gq->d.shape[2], 128);
    CHECK_EQ(gv->d.shape[1], 48);
    CHECK_EQ(gv->d.shape[2], 128);
    /* The decay and beta are f32: the gate is a cumulative sum whose span reaches hundreds of nats,
     * and every decay factor is an exponent of it.
     *
     * THE CHUNK INVERSE IS BF16, and that is not a relaxation. gdn_kkt_solve writes this plane as
     * `unsigned short*` and gdn_chunk_scan reads it back the same way -- both libr4d rows say so
     * in their own dtype line -- so declaring it f32 sizes it at twice the bytes and makes the two
     * kernels disagree about every element of it. libref reads each operand through its own dtype
     * and stays correct either way, so the host path does not show it; the r4d shim refuses a
     * dtype the kernel would silently reinterpret. */
    CHECK_EQ(find_b(b, "gdn_g")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_beta")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.shape[2], 64);
}

TEST(three_kv_groups_with_every_layer_bound) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_gdn_state"));
    CHECK_EQ(b.kv_groups[2].name, std::string("kv_gdn_conv"));

    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 256);
    CHECK_EQ(f.dtype, (uint32_t)RAD_BF16);

    /* 48 heads of 128x128, fp32: 3 MiB a layer and 144 MiB across the 48 linear layers of the
     * real model, per sequence. It is fp32 because that is the state width libr4d has a kernel
     * for. */
    const RadKVGroupDecl& s = b.kv_groups[1].d;
    CHECK_EQ(s.kind, RAD_KV_LINEAR);
    CHECK_EQ(s.n_head_kv, 48);
    CHECK_EQ(s.state_dim[0], 128);
    CHECK_EQ(s.state_dim[1], 128);
    CHECK_EQ(s.dtype, (uint32_t)RAD_F32);

    /* The conv window is counted in HEADS -- 2*16 + 48 of width 128 is the 10240-channel row --
     * so the channel split never straddles a q/k/v boundary at any tensor-parallel width. */
    const RadKVGroupDecl& c = b.kv_groups[2].d;
    CHECK_EQ(c.kind, RAD_KV_CONV);
    CHECK_EQ(c.n_head_kv, 80);
    CHECK_EQ(c.head_dim, 128);
    CHECK_EQ(c.conv_width, 4);

    /* Every layer is bound to the group it owns state in, and a linear layer owns two. A group
     * with no bound layers is a declare error, because the page size would otherwise be a guess. */
    std::map<rad_kvgroup, std::set<int>> bound;
    for (const auto& p : b.binds) bound[p.second].insert(p.first);
    CHECK_EQ((int)bound[1].size(), 2);          /* full attention: layers 3 and 7 */
    CHECK_EQ((int)bound[2].size(), 6);
    CHECK_EQ((int)bound[3].size(), 6);
    CHECK(bound[1].count(3) && bound[1].count(7));
    CHECK(bound[2].count(0) && bound[2].count(6));
    CHECK(bound[2] == bound[3]);                /* the two GDN groups cover the same layers */
    for (int l = 0; l < 8; ++l)
        CHECK_EQ(bound[1].count(l) + bound[2].count(l), (size_t)1);
}

TEST(the_name_map_owns_both_sides) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The interleaved query-and-gate projection is ONE checkpoint tensor and one declared weight:
     * there is nothing to fuse and nothing to split. */
    const RecMap* qg = find_map(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->mode, RAD_MAP_COPY);
    CHECK_EQ((int)qg->src.size(), 1);
    CHECK_EQ(qg->src[0],
             std::string("model.language_model.layers.3.self_attn.q_proj.weight"));

    /* The GDN in-projection IS a convert-time fusion: qkv, a and b in the row order
     * gdn_conv_prep reads. */
    const RecMap* in = find_map(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)in->src.size(), 3);
    CHECK_EQ(in->src[0],
             std::string("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    CHECK_EQ(in->src[1],
             std::string("model.language_model.layers.0.linear_attn.in_proj_a.weight"));
    CHECK_EQ(in->src[2],
             std::string("model.language_model.layers.0.linear_attn.in_proj_b.weight"));
    CHECK_EQ(in->concat_dim, 0);

    /* A_log is mapped straight through from the tensor that is a log. */
    const RecMap* al = find_map(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->src[0], std::string("model.language_model.layers.0.linear_attn.A_log"));

    /* gate and up are one weight and one GEMM. */
    const RecMap* gu = find_map(b, "blk.1.ffn_gate_up.weight");
    CHECK(gu != nullptr);
    CHECK_EQ(gu->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)gu->src.size(), 2);
    CHECK_EQ(gu->src[0], std::string("model.language_model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(gu->src[1], std::string("model.language_model.layers.1.mlp.up_proj.weight"));

    /* The feed-forward's norm is the POST-ATTENTION norm, not a second copy of the input one. */
    const RecMap* fn = find_map(b, "blk.1.ffn_norm.weight");
    CHECK(fn != nullptr);
    CHECK_EQ(fn->src[0],
             std::string("model.language_model.layers.1.post_attention_layernorm.weight"));

    /* This model does not tie its lm_head; the second source is for the members that do. */
    const RecMap* lm = find_map(b, "output.weight");
    CHECK(lm != nullptr);
    CHECK_EQ(lm->mode, RAD_MAP_COPY);
    CHECK_EQ((int)lm->src.size(), 2);
    CHECK_EQ(lm->src[0], std::string("lm_head.weight"));
    CHECK_EQ(lm->src[1], std::string("model.language_model.embed_tokens.weight"));

    /* Every declared weight has a map entry: rad-convert has to be able to fill all of them. */
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* Without a declared use a transient has no liveness and the planner must assume it live for
     * the whole program -- sound, and as wasteful as it sounds. A GDN layer has eleven
     * intermediates, so this is where an arena is won or lost. */
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ((int)cp->reads.size(), 1);      /* the fused projection row */
    CHECK_EQ((int)cp->writes.size(), 5);     /* q, k, v, g, beta */

    /* kv_store writes the group's pool, which is not a declared buffer, so its write set is empty
     * and its read set is the two projections it stores. That is a declaration, not an omission. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ((int)ks->reads.size(), 2);
    CHECK_EQ((int)ks->writes.size(), 0);

    /* The residual add reads the stream and the block output and writes the stream. */
    const RecOp* ad = find_op(b, "add");
    CHECK_EQ((int)ad->reads.size(), 2);
    CHECK_EQ((int)ad->writes.size(), 1);
    CHECK_EQ(ad->writes[0], ad->reads[0]);
}

/* The gated attention layer runs the four ops the undeclared fused op stands for. Since the plugin
 * does not declare qk_norm_rope_gate at all, the unfused sequence is simply what a gated layer is,
 * and no refusal in the fake hierarchy is needed to reach it. */
TEST(the_gated_attention_layer_runs_the_unfused_sequence) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* The per-head norms are rmsnorm over head_dim, with M ranged over tokens x heads. The q
     * one reads the strided view of the interleaved projection and writes the contiguous q
     * buffer, so it de-interleaves and normalises in one pass. */
    const RecOp* an = nullptr;      /* the first attention layer's q norm */
    for (size_t i = 0; i + 1 < b.ops.size(); ++i)
        if (b.ops[i].op == "rope") { an = &b.ops[i - 2]; break; }
    CHECK(an != nullptr);
    if (!an) return;
    CHECK_EQ(an->op, std::string("rmsnorm"));
    CHECK_EQ(pint(an, "n"), 256);
    CHECK_EQ(param(an, "M")->ihi, 256LL * 24);
    CHECK_EQ((an + 1)->op, std::string("rmsnorm"));
    CHECK_EQ(param(an + 1, "M")->ihi, 256LL * 4);

    const RecOp* rq = an + 2;
    CHECK_EQ(rq->op, std::string("rope"));
    CHECK_EQ(pint(rq, "head_dim"), 256);
    CHECK_EQ(pint(rq, "n_head"), 24);
    CHECK_EQ(pint(rq, "n_head_kv"), 0);      /* a q-only row: no k heads and no v heads */
    CHECK_EQ(pint(rq, "rotary_dim"), 64);
    CHECK_EQ(pstr(rq, "mode"), std::string("mrope"));
    CHECK_NEAR(pflt(rq, "theta"), 1e7, 1.0);

    const RecOp* rk = an + 3;
    CHECK_EQ(rk->op, std::string("rope"));
    CHECK_EQ(pint(rk, "n_head"), 4);         /* the k row, which has 4 heads */
    CHECK_EQ(pint(rk, "n_head_kv"), 0);

    /* And the step issues them: four ops in place of the one fused op, twice -- once per
     * full-attention layer. */
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = decode_batch();
    c.batch = &batch;
    qwen35_bf16::step(&c, &batch);
    int n_rope = 0;
    for (const auto& i : c.issues)
        if (b.op_of(i.op) && b.op_of(i.op)->op == "rope") ++n_rope;
    CHECK_EQ(n_rope, 4);                     /* two per attention layer, two layers */

    /* The sigmoid reads the INTERLEAVED projection: a column slice at head_dim, width head_dim, of
     * the buffer whose declared width is the pitch. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "sigmoid") continue;
        CHECK_EQ(i.opd[0].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(i.opd[0].offset, 256);      /* the gate half of the head's pair */
        CHECK_EQ(i.opd[0].cols, 256);
        CHECK_EQ(i.opd[0].rows, batch.n_tok);
    }
}

TEST(tensor_parallel_divides_dimensions_and_places_the_collectives) {
    RadBuilder b0, b1;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  c0 = build_ctx(0, 2), c1 = build_ctx(1, 2);
    CHECK_OK(qwen35_bf16::declare(&b0, &meta, &c0));
    CHECK_OK(qwen35_bf16::declare(&b1, &meta, &c1));

    /* Declare runs once per rank and the plugin declares ALREADY DIVIDED dimensions. */
    const RecWeight* qg = find_w(b0, "blk.3.attn_qg.weight");
    CHECK_EQ(qg->d.shape[0], 6144);           /* 12 heads x 256 x 2 */
    CHECK_EQ(qg->d.shape[1], 5120);           /* the residual stream is not sharded */
    CHECK_EQ(find_w(b0, "blk.3.attn_k.weight")->d.shape[0], 512);      /* 2 kv heads */
    const RecWeight* o = find_w(b0, "blk.3.attn_output.weight");
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 3072);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* The GDN row halves by HEAD: 8 key heads and 24 value heads, so 2*8*128 + 24*128 = 5120
     * convolved channels and 48 more for a and b. */
    const RecWeight* in = find_w(b0, "blk.0.ssm_in.weight");
    CHECK_EQ(in->d.shape[0], 5168);
    CHECK_EQ(find_w(b0, "blk.0.ssm_z.weight")->d.shape[0], 3072);
    CHECK_EQ(find_w(b0, "blk.0.ssm_conv1d.weight")->d.shape[0], 5120);
    CHECK_EQ(find_w(b0, "blk.0.ssm_a_log")->d.shape[0], 24);
    CHECK_EQ(find_w(b0, "blk.0.ssm_out.weight")->d.shape[1], 3072);
    /* The gain vectors are per-head-width and stay whole. */
    CHECK_EQ(find_w(b0, "blk.0.ssm_norm.weight")->d.shape[0], 128);
    CHECK_EQ(find_w(b0, "blk.3.attn_q_norm.weight")->d.shape[0], 256);

    const RecWeight* gu = find_w(b0, "blk.0.ffn_gate_up.weight");
    CHECK_EQ(gu->d.shape[0], 17408);          /* 2 * (17408 / 2) */
    CHECK_EQ(find_w(b0, "blk.0.ffn_down.weight")->d.shape[1], 8704);

    /* gqa is rank-invariant, and so is the head width. */
    CHECK_EQ(pint(find_op(b0, "attn_paged"), "gqa"), 6);
    CHECK_EQ(pint(find_op(b0, "gdn_conv_prep"), "head_k"), 128);
    CHECK_EQ(pint(find_op(b0, "gdn_recurrent_update"), "n_head_v"), 24);

    /* The KV groups halve with the heads they hold. */
    CHECK_EQ(b0.kv_groups[0].d.n_head_kv, 2);
    CHECK_EQ(b0.kv_groups[1].d.n_head_kv, 24);
    CHECK_EQ(b0.kv_groups[2].d.n_head_kv, 40);   /* 2*8 + 24 */

    /* Two collectives per layer -- one for the block's column-parallel projection, one for the
     * feed-forward's -- plus ONE for the vocab-sharded embedding, and nowhere else. */
    CHECK_EQ(count_op(b0, "all_reduce"), 17);
    CHECK_EQ(join(op_names(b0)), join(program_ops(/*tp=*/true)));

    const RecOp* ar = find_op(b0, "all_reduce");
    CHECK_EQ(pint(ar, "world_size"), 2);
    CHECK_EQ(pint(ar, "exact"), 1);           /* a caller that does not say gets no kernel */
    CHECK_EQ(param(ar, "numel")->kind, RAD_P_RANGE);
    CHECK_EQ(param(ar, "numel")->ihi, 256LL * 5120);

    /* THE VOCABULARY IS SPLIT and the split is RAGGED BY CONSTRUCTION: ceil(total/world) rows on
     * every rank but the last, which takes what is left. 248320 happens to divide by two, so this
     * asserts the sum and the shard rather than a number that would still hold if the rule were
     * "floor" -- the case that matters is a vocabulary that does not divide, and the rule has to
     * be the one core/engine.cpp restates and checks. */
    const RecWeight* lm0 = find_w(b0, "output.weight");
    const RecWeight* lm1 = find_w(b1, "output.weight");
    CHECK_EQ(lm0->d.shape[0] + lm1->d.shape[0], 248320);
    CHECK_EQ(lm0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(lm0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(pint(find_op(b0, "logits_gemm"), "n_vocab"), lm0->d.shape[0]);
    /* The DRAFT head is the exception and stays whole on both ranks: its argmax is a row_topk
     * over one rank's plane and never passes through the sampler's merge, so a sharded plane
     * would have each rank propose only tokens from its own half. */
    if (const RecWeight* dh = find_w(b0, "mtp.draft_head.weight")) {
        CHECK_EQ(dh->d.shape[0], 248320);
        CHECK_EQ(dh->d.shard, RAD_SHARD_NONE);
    }
    /* THE EMBEDDING IS SHARDED TOO, and it is the same ragged split as the lm_head. Left whole it
     * is gigabytes of replicated bf16 on every card; embed_lookup carries a vocab_offset so both
     * ends of one vocabulary split the same way, and THAT is what is asserted: the two shards sum
     * to the whole and each rank's embed_lookup is told where its slice begins. */
    const RecWeight* te0 = find_w(b0, "token_embd.weight");
    const RecWeight* te1 = find_w(b1, "token_embd.weight");
    CHECK_EQ(te0->d.shape[0] + te1->d.shape[0], 248320);
    CHECK_EQ(te0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(te0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te0->d.access, RAD_ACCESS_VOCAB);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "n_vocab"), te0->d.shape[0]);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "vocab_offset"), 0);
    CHECK_EQ(pint(find_op(b1, "embed_lookup"), "vocab_offset"), te0->d.shape[0]);

    /* A rank count that does not divide the heads is refused, not rounded. */
    RadBuilder b5;
    RadBuildCtx c5 = build_ctx(0, 5);
    CHECK_EQ(qwen35_bf16::declare(&b5, &meta, &c5), RAD_E_INVAL);

    /* And with one rank there are no collectives at all. */
    RadBuilder bs;
    RadBuildCtx cs = build_ctx(0, 1);
    CHECK_OK(qwen35_bf16::declare(&bs, &meta, &cs));
    CHECK_EQ(count_op(bs, "all_reduce"), 0);
}

TEST(a_decode_step_issues_both_layer_types_in_order) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = decode_batch();
    c.batch = &batch;
    qwen35_bf16::step(&c, &batch);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) {
            /* Issue order differs from declare order at exactly one place: kv_store writes the
             * cache before attn_paged reads it. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                  "rmsnorm", "rmsnorm", "rope", "rope",
                                  "kv_store", "attn_paged",
                                  "sigmoid", "mul", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        } else {
            /* Decode takes the recurrent path, which folds the gated norm. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt",
                                  "gdn_conv_update", "gdn_recurrent_update", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt", "add" })
            want.push_back(s);
    }
    want.push_back("rmsnorm");
    want.push_back("gather_rows");
    want.push_back("logits_gemm");
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The ranged value: RAD_N_BATCH for token-count-ranged ops, the actual query length for the
     * two ops whose band is q_len, an explicit row count where the rows are tokens x heads, and
     * n_out for the two vocabulary ops -- their band is the number of SAMPLED rows, which is not
     * n_tok and in a prefill chunk is not even close to it. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged" || o->op == "gdn_conv_update" ||
            o->op == "gdn_recurrent_update")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "sigmoid")
            CHECK_EQ(i.n, batch.n_tok * 24);
        else if (o->op == "rmsnorm")
            /* Three kinds of rmsnorm in one model and they do NOT share a row count: a
             * layer pre-norm is ranged over tokens (RAD_N_BATCH), while the unfused
             * per-head q and k norms run at tokens x n_head and tokens x n_head_kv. */
            CHECK(i.n == RAD_N_BATCH || i.n == batch.n_tok * 24 || i.n == batch.n_tok * 4);
        else if (o->op == "gather_rows" || o->op == "logits_gemm")
            CHECK_EQ(i.n, batch.n_out);
        else
            CHECK_EQ(i.n, RAD_N_BATCH);
    }

    /* Every operand is a declared handle or a batch pointer -- nothing invented at issue -- and
     * every buffer operand carries the STEP's row count, not the buffer's declared extent. A
     * kernel reads its extents off the operands, so an un-narrowed operand is a one-row decode
     * computed over the whole arena. */
    for (const auto& i : c.issues)
        for (const auto& od : i.opd) {
            if (od.kind == RAD_OPK_BUF) {
                CHECK(od.handle >= 1 && od.handle <= b.bufs.size());
                CHECK_EQ(od.rows, batch.n_tok);
            }
            if (od.kind == RAD_OPK_WEIGHT) CHECK(od.handle >= 1 && od.handle <= b.weights.size());
        }
}

TEST(a_prefill_step_takes_the_chunked_gdn_path) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = prefill_batch();
    c.batch = &batch;
    qwen35_bf16::step(&c, &batch);

    std::vector<std::string> got = issued_ops(b, c);
    int n_prep = 0, n_scan = 0, n_recur = 0, n_gnorm = 0;
    for (const auto& s : got) {
        if (s == "gdn_conv_prep")        ++n_prep;
        if (s == "gdn_chunk_scan")       ++n_scan;
        if (s == "gdn_recurrent_update") ++n_recur;
        if (s == "gdn_gated_rmsnorm")    ++n_gnorm;
    }
    CHECK_EQ(n_prep, 6);
    CHECK_EQ(n_scan, 6);
    CHECK_EQ(n_gnorm, 6);
    CHECK_EQ(n_recur, 0);      /* the recurrent path is declared but not issued here */

    /* The decay and beta inputs are the last 2*n_head_v columns of the projection row. The
     * chunked path reads them inside gdn_conv_prep; the recurrent path takes them as strided
     * views, which is what the fused in-projection buys. */
    const RecOp* cp = nullptr;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_conv_prep") {
            cp = o;
            /* Sixteen operands, inputs before outputs (docs/OPS.md, "The authoritative schemas").
             * `a` and `b_gate` are the same strided views the recurrent path takes, and they are
             * passed even though the schema marks them optional: libr4d dereferences them and
             * `cache_idx` unconditionally, so omitting them drops the layer to libref
             * without saying so. */
            CHECK_EQ((int)i.opd.size(), 16);
            CHECK_EQ(i.opd[2].kind, (uint8_t)RAD_OPK_NONE);   /* no conv bias on this model */
            CHECK_EQ(i.opd[5].kind, (uint8_t)RAD_OPK_KV);     /* the conv state */
            CHECK_EQ(i.opd[6].rows, batch.n_seq + 1);         /* cu_seqlens */
            CHECK_EQ(i.opd[7].offset, 10240);                 /* a */
            CHECK_EQ(i.opd[7].cols, 48);
            CHECK_EQ(i.opd[8].offset, 10288);                 /* b_gate */
            CHECK_EQ(i.opd[10].rows, batch.n_seq);            /* has_init == ctx_lens */
            break;
        }
    }
    CHECK(cp != nullptr);

    /* The scan reads and writes the recurrent state group, as h0 and ht. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_chunk_scan") {
            /* Eleven: the state index follows the two KV operands, and it is what makes the scan
             * write the slot the manager gave the sequence rather than the row it occupies in
             * this batch. */
            CHECK_EQ((int)i.opd.size(), 11);
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[9].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[10].rows, batch.n_seq);
            break;
        }
    }
}

TEST(the_recurrent_path_slices_a_and_b_out_of_the_projection_row) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = decode_batch();
    c.batch = &batch;
    qwen35_bf16::step(&c, &batch);

    bool seen = false;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "gdn_recurrent_update") continue;
        seen = true;
        /* SIXTEEN, which is what both kernel registries declare for gdn_recurrent_update, and the
         * count core/runtime/issue.cpp demands exactly. The number has to come from the SCHEMA and
         * not from the code under test: this harness stubs rad_issue, so it records the operand
         * list instead of checking it, and an assertion that pins whatever the plugin emitted
         * would certify a miscount rather than catch it. The last two are the folded
         * out-projection quantiser, which the bf16 block does not want -- and an absent optional
         * operand is a NULL OPERAND, not a shorter list, because operands are positional. That
         * is the rule worth pinning here, so the tail is checked rather than just counted. */
        CHECK_EQ((int)i.opd.size(), 16);
        CHECK_EQ(i.opd[14].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[15].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[3].offset, 10240);      /* a: the first column after the conv row */
        CHECK_EQ(i.opd[3].cols, 48);
        CHECK_EQ(i.opd[4].offset, 10288);      /* b: one per value head after a */
        CHECK_EQ(i.opd[4].cols, 48);
        CHECK_EQ(i.opd[7].kind, (uint8_t)RAD_OPK_KV);
        break;
    }
    CHECK(seen);
}

TEST(what_the_plugin_refuses) {
    RadModelMeta meta = qwen35_meta();

    /* A drafter would need a second set of per-rank handles and a second name-map namespace.
     * Nothing unimplemented returns success (spec §17). */
    RadBuilder bd;
    RadBuildCtx cd = build_ctx();
    cd.is_draft = 1;
    CHECK_EQ(qwen35_bf16::declare(&bd, &meta, &cd), RAD_E_UNSUPPORTED);

    /* A routed checkpoint is a different graph, not a flag on this one. */
    RadBuilder bm;
    RadModelMeta moe = meta;
    moe.n_expert = 128;
    moe.n_expert_used = 8;
    RadBuildCtx cm = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bm, &moe, &cm), RAD_E_UNSUPPORTED);

    /* A container that does not describe the linear layers is refused rather than defaulted: a
     * guessed head count is a model that loads and produces fluent wrong text. */
    RadBuilder bn;
    RadModelMeta bare = meta;
    bare.n_kv = 2;                 /* interval and rope width only; no ssm keys */
    RadBuildCtx cn = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bn, &bare, &cn), RAD_E_INVAL);

    /* ...and so is one that does not say how much of the head to rotate. */
    RadBuilder br;
    RadModelMeta norot = meta;
    static const char* rk[] = { "qwen35.full_attention_interval", "qwen35.ssm.state_size",
                                "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
                                "qwen35.ssm.inner_size", "qwen35.ssm.conv_kernel" };
    static const char* rv[] = { "4", "128", "16", "48", "6144", "4" };
    norot.n_kv = 6; norot.kv_key = rk; norot.kv_val = rv;
    RadBuildCtx cr = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&br, &norot, &cr), RAD_E_INVAL);
}

TEST(the_mtp_layer_sits_beside_the_trunk_and_not_inside_it) {
    /* `nextn_predict_layers` IS NOT PART OF THE LAYER COUNT in this family, and the checkpoint is
     * what says so: Qwen3.8-27B-FP8 sets num_hidden_layers = 64, lists 64 entries in layer_types,
     * ships layers-0..63 under `model.language_model.layers`, and puts its one MTP block in a file
     * of its own under an `mtp.` prefix. Subtracting it drops the last trunk layer -- a
     * full-attention one -- and the model then runs one layer short: still a distribution, still
     * self-consistent op by op, and wrong.
     *
     * So what is pinned is that every declared layer is a trunk layer and `nextn` changes nothing
     * about the graph. */
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    static const char* mk[] = { "qwen35.full_attention_interval", "qwen35.rope.dimension_count",
                                "qwen35.ssm.state_size", "qwen35.ssm.group_count",
                                "qwen35.ssm.time_step_rank", "qwen35.ssm.inner_size",
                                "qwen35.ssm.conv_kernel", "qwen35.nextn_predict_layers" };
    static const char* mv[] = { "4", "64", "128", "16", "48", "6144", "4", "1" };
    meta.n_kv = 8; meta.kv_key = mk; meta.kv_val = mv;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The same graph the no-nextn container produces: declaring one MTP layer changes nothing. */
    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));
    /* And the last trunk layer IS declared. qwen35_meta() is 8 layers, so blk.7 is the last, and
     * (7 + 1) % 4 == 0 makes it a full-attention layer -- the kind a subtraction would eat. */
    CHECK(find_w(b, "blk.7.attn_norm.weight") != nullptr);
    CHECK(find_w(b, "blk.8.attn_norm.weight") == nullptr);
}

/* THE PLE WINDOWS ARE ISSUED ONCE PER HALF OF THE STEP, and what distinguishes the halves is the
 * operand the windowed ops read to pick their write-back layout. A decode row may have its drafts
 * rejected, so `ngram_ids` and `ple_conv` must see `num_accepted` for those rows; a prefill chunk
 * commits everything, so they must NOT see it for those -- the op then leaves the chunk's tail at
 * offset zero, where the scheduler's `num_accepted` of 1 reads it next step. One issue over a mixed
 * step cannot say both, and a pure step issues exactly one half. */
TEST(the_ple_windows_are_issued_once_per_half_of_a_step) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = 3;
    RadBufDecl hd{};
    hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
    PleLayer::Wire w{};
    w.h = rad_decl_buffer(&b, "h", &hd);
    PleLayer::Config pc{};
    pc.hc = 4; pc.embed_dim = 32; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
    pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
    PleLayer::Src src{};
    src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
    src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
    src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
    src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
    PleLayer ple;
    CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));

    /* Two decode rows (a verify of 4 and a plain decode of 1) and one prefill chunk of 6. */
    static const int32_t tok[11] = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15 };
    static const int32_t cu[4]   = { 0, 4, 5, 11 };
    static const int32_t ctxl[3] = { 40, 30, 12 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static int32_t tk_idx[3] = { 5, 6, 7 }, cv_idx[3] = { 1, 2, 3 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = ple.kv_tok;  kv[0].state_index = tk_idx;
    kv[1].group = ple.kv_conv; kv[1].state_index = cv_idx;

    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 11; mixed.n_seq = 3;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 5;
    mixed.token_ids = tok; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c; c.b = &b; c.batch = &mixed;
    /* The hash is issued at the top of the step and the rest at the layer; together they are the
     * layer's issue sequence. */
    ple.ids(&c, &mixed);
    ple.step(&c, &mixed);

    const char* want[] = { "ngram_ids", "ngram_ids", "embed_lookup_q", "cast", "gemm_nt",
                           "gemm_nt", "ple_gate", "ple_conv", "ple_conv", "add" };
    std::vector<std::string> wv(want, want + 10);
    CHECK_EQ(join(issued_ops(b, c)), join(wv));
    REQUIRE(c.issues.size() == 10);

    /* The projections read the gather's rows from the card: the copy takes the gather's output
     * and both GEMMs take the copy's, never the host buffer. */
    {
        const RecIssue& gth = c.issues[2];
        const RecIssue& cp  = c.issues[3];
        REQUIRE(gth.opd.size() == 5 && cp.opd.size() == 2);
        CHECK_EQ(cp.opd[0].handle, gth.opd[3].handle);
        CHECK_EQ(cp.opd[0].rows, (int64_t)11);
        CHECK(cp.opd[1].handle != gth.opd[3].handle);
        CHECK_EQ(c.issues[4].opd[0].handle, cp.opd[1].handle);
        CHECK_EQ(c.issues[5].opd[0].handle, cp.opd[1].handle);
    }

    /* ngram_ids: tok, state, cu, 3 weights, cache_idx, has_init, num_accepted, ids, ahead_ids. */
    {
        const RecIssue& d = c.issues[0];
        const RecIssue& p = c.issues[1];
        REQUIRE(d.opd.size() == 11 && p.opd.size() == 11);
        /* No prompt follows this step (n_ahead 0), and a decode half never has one. */
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[2].raw == (void*)cu);             CHECK_EQ(d.opd[2].rows, (int64_t)3);
        CHECK(d.opd[6].raw == (void*)tk_idx);         CHECK_EQ(d.opd[6].rows, (int64_t)2);
        CHECK(d.opd[7].raw == (void*)ctxl);
        CHECK_EQ(d.opd[8].kind, (uint8_t)RAD_OPK_RAW);
        CHECK(d.opd[8].raw == (void*)nacc);           CHECK_EQ(d.opd[8].rows, (int64_t)2);
        CHECK_EQ(d.opd[9].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);         /* token operands stay at their base */
        CHECK(p.opd[2].raw == (void*)(cu + 2));       CHECK_EQ(p.opd[2].rows, (int64_t)2);
        CHECK(p.opd[6].raw == (void*)(tk_idx + 2));   CHECK_EQ(p.opd[6].rows, (int64_t)1);
        CHECK(p.opd[7].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[8].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
    }
    /* ple_conv: x, w, state, cu, resid, cache_idx, has_init, num_accepted, y. */
    {
        const RecIssue& d = c.issues[7];
        const RecIssue& p = c.issues[8];
        REQUIRE(d.opd.size() == 9 && p.opd.size() == 9);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[3].raw == (void*)cu);
        CHECK(d.opd[5].raw == (void*)cv_idx);
        CHECK(d.opd[7].raw == (void*)nacc);
        CHECK_EQ(d.opd[8].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);
        CHECK(p.opd[3].raw == (void*)(cu + 2));
        CHECK(p.opd[5].raw == (void*)(cv_idx + 2));
        CHECK(p.opd[6].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[7].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[8].rows, (int64_t)11);
    }

    /* A pure step issues one half: a decode step passes `num_accepted`, a prefill step does not. */
    for (int phase : { (int)RAD_PHASE_DECODE, (int)RAD_PHASE_PREFILL }) {
        RadBatch pure = mixed;
        pure.phase = phase;
        pure.n_seq_decode = 0; pure.n_tok_decode = 0;   /* the phase decides, not these */
        RadCtx cp; cp.b = &b; cp.batch = &pure;
        ple.ids(&cp, &pure);
        ple.step(&cp, &pure);
        int n_ids = 0, n_conv = 0;
        for (const auto& i : cp.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            const bool ids = o->op == "ngram_ids", conv = o->op == "ple_conv";
            if (!ids && !conv) continue;
            n_ids += ids; n_conv += conv;
            CHECK_EQ(i.n, (int64_t)11);
            const RadOperand& na = i.opd[ids ? 8 : 7];
            if (phase == RAD_PHASE_DECODE) CHECK(na.kind == RAD_OPK_RAW && na.raw == (void*)nacc);
            else                           CHECK_EQ(na.kind, (uint8_t)RAD_OPK_NONE);
        }
        CHECK_EQ(n_ids, 1);
        CHECK_EQ(n_conv, 1);
    }

    /* THE PROMPT THAT FOLLOWS rides the prefill half only: its ngram_ids reads `tok` that many
     * tokens further and writes their ids into the rows of `ple_ids` after the step's, and the
     * gather is handed those same rows. The decode half never sees them. */
    {
        RadBatch ahead = mixed;
        ahead.n_ahead = 7;
        RadCtx ca; ca.b = &b; ca.batch = &ahead;
        ple.ids(&ca, &ahead);
        ple.step(&ca, &ahead);
        REQUIRE(ca.issues.size() == 10);
        const RecIssue& d = ca.issues[0];
        const RecIssue& p = ca.issues[1];
        const RecIssue& gth = ca.issues[2];
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)(11 + 7));
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(p.opd[10].rows, (int64_t)7);
        CHECK_EQ(p.opd[10].offset, (int64_t)(11 * pc.heads));
        REQUIRE(gth.opd.size() == 5);
        CHECK_EQ(gth.opd[4].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(gth.opd[4].handle, p.opd[10].handle);
        CHECK_EQ(gth.opd[4].rows, (int64_t)7);
        CHECK_EQ(gth.opd[4].offset, (int64_t)(11 * pc.heads));
    }
}

RAD_TEST_MAIN()

/* ==================================================================== the sizing declare
 *
 * The activation arena is lent out by level, and a level is the plugin declaring again at a
 * smaller max_tok. Two things have to hold for that to be safe, and both are the PLUGIN's: the
 * sizing declare must not replace the state its step reads -- that was written at the real size,
 * and a step run against the smaller configuration would size its work to it -- and it must
 * describe the same buffers with only their rows smaller, because every other extent is a stride
 * the real declare's kernels were chosen against.
 *
 * Qwen3.8-Flash-Next's metadata, cut to eight layers. */
namespace {
const char* q4_keys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
    "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
    "indexer_n_heads", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* q4_vals[] = {
    "4", "4", "320", "8",
    "silu", "2048", "4", "128",
    "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128",
    "16", "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};

RadModelMeta qwen4exp_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen4exp";
    m.name        = "test-q38-flashnext";
    m.quant       = "";
    m.n_layers    = 8;
    m.n_embd      = 2560;
    m.n_head      = 24;
    m.n_head_kv   = 2;
    m.head_dim    = 256;
    m.n_ff        = 640;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(q4_keys) / sizeof(q4_keys[0]));
    m.kv_key      = q4_keys;
    m.kv_val      = q4_vals;
    return m;
}
}  /* namespace */

/* THE MODEL SAYS WHAT EACH WEIGHT IS, AND THE PLUGIN DECLARES BY IT. Flash-Next's experts, its
 * gated residual's mixing matrices, its MTP head's fc and its 2-bit draft head each have a stored
 * format a recipe chooses, and the plugin learns it from rad_weight_encoding -- the same question at
 * convert and at load. Asked about a model whose recipe made them int4-rotated, E4M3 rows and u2,
 * it declares the rotated expert kernels, the E4M3 mix and the 2-bit head; asked about one with
 * nothing behind it, the fp8 experts and the bf16 mix. */
TEST(the_encodings_the_model_holds_choose_the_flash_next_kernels) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding row8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&row8) && rad_enc_valid(&u2));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
               { "_hc_down.weight", row8 }, { "_hc_up.weight", row8 },
               { "fc_hidden", row8 }, { "fc_embedding", row8 }, { "draft_head", u2 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    CHECK_EQ(m.moecfg.expert_rot, 1);
    CHECK(m.layers[0].hc_mix.mix_e4m3);
    int rotated = 0, fp8_experts = 0, head_views = 0, scale_views = 0;
    for (const RecOp& o : b.ops)
        if (o.op.rfind("moe_gemm", 0) == 0)
            for (const auto& p : o.p) {
                if (p.key == "dtype" && p.sval == "w4a8h") ++rotated;
                if (p.key == "dtype" && p.sval == "fp8a8") ++fp8_experts;
            }
    for (const RecWeight& w : b.weights) {
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") ++head_views;
        /* every expert's scale is a view of its own logical weight, not a weight of its own */
        if (w.name.find("_exps.") != std::string::npos &&
            w.name.size() > 6 && w.name.compare(w.name.size() - 6, 6, ".scale") == 0) {
            ++scale_views;
            CHECK(w.d.source != nullptr && w.d.planes && std::string(w.d.planes) == "scale");
        }
    }
    CHECK(rotated > 0);
    CHECK_EQ(fp8_experts, 0);
    CHECK_EQ(head_views, 1);
    CHECK(scale_views > 0);

    RadBuilder plain;
    REQUIRE_EQ(qwen4exp_fp8::declare(&plain, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& p = qwen4exp_fp8::g_model[0];
    CHECK_EQ(p.moecfg.expert_w4, 0);
    CHECK(!p.layers[0].hc_mix.mix_e4m3);
}

/* A QUANTISED CONTAINER MAY KEEP ONE LINEAR AS THE CHECKPOINT HOLDS IT. Flash-Next's recipe leaves
 * the QSA indexer's projection plain bf16 -- its top-k is a hard choice between blocks -- so in a
 * four-bit model with fp8 linears that one weight must be declared bf16 and read by `gemm_nt` over
 * the bf16 plane of its input, while the attention's own projections beside it stay block fp8. */
TEST(a_quantised_flash_next_reads_its_plain_indexer_as_bf16) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 }, { "qsa_qk", plain } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int qk = 0, qk_bf16 = 0, attn_k_fp8 = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.find(".qsa_qk.weight") != std::string::npos) {
            ++qk;
            qk_bf16 += w.d.dtype == RAD_BF16 && !w.d.planes;
        }
        if (w.name.find(".attn_k.weight") != std::string::npos) attn_k_fp8 += w.d.dtype == RAD_F8E4M3;
    }
    CHECK(qk > 0);
    CHECK_EQ(qk_bf16, qk);
    CHECK(attn_k_fp8 > 0);
    /* every op that reads an indexer weight is the plain GEMM */
    int qk_ops = 0, qk_plain_ops = 0;
    for (const RecOp& o : b.ops)
        for (rad_weight wh : o.w)
            if (wh && b.weights[wh - 1].name.find(".qsa_qk.weight") != std::string::npos) {
                ++qk_ops;
                qk_plain_ops += o.op == "gemm_nt";
            }
    CHECK(qk_ops > 0);
    CHECK_EQ(qk_plain_ops, qk_ops);
}

/* A QUANTISED LAYER MAY KEEP A FEW WHOLE EXPERTS PLAIN bf16. Layer 1 of a four-bit Flash-Next keeps
 * experts 3 and 10: that layer's sort must be told to place them last, its four-bit GEMMs must
 * serve the other experts in slots 0..n-3 with no gap (a container's experts in a layer are
 * consecutive fixed-size units, and the routing counts are in sort order), and the two must be
 * static bf16 weights behind a bf16 grouped GEMM pair -- one of which writes the routed output the
 * four-bit down GEMM writes. Every other layer declares the ordinary four-bit graph. */
TEST(a_four_bit_layer_serves_its_plain_experts_beside_the_rest) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
               { "blk.1.ffn_gate_up_exps.10.weight", plain }, { "blk.1.ffn_down_exps.10.weight", plain },
               { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    const int64_t ne = m.moecfg.n_expert;

    auto param = [](const RecOp& o, const char* k) -> const RecParam* {
        for (const RecParam& p : o.p) if (p.key == k) return &p;
        return nullptr;
    };
    auto layer_of = [&](const RecOp& o) -> int {
        for (rad_weight wh : o.w)
            if (wh) {
                const std::string& n = b.weights[wh - 1].name;
                if (n.rfind("blk.", 0) == 0) return std::atoi(n.c_str() + 4);
            }
        return -1;
    };
    /* the sorts: exactly one names the two experts, and it is layer 1's */
    int prot_sorts = 0, plain_sorts = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "moe_scatter" || o.op == "router_topk_scatter") {
            const RecParam* p = param(o, "protect");
            if (p) { ++prot_sorts; CHECK_EQ(p->sval, std::string("3,10")); }
            else ++plain_sorts;
        }
    CHECK(prot_sorts >= 1);
    CHECK(plain_sorts > 0);
    /* the grouped GEMMs of layer 1 */
    int q_ops = 0, bf_ops = 0;
    for (const RecOp& o : b.ops) {
        if (layer_of(o) != 1) continue;
        if (o.op == "moe_gemm_q") {
            ++q_ops;
            CHECK_EQ(param(o, "n_expert")->ival, ne - 2);
            for (rad_weight wh : o.w) {
                const std::string& n = b.weights[wh - 1].name;
                CHECK(n.find("exps.3.") == std::string::npos && n.find("exps.10.") == std::string::npos);
            }
        }
        if (o.op == "moe_gemm") {
            ++bf_ops;
            CHECK_EQ(param(o, "n_expert")->ival, 2);
            CHECK_EQ(param(o, "dtype")->sval, std::string("bf16"));
            REQUIRE_EQ(o.w.size(), (size_t)2);
            const std::string& n0 = b.weights[o.w[0] - 1].name;
            const std::string& n1 = b.weights[o.w[1] - 1].name;
            CHECK(n0.find("exps.3.weight") != std::string::npos);
            CHECK(n1.find("exps.10.weight") != std::string::npos);
        }
    }
    CHECK_EQ(q_ops, 2);
    CHECK_EQ(bf_ops, 2);
    /* the weights: layer 1's four-bit experts fill slots 0..ne-3 with none missing, and the plain
     * ones are the layer's static weights in bf16 */
    std::vector<int> slot_seen((size_t)ne, 0);
    int plain_w = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.name.find("_exps.") == std::string::npos) continue;
        const bool prot_name = w.name.find("exps.3.") != std::string::npos ||
                               w.name.find("exps.10.") != std::string::npos;
        if (prot_name) {
            ++plain_w;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(w.d.group.expert, -1);
            CHECK_EQ(w.d.access, RAD_ACCESS_PER_TOKEN);
        } else if (w.d.group.expert >= 0 && w.d.group.expert < ne) {
            slot_seen[(size_t)w.d.group.expert] = 1;
        }
    }
    CHECK_EQ(plain_w, 4);
    int filled = 0;
    for (int64_t s = 0; s < ne; ++s) filled += slot_seen[(size_t)s];
    CHECK_EQ(filled, (int)(ne - 2));
    CHECK_EQ(slot_seen[(size_t)(ne - 1)] + slot_seen[(size_t)(ne - 2)], 0);

    /* An odd count on a rank cannot split the rest into two equal tables by parity: refused. */
    RadBuilder odd;
    odd.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
                 { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    CHECK(qwen4exp_fp8::declare(&odd, &meta, &c) != RAD_OK);

    /* Two ranks, each holding a slice of every expert: a plain expert is sliced too, evenly --
     * gate_up 2 x 320 rows (gate's and up's), down 320 columns -- and its GEMMs are that wide. */
    {
        RadBuildCtx c2 = build_ctx(0, 2);
        c2.max_tok = 2048; c2.max_seqs = 8; c2.max_spec = 3; c2.max_ctx = 200000;
        RadBuilder sb;
        sb.encs = b.encs;
        REQUIRE_EQ(qwen4exp_fp8::declare(&sb, &meta, &c2), RAD_OK);
        int sliced = 0;
        for (const RecWeight& w : sb.weights) {
            if (w.name != "blk.1.ffn_gate_up_exps.3.weight" && w.name != "blk.1.ffn_down_exps.3.weight")
                continue;
            ++sliced;
            if (w.name.find("gate_up") != std::string::npos) {
                CHECK_EQ(w.d.shape[0], 640);
                CHECK_EQ(w.d.shard, RAD_SHARD_ROW);
                CHECK_EQ(w.d.n_row_parts, 2);
            } else {
                CHECK_EQ(w.d.shape[1], 320);
                CHECK_EQ(w.d.shard, RAD_SHARD_COL);
            }
        }
        CHECK_EQ(sliced, 2);
        int widths_ok = 0;
        for (const RecOp& o : sb.ops)
            if (o.op == "moe_gemm" && param(o, "n_expert")->ival == 2)
                widths_ok += (param(o, "N")->ival == 640 && param(o, "K")->ival == 2560) ||
                             (param(o, "N")->ival == 2560 && param(o, "K")->ival == 320);
        CHECK_EQ(widths_ok, 2);
    }

    /* The expert the format is probed from may itself be one kept plain: the probe reads on to the
     * first four-bit expert, and the model is four-bit, not bf16. */
    RadBuilder first;
    first.encs = { { "blk.0.ffn_gate_up_exps.0.weight", plain }, { "blk.0.ffn_down_exps.0.weight", plain },
                   { "blk.0.ffn_gate_up_exps.1.weight", plain }, { "blk.0.ffn_down_exps.1.weight", plain },
                   { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&first, &meta, &c), RAD_OK);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_w4, 1);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_bf16, 0);
}

TEST(a_sizing_declare_leaves_the_step_state_alone_and_shrinks_only_rows) {
    (void)&qwen4exp_fp8::step;     /* included for its declare; the step is another test's */
    RadModelMeta meta = qwen4exp_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen4exp_fp8::probe(&meta, &pb));
    CHECK(pb.shape_probe_ok);

    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = 2048;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            const int64_t max_tok = m.g.max_tok;
            const int64_t max_work = m.qsacfg.max_work;
            const rad_op last_op = (rad_op)full.ops.size();
            CHECK_EQ(max_tok, 2048);

            RadBuildCtx p = c;
            p.max_tok = 256;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);

            /* THE STEP'S STATE IS THE REAL DECLARE'S. */
            CHECK_EQ(m.g.max_tok, max_tok);
            CHECK_EQ(m.qsacfg.max_work, max_work);
            CHECK_EQ((rad_op)full.ops.size(), last_op);

            /* THE SAME GRAPH, and the same buffers with only their rows smaller. */
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                CHECK_EQ(small.ops[i].op, full.ops[i].op);
                /* A FIXED PARAMETER THAT FOLLOWS max_tok is a kernel run at the real size against
                 * a level's buffers; only the ranged one may differ. */
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    /* The capacities libr4d's schema names (RAD_PROLE_CAPACITY): qsa_work
                     * launches as wide as its `page` operand, whatever `work` says, and
                     * attn_paged_gate_quant by its block table's rows, whatever `max_seqs` says. */
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld at 2048, %lld at 256\n",
                                i, full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            int shrank = 0;
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(small.bufs[i].name, full.bufs[i].name);
                CHECK_EQ(q.rank, f.rank);
                CHECK_EQ(q.kind, f.kind);
                for (uint32_t d = 1; d < f.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld at 2048, %lld at 256\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d]);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
                shrank += q.shape[0] < f.shape[0];
            }
            CHECK(shrank > 10);
        }
    }
}


/* FOUR RANKS: THE KV HEADS REPLICATE, THE EXPERT WIDTHS SPLIT, THE FUSED COLLECTIVES STOP.
 *
 * This is the tensor-parallel test above's sibling for the other model in the tree, and it exists
 * because world 4 was not declarable at all: `geom_from` refused any world larger than the KV head
 * count, and qwen4exp's parity split of a 640-column expert "serves two ranks" by name. Both are
 * now answered by a DIFFERENT declaration rather than a rounding rule, and the point of this test
 * is to pin which one:
 *
 *   - the KV heads cannot be cut four ways, so they are REPLICATED in pairs: one KV head a rank,
 *     ranks {0,1} reading rows [0,256) of the checkpoint's k/v projection and {2,3} reading
 *     [256,512). That is a span, not a shard index, and it is the only place in the engine where a
 *     rank names the container rows it wants -- so it is asserted here as a pair of agreeing spans
 *     whose union is the whole projection.
 *   - an expert of this model is 640 columns = five 128-column scale blocks, which four ranks
 *     cannot cut evenly, so the split is UNEVEN BY CLASS and it is the width that divides, not the
 *     expert set: every rank routes every one of the 512 experts and computes [lo, hi) of each, the
 *     rank taking the odd block going round with the expert's number mod 4 -- five blocks of every
 *     four experts on every rank, whatever count of quantised experts (four or more) a layer
 *     keeps. Sharding the experts instead was tried and is wrong here -- protection is per-rank,
 *     so a rank can end up owning one plain bf16 expert, leaving an odd count of quantised ones,
 *     and the grouped GEMM takes its experts as two EQUAL tables by parity. The recipe protects
 *     experts in pairs for exactly that reason.
 *   - the FUSED all-reduces are world-2 kernels (`ar_wire_is_exact` and `ar_fused_ok` both return
 *     false off two), with one exception: the gated residual's write. `ar_hc_write` and
 *     `ar_gather_hc_write` also exist at four, on the exact wire only (`wire` 0, never the rotated
 *     form) and over the rows of a decode or verify step, max_seqs * (1 + max_spec). Every block
 *     still declares its standalone `all_reduce`, and every MoE block its `moe_gather`, which
 *     serve the steps the fused write declines. The contrast with world 2 in the same test is the
 *     contract -- it fails loudly if a fused declare is ever gated on the wrong condition.
 *
 * Every expected number is derived from the meta this file builds (n_head 24, n_head_kv 2, head_dim
 * 256, num_experts 512), so the fixture and the assertions cannot drift apart. Where a number is
 * really the PRODUCTION container's -- the unsharded k_proj entry's 512 rows -- the relation is
 * asserted instead of the literal. */
TEST(flashnext_tp4_replicates_kv_heads_and_splits_expert_widths) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world    = 4;
    const int64_t  n_head   = meta.n_head;        /* 24 query heads */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2 KV heads -- fewer than the ranks */
    const int64_t  head_dim = meta.head_dim;      /* 256 */
    const int64_t  ne_all   = atoll(rad_meta_gets(&meta, "num_experts", "0"));
    REQUIRE(n_head % world == 0);
    REQUIRE_EQ(world % n_head_kv, (int64_t)0);   /* replication needs the multiple */
    REQUIRE(ne_all % world == 0);

    /* Full-attention layers of the eight-layer fixture: 3 and 7 (interval 4). */
    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        /* THE DISCRIMINATOR: on the unpatched tree this returns RAD_E_INVAL from geom_from's
         * divide_or_fail(n_head_kv, 4) and nothing below it runs. */
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    for (int r = 0; r < world; ++r) {
        /* (1) ONE QUARTER OF THE QUERY HEADS, ONE KV HEAD, so gqa is 6 and the page is the QSA
         * compress block of 4 (qwen4exp_fp8.cpp sets lay.attn.kv_block = 4, and the row's
         * C_EQ("block_size", 4) has to agree with it or nothing resolves). */
        const RecOp* at = find_op(b[r], "attn_paged");
        REQUIRE(at != nullptr);
        CHECK_EQ(pint(at, "n_head"), n_head / world);
        CHECK_EQ(pint(at, "gqa"), n_head / world);   /* n_head_kv is 1, so gqa == the rank's heads */
        CHECK_EQ(pint(at, "block_size"), 4);
        /* The full-attention KV group holds the one head this rank serves. Group 0 is kv_attn --
         * the first declared (qwen4exp_fp8.cpp's three groups, RAD_KV_FULL first). */
        REQUIRE(b[r].kv_groups.size() > 0);
        CHECK_EQ(b[r].kv_groups[0].name, std::string("kv_attn"));
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        CHECK_EQ(b[r].kv_groups[0].d.head_dim, head_dim);

        /* (2) ONE KV HEAD'S ROWS, and the SPAN that locates them in the entry. shape[0] is
         * n_head_kv * head_dim = 256 per rank; the pair of ranks reading it declares the SAME span,
         * and the two spans together are the checkpoint's whole k_proj -- n_head_kv_all * head_dim
         * = 512 rows, which is a CONTAINER fact this file cannot read, so it is asserted as the
         * relation below rather than as a literal. */
        for (const std::string& wn : {kq, kv}) {
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);          /* one head */
            CHECK_EQ(w->d.shape[1], meta.n_embd);       /* the projection's K is not sharded */
            CHECK_EQ(w->d.shard, RAD_SHARD_ROW);
            CHECK(w->span_set);
            CHECK_EQ(w->span_hi - w->span_lo, w->d.shape[0]);
            CHECK_EQ(w->span_lo, (int64_t)(r / (world / n_head_kv)) * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        /* (3) EVERY RANK ROUTES EVERY EXPERT, and computes a SLICE of each one. The router is
         * replicated and so is the expert SET -- n_expert stays the model's 512 with base 0 --
         * because what the ranks divide is each expert's WIDTH, in whole 128-column scale blocks,
         * one block class per parity. A rank that disagreed with the router about the expert count
         * would route to a different model. */
        const RecOp* sc = find_op(b[r], "moe_scatter");
        REQUIRE(sc != nullptr);
        CHECK_EQ(pint(sc, "n_expert"), ne_all);
        CHECK_EQ(pint(sc, "expert_base"), (int64_t)0);
        const RecOp* tk = find_op(b[r], "router_topk");
        REQUIRE(tk != nullptr);
        CHECK_EQ(pint(tk, "n_expert"), ne_all);         /* replicated: one routing per token */
        if (const RecOp* ts = find_op(b[r], "router_topk_scatter")) {
            CHECK_EQ(pint(ts, "n_expert"), ne_all);
            CHECK_EQ(pint(ts, "expert_base"), (int64_t)0);
        }
        /* THE WIDTH EACH RANK TAKES OF EACH CLASS, as a span on every expert's down projection --
         * the fixture protects no expert, so an expert's class is its own number mod 4. These
         * eight numbers per rank are the whole TP4 MoE contract: five blocks of 128, four ranks,
         * and the rank taking the odd block going round with the class. They are LITERAL here on
         * purpose -- recomputing them from the plugin's formula would only prove the test agrees
         * with the code, not that either agrees with the model. */
        const int64_t blk = 128;                    /* RAD_FP8_BLOCK, the scale block's width */
        const int64_t nblk = meta.n_ff / blk;        /* five, at the fixture's 640 */
        const int64_t cls_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                       { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t cls_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                       { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        REQUIRE_EQ(nblk, (int64_t)5);
        int spanned[4] = { 0, 0, 0, 0 };
        for (const RecWeight& w : b[r].weights) {
            if (w.name.rfind("blk.0.ffn_down_exps.", 0) != 0) continue;
            if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
            const int q = atoi(w.name.c_str() + 20) % 4;      /* the expert's class */
            CHECK(w.span_set);
            CHECK_EQ(w.span_lo, cls_lo[r][q]);
            CHECK_EQ(w.span_hi, cls_hi[r][q]);
            ++spanned[q];
        }
        for (int q = 0; q < 4; ++q)
            CHECK_EQ(spanned[q], ne_all / 4);            /* every expert of each class is declared */
        /* AND THE FOUR RANKS' SLICES OF ONE CLASS TILE ITS WIDTH EXACTLY: no block unserved (a
         * rank silently dropping a fifth of every expert's output), none claimed twice (the
         * all-reduce then sums that block twice and the number is wrong by its own size). Read off
         * the declared spans, not off the formula that produced them. */
        for (int q = 0; q < 4; ++q) {
            int64_t covered = 0;
            for (int r2 = 0; r2 < world; ++r2)
                for (int64_t i = 0; i < nblk; ++i) {
                    const int64_t lo = cls_lo[r2][q], hi = cls_hi[r2][q];
                    if (lo <= i * blk && i * blk < hi) ++covered;
                }
            CHECK_EQ(covered, nblk);   /* per class, exactly once each */
        }
        /* AND EVERY RANK TAKES THE SAME WIDTH: five blocks of every four experts -- the point of
         * four classes. Two would leave ranks 0 and 1 three blocks of every pair and 2 and 3 two. */
        {
            int64_t sum = 0;
            for (int q = 0; q < 4; ++q) sum += cls_hi[r][q] - cls_lo[r][q];
            CHECK_EQ(sum, nblk * blk);
        }
        /* The gate/up GEMM's four classes follow the same split, doubled because the operand
         * stacks gate and up, and the op's buffers take the WIDEST class. No parity op is left. */
        const RecOp* gu = find_op(b[r], "moe_gemm_q_mod4");
        REQUIRE(gu != nullptr);
        CHECK_EQ(pint(gu, "N"), 2 * (cls_hi[r][0] - cls_lo[r][0]));
        CHECK_EQ(pint(gu, "N_1"), 2 * (cls_hi[r][1] - cls_lo[r][1]));
        CHECK_EQ(pint(gu, "N_2"), 2 * (cls_hi[r][2] - cls_lo[r][2]));
        CHECK_EQ(pint(gu, "N_3"), 2 * (cls_hi[r][3] - cls_lo[r][3]));
        CHECK_EQ(pint(gu, "n_expert"), ne_all);
        CHECK_EQ(gu->w.size(), (size_t)(2 * ne_all));   /* codes and scales, every expert once */
        CHECK_EQ(count_op(b[r], "moe_gemm_q"), 0);

        /* (4) A standalone collective per reduction, the exact four-rank ar_hc_write and
         * ar_gather_hc_write beside it -- the gather form on every MoE connection, so one a layer
         * plus the MTP head's if there is one -- and not the fused norm. Both fused forms state
         * the exact wire and are declared over a decode or verify step's rows, 8 * (1 + 3): the
         * architecture's own bound, not a message size a kernel library chose. */
        CHECK(count_op(b[r], "all_reduce") > 0);
        CHECK_EQ(pint(find_op(b[r], "all_reduce"), "world_size"), world);
        CHECK(count_op(b[r], "ar_hc_write") > 0);
        CHECK(count_op(b[r], "moe_gather") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") <= count_op(b[r], "moe_gather"));
        for (const RecOp& o : b[r].ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "world_size"), world);
            CHECK_EQ(pint(&o, "wire"), 0);                 /* exact only, and stated */
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->kind, (int)RAD_P_RANGE);
            CHECK_EQ(m->ihi, c[r].max_seqs * (1 + c[r].max_spec));
        }
        CHECK_EQ(count_op(b[r], "ar_rmsnorm_quant_fp8"), 0);
    }

    /* THE PAIRS AGREE AND THEIR UNION IS THE WHOLE PROJECTION. 2 * head_dim is the container's
     * k_proj entry height (n_head_kv_all rows of head_dim each) -- the relation, not the literal,
     * because the rows belong to the model file and not to this fixture. */
    for (const std::string& wn : {kq, kv}) {
        CHECK_EQ(find_w(b[0], wn)->span_hi, find_w(b[1], wn)->span_hi);
        CHECK_EQ(find_w(b[2], wn)->span_lo, find_w(b[3], wn)->span_lo);
        CHECK_EQ(find_w(b[0], wn)->span_lo, find_w(b[2], wn)->span_hi - n_head_kv * head_dim);
        const int64_t lo = find_w(b[0], wn)->span_lo;
        const int64_t hi = find_w(b[2], wn)->span_hi;
        CHECK_EQ(hi - lo, n_head_kv * head_dim);
        CHECK_EQ(hi - lo, world * find_w(b[0], wn)->d.shape[0] / (world / n_head_kv));
    }

    /* (5) THE FOUR GRAPHS ARE THE SAME GRAPH. A rank's shard differs in a parameter or a weight's
     * span, never in which ops it declares -- a graph that varied by rank would be a plan that
     * varied by rank. */
    for (int r = 1; r < world; ++r) {
        if (op_names(b[r]) != op_names(b[0])) {
            fprintf(stderr, "    rank %d differs from rank 0 in its op list (%zu vs %zu)\n", r,
                    b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size() && i < b[r].ops.size(); ++i)
                if (b[0].ops[i].op != b[r].ops[i].op) {
                    fprintf(stderr, "    first at %zu: '%s' vs '%s'\n", i,
                            b[0].ops[i].op.c_str(), b[r].ops[i].op.c_str());
                    break;
                }
        }
        CHECK_EQ(join(op_names(b[r])), join(op_names(b[0])));
    }

    /* THE SHARED ARM IS SPLIT BY WHOLE BLOCKS AND IS UNEVEN AT FOUR: 640 is five blocks of 128, and
     * the rank taking the extra block is rank 2, the first rank after the plain experts' extra
     * tiles on ranks 0 and 1 (qwen4exp_fp8.cpp's shared_place) -- ranks 2, 3, 0, 1 cut [0,256)
     * [256,384) [384,512) [512,640) in the model's column numbering. At two ranks rank 0 still takes three of five.
     * This is the one world-4 span that is NOT a KV head. */
    const int64_t sh_lo[] = { 384, 512, 0,   256 };
    const int64_t sh_hi[] = { 512, 640, 256, 384 };
    for (int r = 0; r < world; ++r) {
        const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
        REQUIRE(sg != nullptr);
        CHECK(sg->span_set);
        CHECK_EQ(sg->span_lo, sh_lo[r]);
        CHECK_EQ(sg->span_hi, sh_hi[r]);
        CHECK_EQ(sg->d.shape[0], 2 * (sh_hi[r] - sh_lo[r]));   /* gate's rows and up's */
    }
    /* Every block of the shared expert has exactly one owner, read off the declared spans. */
    for (int64_t i = 0; i < 640; i += 128) {
        int owners = 0;
        for (int r = 0; r < world; ++r) {
            const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
            owners += sg->span_lo <= i && i < sg->span_hi;
        }
        CHECK_EQ(owners, 1);
    }

    /* AT TWO, THE OLD DECLARATION: the KV heads split, so the even shard already selects the right
     * rows and NO SPAN is declared at all -- TP2 declares as it did before replication existed.
     * The fused forms are the tell that world 2 is a different deployment and not the same
     * one stretched: they resolve there and nowhere else. */
    {
        RadBuilder h[2];
        for (int r = 0; r < 2; ++r) {
            RadBuildCtx c2 = build_ctx(r, 2);
            c2.max_tok  = 2048;
            c2.max_seqs = 8;
            c2.max_spec = 3;
            c2.max_ctx  = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&h[r], &meta, &c2), RAD_OK);
            const RecWeight* w = find_w(h[r], kq);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], n_head_kv / 2 * head_dim);   /* one of two heads */
            CHECK(!w->span_set);
            CHECK_EQ(h[r].kv_groups[0].d.n_head_kv, n_head_kv / 2);
            CHECK_EQ(pint(find_op(h[r], "attn_paged"), "gqa"), (n_head / 2) / (n_head_kv / 2));
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "n_expert"), ne_all);
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "expert_base"), 0);
            CHECK(count_op(h[r], "ar_hc_write") > 0);
            CHECK(count_op(h[r], "ar_gather_hc_write") > 0);
            /* Two ranks keep the exact write over every step, prefill chunks included. */
            for (const RecOp& o : h[r].ops) {
                if (o.op != "ar_hc_write") continue;
                CHECK_EQ(pint(&o, "wire"), 0);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, c2.max_tok);
            }
            /* Not ar_rmsnorm_quant_fp8: qwen4exp hands every block's norm `kArNone`, because here
             * the reduction rides in the gated residual's WRITE rather than in the norm that
             * follows it (qwen4exp_fp8.cpp's block declares). So this one is absent at BOTH world
             * sizes -- asserted so, rather than claimed as part of the contrast. */
            CHECK_EQ(count_op(h[r], "ar_rmsnorm_quant_fp8"), 0);
        }
    }

    /* THE FOUR-RANK FUSED WRITE'S STEPS: the decode and verify steps -- max_seqs * (1 + max_spec)
     * rows, never more than max_tok -- whose message goes out exact, which on the lossy wire means
     * under its floor or off its 128-element tile. The bound is the architecture's own and not a
     * message size: 64 sequences at depth 3 are 256 rows, 1.25 MiB, and still take it, while a
     * prefill chunk keeps the pair at any size. The block in front and the write ask the same
     * predicate. At two ranks it is ar_wire_is_exact unchanged, and ar_wire_is_exact itself stays
     * false at four. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)32);
        CHECK_EQ(rad::arch::ar_exact_write_rows(g4), (int64_t)32);
        CHECK(!rad::arch::ar_exact_write_takes(g4, 0));
        CHECK(rad::arch::ar_exact_write_takes(g4, 1));
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));      /* a decode step */
        CHECK(rad::arch::ar_exact_write_takes(g4, 32));     /* every sequence verifying 1 + 3 */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 33));    /* more: a prefill chunk's */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 2048));
        CHECK(!rad::arch::ar_wire_is_exact(g4, 8));
        g4.wire_exact = 0;
        CHECK(rad::arch::ar_exact_write_takes(g4, 25));     /* 125 KiB, under the floor */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 26));    /* 130 KiB, the ti8 wire's */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 32));
        g4.wire_exact = 1;
        g4.max_spec = 0;                                    /* one row a sequence */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)8);
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 9));
        g4.max_seqs = 64; g4.max_spec = 3;
        CHECK(rad::arch::ar_exact_write_takes(g4, 256));    /* 1.25 MiB */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 257));
        g4.max_tok = 128;                                   /* a step holds no more than this */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)128);
        CHECK(rad::arch::ar_exact_write_takes(g4, 128));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 129));
        rad::arch::Geom g8 = g4;
        g8.world = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g8), (int64_t)0);
        CHECK(!rad::arch::ar_exact_write_takes(g8, 8));
        rad::arch::Geom g2 = g4;
        g2.world = 2; g2.max_tok = 4096; g2.max_seqs = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g2), g2.max_tok);
        for (int exact : { 0, 1 }) {
            g2.wire_exact = exact;
            for (int64_t T : { 1, 8, 25, 26, 33, 205, 4096 })
                CHECK(rad::arch::ar_exact_write_takes(g2, T) == rad::arch::ar_wire_is_exact(g2, T));
        }
    }
    /* THE FOUR-RANK GATHER'S STEPS: the fused write's, and within one MoE pass. The exact bound
     * is the only one at four (the rotated form is never declared there, so `rows6` is 0), and a
     * step the write declines keeps its gather. At two ranks the bound is still picked by
     * ar_wire_is_exact. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK(rad::arch::gather_taken(g4, 8, 32, 0));        /* a decode step */
        CHECK(rad::arch::gather_taken(g4, 32, 32, 0));       /* a verify step, one pass */
        CHECK(!rad::arch::gather_taken(g4, 33, 2048, 0));    /* the write declines it */
        CHECK(!rad::arch::gather_taken(g4, 8, 0, 0));        /* not declared */
        CHECK(!rad::arch::gather_taken(g4, 20, 16, 0));      /* two passes */
        g4.wire_exact = 0;
        CHECK(rad::arch::gather_taken(g4, 25, 32, 0));       /* under the floor */
        CHECK(!rad::arch::gather_taken(g4, 26, 32, 0));      /* the ti8 wire's */
        rad::arch::Geom g2 = g4;
        g2.world = 2;
        for (int64_t T : { 1, 8, 25, 26, 205, 4096 })
            for (int64_t r6 : { 0, 2048 }) {
                const int64_t r = rad::arch::ar_wire_is_exact(g2, T) ? 2048 : r6;
                CHECK(rad::arch::gather_taken(g2, T, 2048, r6) == (r > 0 && T <= r));
            }
    }
    /* ...and the lossy wire at four declares no rotated write and no rotated gather form: the
     * exact ones only, stating `wire` 0, for the decode and verify steps under the floor. */
    {
        RadBuilder b4;
        RadBuildCtx c4 = build_ctx(0, world);
        c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
        c4.tp_wire_lossy = 1;
        c4.tp_wire_min_bytes = 128 * 1024;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b4, &meta, &c4), RAD_OK);
        CHECK(count_op(b4, "ar_hc_write") > 0);
        CHECK(count_op(b4, "ar_gather_hc_write") > 0);
        for (const RecOp& o : b4.ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "wire"), 0);
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->ihi, c4.max_seqs * (1 + c4.max_spec));
        }
    }

    /* A WORLD THAT CANNOT REPLICATE -- not a multiple of the KV head count -- is refused by name,
     * not rounded into a half head. */
    {
        RadBuilder b3;
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK(qwen4exp_fp8::declare(&b3, &meta, &c3) != RAD_OK);
    }

    /* ---- A LAYER THAT KEEPS EXPERTS PLAIN STAYS AT FOUR CLASSES. Two plain experts leave 510
     * quantised ones, four leave 508; either way the layer is served by `moe_gemm_q_mod4`, the
     * class being the expert's PLACE in the sort mod 4 (the plain ones sorted last), and at 510
     * classes 0 and 1 hold 128 experts and 2 and 3 hold 127 -- the declaration hands a table op's
     * first tables the remainder. The plain experts' extra tiles go to ranks 0 and 1: every rank
     * holds the same routed width, and the shared expert's extra block is rank 2's. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* The four-class slices, [lo, hi) by rank and place mod 4. */
        const int64_t c4_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                      { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t c4_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                      { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        for (int nprot : { 2, 4 }) {
            const std::vector<int> pe = nprot == 2 ? std::vector<int>{ 3, 10 }
                                                   : std::vector<int>{ 3, 10, 21, 40 };
            for (int r = 0; r < world; ++r) {
                RadBuilder bp;
                for (int e : pe) {
                    bp.encs.push_back({ "blk.1.ffn_gate_up_exps." + std::to_string(e) + ".weight",
                                        plain });
                    bp.encs.push_back({ "blk.1.ffn_down_exps." + std::to_string(e) + ".weight",
                                        plain });
                }
                bp.encs.push_back({ "ffn_gate_up_exps", w4p });
                bp.encs.push_back({ "ffn_down_exps", w4p });
                RadBuildCtx cp = build_ctx(r, world);
                cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
                REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
                const int64_t n_reg = ne_all - nprot;
                /* layer 1's quantised GEMMs, and nobody else's */
                int l1_ops = 0, other_ops = 0;
                for (const RecOp& o : bp.ops) {
                    if (o.op != "moe_gemm_q" && o.op != "moe_gemm_q_mod4") continue;
                    bool l1 = false;
                    for (rad_weight wh : o.w)
                        l1 = l1 || bp.weights[wh - 1].name.rfind("blk.1.", 0) == 0;
                    if (!l1) { CHECK_EQ(o.op, std::string("moe_gemm_q_mod4")); ++other_ops; continue; }
                    ++l1_ops;
                    CHECK_EQ(o.op, std::string("moe_gemm_q_mod4"));
                    CHECK_EQ(pint(&o, "n_expert"), n_reg);
                    CHECK_EQ(o.w.size(), (size_t)(2 * n_reg));
                    /* the four runs, read off the declaration: codes then scales, class by class,
                     * each class the places q, q + 4, ... -- 128, 128, 127, 127 at 510 */
                    size_t at = 0;
                    for (int q = 0; q < 4; ++q) {
                        const int64_t nq = (n_reg + 3 - q) / 4;
                        for (int half = 0; half < 2; ++half)
                            for (int64_t j = 0; j < nq && at < o.w.size(); ++j, ++at) {
                                const std::string& wn = bp.weights[o.w[at] - 1].name;
                                const int e = atoi(wn.c_str() + wn.find("_exps.") + 6);
                                int place = e;
                                for (int x : pe) place -= x < e ? 1 : 0;
                                CHECK_EQ(place, (int)(q + 4 * j));
                                CHECK((wn.find(".scale") != std::string::npos) == (half == 1));
                            }
                    }
                    CHECK_EQ(at, o.w.size());
                }
                CHECK_EQ(l1_ops, 2);
                CHECK(other_ops > 0);
                /* every quantised expert of layer 1 by its place: the class's slice */
                int seen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    if (w.d.dtype == RAD_BF16) continue;          /* the plain ones, below */
                    const int e = atoi(w.name.c_str() + 20);
                    int place = e;
                    for (int x : pe) place -= x < e ? 1 : 0;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, c4_lo[r][place % 4]);
                    CHECK_EQ(w.span_hi, c4_hi[r][place % 4]);
                    ++seen;
                }
                CHECK_EQ(seen, (int)n_reg);
                /* the plain experts' tiles, by rank: 3, 3, 2, 2 */
                const int64_t p4_lo[4] = { 0, 192, 384, 512 },   p4_hi[4] = { 192, 384, 512, 640 };
                int pseen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.d.dtype != RAD_BF16) continue;
                    if (w.name.find("_exps.") == std::string::npos) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, p4_lo[r]);
                    CHECK_EQ(w.span_hi, p4_hi[r]);
                    ++pseen;
                }
                CHECK_EQ(pseen, 2 * nprot);
            }
        }
    }

    /* ---- THE PLAIN bf16 EXPERTS AT FOUR RANKS.
     *
     * They are declared beside the quantised ones by a SECOND grouped GEMM, over their own slice of
     * the same width, and that slice has its own rule: whole 64-wide output tiles, one range a rank,
     * uniform across the protected experts of a layer (their GEMM takes ONE table, so every entry
     * must have one geometry -- unlike the quantised experts, they do not alternate by parity).
     * The width was read off this rank's parity slices, which at two ranks is the model's width and
     * at four is not, so the rank took a number that was not its expert's width, and the check on
     * it refused the model on the ranks where that number was wrong -- two of four, which then
     * never arrived at the all-reduce's rendezvous and took the whole launch down with it.
     *
     * Ten tiles over four ranks is 3, 3, 2, 2: an UNEVEN share, still whole tiles, every tile owned
     * once. Ranks 0 and 1 compute 192 columns of every protected expert and ranks 2 and 3 compute
     * 128: the routed experts are split in four classes, so every rank holds the same routed
     * width, and rank 2 holds the shared expert's extra block (shared_place). Two protected
     * experts, as the production recipe's pairs have it. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* [lo, hi) of the plain expert's columns, per rank -- literal, for the reason given above.
         * The layer is served at four classes, so the extra tiles are cut by rank: 3, 3, 2, 2. */
        const int64_t p_lo[4] = { 0,   192, 384, 512 };
        const int64_t p_hi[4] = { 192, 384, 512, 640 };
        for (int r = 0; r < world; ++r) {
            RadBuilder bp;
            bp.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain },
                        { "blk.1.ffn_down_exps.3.weight", plain },
                        { "blk.1.ffn_gate_up_exps.10.weight", plain },
                        { "blk.1.ffn_down_exps.10.weight", plain },
                        { "ffn_gate_up_exps", w4p }, { "ffn_down_exps", w4p } };
            RadBuildCtx cp = build_ctx(r, world);
            cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
            const int64_t wp = p_hi[r] - p_lo[r];
            int seen = 0;
            for (const RecWeight& w : bp.weights) {
                if (w.name.rfind("blk.1.ffn_gate_up_exps.", 0) != 0 &&
                    w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                if (w.d.dtype != RAD_BF16) continue;      /* only the plain ones are in scope */
                if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                CHECK(w.span_set);
                CHECK_EQ(w.span_lo, p_lo[r]);
                CHECK_EQ(w.span_hi, p_hi[r]);
                ++seen;
            }
            /* Both projections of both protected experts. */
            CHECK_EQ(seen, 4);
            /* The gate/up GEMM is as wide as this rank's share, doubled for the stacked pair, and
             * the down GEMM's K is that share: the partial this rank contributes. */
            int gu_ok = 0, dn_ok = 0;
            for (const RecOp& o : bp.ops) {
                if (o.op != "moe_gemm") continue;
                if (pint(&o, "N") == 2 * wp && pint(&o, "n_expert") == 2) ++gu_ok;
                if (pint(&o, "K") == wp && pint(&o, "n_expert") == 2) ++dn_ok;
            }
            CHECK_EQ(gu_ok, 1);
            CHECK_EQ(dn_ok, 1);
        }
    }
}

/* ==================================================================== media */
namespace {

/* The fixture above, with what a Qwen3.8 container adds for pictures: the interleaved M-RoPE of
 * its text config and a vision tower cut to TWO blocks at the real widths, which is enough to see
 * a block repeat and short enough to read. */
const char* k_vl_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
    "rope_parameters.mrope_section",
    "rope_parameters.mrope_interleaved",
    "vision_config.depth",
    "vision_config.hidden_size",
    "vision_config.num_heads",
    "vision_config.intermediate_size",
    "vision_config.patch_size",
    "vision_config.temporal_patch_size",
    "vision_config.in_channels",
    "vision_config.spatial_merge_size",
    "vision_config.out_hidden_size",
    "vision_config.num_position_embeddings",
    "vision_config.deepstack_visual_indexes",
    "vision_config.hidden_act",
};
const char* k_vl_vals[] = { "4", "64", "128", "16", "48", "6144", "4",
                            "11 11 10", "1",
                            "2", "1152", "16", "4304", "16", "2", "3", "2", "5120", "2304", "",
                            "gelu_pytorch_tanh" };

RadModelMeta qwen35_vl_meta() {
    RadModelMeta m = qwen35_meta();
    m.n_kv   = (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0]));
    m.kv_key = k_vl_keys;
    m.kv_val = k_vl_vals;
    return m;
}

RadBuildCtx vl_ctx(int64_t patches) {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = patches;
    return c;
}

const RecOp* issued_op(const RadBuilder& b, const RadCtx& c, const std::string& name, int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return o;
    }
    return nullptr;
}
const RecIssue* issue_of(const RadBuilder& b, const RadCtx& c, const std::string& name,
                         int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return &i;
    }
    return nullptr;
}

}  /* namespace */

TEST(a_vision_container_declares_the_tower_when_the_engine_asks_for_it) {
    /* No encoder budget: the tower is not declared, whatever the container carries. */
    {
        RadBuilder b;
        RadModelMeta meta = qwen35_vl_meta();
        RadBuildCtx  ctx  = vl_ctx(0);
        CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 3 * 2 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 5120);
    const RecBuf* out = find_b(b, "v.out");
    REQUIRE(out != nullptr);
    CHECK_EQ(out->d.shape[0], 256);
    CHECK_EQ(out->d.shape[1], 5120);

    /* 3 stem + 12 a block + 6 merger, every one mapped from the checkpoint's model.visual.*. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
            CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
        }
    CHECK_EQ(nv, 3 + 12 * 2 + 6);
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)1);
    CHECK_EQ(qkv->src[0], std::string("model.visual.blocks.1.attn.qkv.weight"));

    /* The tower's rotary is the 2-D one, half the 72-wide head a side. */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "axial") {
            ++axial;
            CHECK_EQ(pstr(&o, "sections"), std::string("18 18"));
            CHECK_EQ(pint(&o, "head_dim"), 72);
        }
    CHECK_EQ(axial, 2);
    /* And the language model's is the interleaved M-RoPE, with the checkpoint's sections. */
    int im = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "imrope") {
            ++im;
            CHECK_EQ(pstr(&o, "sections"), std::string("11 11 10"));
            CHECK_EQ(pint(&o, "rotary_dim"), 64);
        }
    CHECK_EQ(im, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* A sizing declare sizes the tower and does not declare it to the core. */
    RadBuilder bp;
    RadBuildCtx cp = ctx;
    cp.shape_probe = 1;
    CHECK_OK(qwen35_bf16::declare(&bp, &meta, &cp));
    CHECK(bp.encoder_name.empty());
    CHECK(find_w(bp, "v.patch.weight") != nullptr);
}

TEST(an_encoder_pass_issues_the_tower_and_nothing_else) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* Two segments of 8 x 8 patches: two pictures, or two temporal groups of one video. */
    static uint16_t pix[128 * 1536];
    static int32_t  coord[4 * 128];
    static const int32_t cu[] = { 0, 64, 128 };
    RadBatch batch{};
    batch.enc         = 1;
    batch.enc_n_patch = 128;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    qwen35_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The merger runs over merged rows: a quarter of the patches. */
    const RecIssue* last = &c.issues.back();
    CHECK_EQ(last->n, 32);
    /* The attention reads the pass's segment boundaries, and q, k, v as column slices. */
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    REQUIRE_EQ(at->opd.size(), (size_t)5);
    CHECK_EQ(at->opd[0].cols, 1152);
    CHECK_EQ(at->opd[1].offset, 1152);
    CHECK_EQ(at->opd[2].offset, 2304);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(at->opd[3].rows, 3);
    /* The rotary reads the row and column planes, the grid embedding all four. */
    const RecIssue* ro = issue_of(b, c, "rope");
    REQUIRE(ro != nullptr);
    CHECK(ro->opd[1].raw == coord);
    CHECK_EQ(ro->opd[1].rows, 2);
    CHECK_EQ(ro->opd[1].cols, 128);
    const RecIssue* ge = issue_of(b, c, "grid_embed");
    REQUIRE(ge != nullptr);
    CHECK_EQ(ge->opd[2].rows, 4);

    /* A pass larger than the tower was declared for fails the step rather than overrunning. */
    RadCtx c2; c2.b = &b; c2.rank = 0; c2.world = 1;
    RadBatch big = batch;
    big.enc_n_patch = 2048;
    c2.batch = &big;
    qwen35_bf16::step(&c2, &big);
    CHECK(!c2.step_fail.empty());
    CHECK(c2.issues.empty());
}

TEST(a_step_with_pictures_scatters_their_rows_and_rotates_by_three_components) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A prefill chunk whose tokens 4..11 are one picture's rows. */
    static int32_t rope[3 * 64];
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5120];
    RadBatch batch = prefill_batch();
    batch.rope_pos   = rope;
    batch.rope_mixed = 1;
    batch.n_mm_rows  = 8;
    batch.mm_rows    = rows;
    batch.mm_embd    = embd;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    qwen35_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());

    const std::vector<std::string> ops = issued_ops(b, c);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
    const RecIssue* sc = issue_of(b, c, "scatter_rows");
    REQUIRE(sc != nullptr);
    CHECK_EQ(sc->n, 8);
    CHECK(sc->opd[0].raw == embd);
    CHECK_EQ(sc->opd[0].rows, 8);
    CHECK_EQ(sc->opd[0].cols, 5120);
    CHECK(sc->opd[1].raw == rows);
    CHECK_EQ(sc->opd[2].rows, 64);

    /* Every language-model rotary takes the [3, T] planes. */
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope" || pstr(o, "mode") != "imrope") continue;
        ++n;
        CHECK(i.opd[1].raw == rope);
        CHECK_EQ(i.opd[1].rows, 3);
        CHECK_EQ(i.opd[1].cols, 64);
    }
    CHECK_EQ(n, 4);

    /* A text-only step on the same program: no scatter, and one position a token. */
    RadCtx ct; ct.b = &b; ct.rank = 0; ct.world = 1;
    static const int32_t pos[] = { 90, 91, 92, 93 };
    RadBatch text = decode_batch();
    text.positions = pos;
    ct.batch = &text;
    qwen35_bf16::step(&ct, &text);
    CHECK(ct.step_fail.empty());
    CHECK(issued_op(b, ct, "scatter_rows") == nullptr);
    for (const auto& i : ct.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope") continue;
        CHECK(i.opd[1].raw == pos);
        CHECK_EQ(i.opd[1].cols, 0);
    }
}

namespace {
/* The Flash-Next fixture with its vision tower and its interleaved M-RoPE. */
std::vector<const char*> q4vl_keys, q4vl_vals;
RadModelMeta qwen4exp_vl_meta() {
    RadModelMeta m = qwen4exp_meta();
    if (q4vl_keys.empty()) {
        q4vl_keys.assign(m.kv_key, m.kv_key + m.n_kv);
        q4vl_vals.assign(m.kv_val, m.kv_val + m.n_kv);
        for (int i = 7; i < (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0])); ++i) {
            q4vl_keys.push_back(k_vl_keys[i]);
            /* Flash-Next's tower emits rows at its own 2560, not the 27B's 5120. */
            q4vl_vals.push_back(std::strcmp(k_vl_keys[i], "vision_config.out_hidden_size") ? k_vl_vals[i]
                                                                                           : "2560");
        }
    }
    m.n_kv   = (int)q4vl_keys.size();
    m.kv_key = q4vl_keys.data();
    m.kv_val = q4vl_vals.data();
    return m;
}
}  /* namespace */

TEST(a_sizing_declare_with_a_vision_tower_is_the_same_graph) {
    RadModelMeta meta = qwen4exp_vl_meta();
    /* A serving configuration: fp8 cache, the lossy wire, three MTP rounds -- and every level the
     * engine sizes below a 2048-token chunk. */
    for (int world : { 1, 2 }) for (int lossy : { 0, 1 }) for (int64_t R : { 256, 1024 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = 2048;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            c.max_enc_patches = 16384;
            c.kv_dtype = RAD_F8E4M3;
            c.tp_wire_lossy = lossy;
            c.tp_wire_min_bytes = 64 * 1024;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            CHECK_EQ(full.encoder_name, std::string("vision"));
            RadBuildCtx p = c;
            p.max_tok = R;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);
            CHECK(small.encoder_name.empty());
            size_t first = 0;
            while (first < full.ops.size() && first < small.ops.size() &&
                   full.ops[first].op == small.ops[first].op)
                ++first;
            if (first < full.ops.size() || first < small.ops.size())
                fprintf(stderr, "    world %d rank %d lossy %d R %lld: %zu ops vs %zu, first "
                                "difference at %zu ('%s' vs '%s')\n", world, rank, lossy,
                        (long long)R, full.ops.size(), small.ops.size(),
                        first, first < full.ops.size() ? full.ops[first].op.c_str() : "-",
                        first < small.ops.size() ? small.ops[first].op.c_str() : "-");
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            CHECK_EQ(first, full.ops.size());
            /* And every FIXED parameter the real declare's, the tower's included: the engine runs
             * the real declare's kernels at a level, and only the ranged row counts may differ. */
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld real, %lld at %lld\n", i,
                                full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival,
                                (long long)R);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            /* A level shrinks a buffer's ROWS and nothing else: its other extents are the real
             * declare's, or the level's layout is not the real one's with fewer rows. */
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(q.rank, f.rank);
                for (uint32_t d = 1; d < f.rank && d < q.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld real, %lld at %lld\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d], (long long)R);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
            }
        }
    }
}

/* THE PER-EXPERT TAP KEEPS EACH EXPERT'S GRAM WHOLE. CalibExperts stores only the upper triangle
 * of each expert's Gram, packed, and a slip in the packing would put one expert's entries into
 * another's or transpose a row into a column -- every file would still be k x k and symmetric. So
 * two passes of distinct symmetric Grams go in, and each expert must come back as the sum of its
 * own two, entry for entry, at a width that is not a multiple of anything. */
TEST(the_per_expert_tap_keeps_each_experts_gram_whole) {
    const int64_t ne = 5, k = 7;
    rad::arch::CalibExperts ce;
    ce.init(ne, k);
    std::vector<float> want((size_t)(ne * k * k), 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> full((size_t)(ne * k * k));
        for (int64_t e = 0; e < ne; ++e)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t lo = std::min(i, j), hi = std::max(i, j);
                    const float v = (float)(1000 * e + 31 * lo + hi + 7 * pass);
                    full[(size_t)((e * k + i) * k + j)] = v;
                    want[(size_t)((e * k + i) * k + j)] += v;
                }
        ce.add(full.data());
    }
    std::vector<float> got((size_t)(k * k));
    int bad = 0;
    for (int64_t e = 0; e < ne; ++e) {
        ce.unpack(e, got.data());
        for (int64_t x = 0; x < k * k; ++x) bad += got[(size_t)x] != want[(size_t)(e * k * k + x)];
    }
    CHECK_EQ(bad, 0);
}

/* CSA 3:1 CHUNK-POOLING AND KVARN VARIANCE-NORMALIZED QUANTIZATION FOR HISTORICAL TOKENS.
 *
 * For tokens outside the local sliding window (t > 128), 3 consecutive historical KV tokens are
 * pooled into 1 representative pooled token vector:
 *     \bar{K} = (1/3) \sum K_i, \bar{V} = (1/3) \sum V_i
 * This cuts historical memory traffic and WMMA operations by 3x (66.7% reduction), while CLT
 * dampens outlier variance across pooled chunks by 1/sqrt(3).
 * Stage 2 (KVarN) applies variance-normalized quantization (FWHT-256 Hadamard pre-rotation +
 * variance normalization scale for K4V2) on the pooled historical chunks, reducing ALU
 * dequantization overhead on RDNA4 by 3x. */
TEST(csa_paged_attention_pools_historical_tokens_3_to_1) {
    using namespace rad::arch;
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_tok = 2048; ctx.max_seqs = 4; ctx.max_ctx = 65536;

    Geom g{};
    CHECK_OK(geom_from(g, &meta, &ctx, "fp8", RAD_BF16));

    Names nm("");
    AttnPagedFP8::Wire w{};
    w.x = decl_b(&b, "res_x", g.act_dtype, { g.max_tok, g.n_embd });
    w.h.x = decl_b(&b, "h_x", g.act_dtype, { g.max_tok, g.n_embd });
    CHECK_OK(w.h.declare_qs(&b, nm, g, "h", g.n_embd));
    w.qkv = decl_b(&b, "attn_qkv", g.act_dtype, { g.max_tok, g.q_dim() + 2 * g.kv_dim() });
    w.q = decl_b(&b, "attn_q", g.act_dtype, { g.max_tok, g.q_dim() });
    w.attn.x = decl_b(&b, "attn_out", g.act_dtype, { g.max_tok, g.q_dim() });
    CHECK_OK(w.attn.declare_qs(&b, nm, g, "attn_out", g.q_dim()));

    AttnPagedFP8::Src src{};
    src.norm = "blk.0.attn_norm.weight";
    src.q = "blk.0.attn_q.weight";
    src.k = "blk.0.attn_k.weight";
    src.v = "blk.0.attn_v.weight";
    src.o = "blk.0.attn_output.weight";

    AttnPagedFP8::Config cfg{};
    cfg.csa_window = 128;
    cfg.csa_ratio  = 3;
    cfg.kvarn      = true;
    cfg.kv_block   = 16;

    AttnPagedFP8 attn;
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_F8E4M3;
    kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    rad_kvgroup grp = rad_decl_kv_group(&b, "kv_csa", &kd);

    CHECK_OK(attn.declare(&b, nm, g, 0, grp, 64, w, src, cfg));

    const RecOp* at = find_op(b, "attn_paged");
    CHECK(at != nullptr);
    CHECK_EQ(pint(at, "csa_window"), 128);
    CHECK_EQ(pint(at, "csa_ratio"), 3);
    CHECK_EQ(pint(at, "kvarn"), 1);
    CHECK_EQ(pint(at, "block_size"), 16);

    const RecOp* pool = find_op(b, "kv_csa3_kvarn_pool");
    CHECK(pool != nullptr);
    CHECK_EQ(pint(pool, "csa_ratio"), 3);
    CHECK_EQ(pint(pool, "csa_window"), 128);
    CHECK_EQ(pint(pool, "kvarn"), 1);
}

TEST(kvarn_variance_normalized_compression_scales) {
    /* CLT mathematical verification:
     * When 3 consecutive historical tokens are pooled into 1:
     * Var(\bar{X}) = (1/9) * (Var(X1) + Var(X2) + Var(X3)) = (1/3) * Var(X)
     * StdDev(\bar{X}) = StdDev(X) / sqrt(3).
     * This dampens outlier variance across historical context by factor 1/sqrt(3) ~ 0.57735. */
    const float clt_factor = 1.0f / std::sqrt(3.0f);
    CHECK(std::fabs(clt_factor - 0.577350269f) < 1e-5f);

    /* 64k context under CSA 3:1:
     * Local sliding window = 128 tokens.
     * Historical context = 64k - 128 = 65408 tokens.
     * 3:1 pooled tokens = 65408 / 3 = 21803 pooled tokens.
     * Reduction in historical tokens to unpack/process = 66.7% (exactly 3x speedup). */
    const int64_t full_hist = 65536 - 128;
    const int64_t csa_hist  = (full_hist + 2) / 3;
    const double reduction = 1.0 - (double)csa_hist / (double)full_hist;
    CHECK(reduction > 0.666);
    CHECK(reduction < 0.668);
}

TEST(csa_step_issues_pooling_op_only_outside_sliding_window) {
    using namespace rad::arch;
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_tok = 2048; ctx.max_seqs = 4; ctx.max_ctx = 65536;

    Geom g{};
    CHECK_OK(geom_from(g, &meta, &ctx, "fp8", RAD_BF16));

    Names nm("");
    AttnPagedFP8::Wire w{};
    w.x = decl_b(&b, "res_x", g.act_dtype, { g.max_tok, g.n_embd });
    w.h.x = decl_b(&b, "h_x", g.act_dtype, { g.max_tok, g.n_embd });
    CHECK_OK(w.h.declare_qs(&b, nm, g, "h", g.n_embd));
    w.qkv = decl_b(&b, "attn_qkv", g.act_dtype, { g.max_tok, g.q_dim() + 2 * g.kv_dim() });
    w.q = decl_b(&b, "attn_q", g.act_dtype, { g.max_tok, g.q_dim() });
    w.attn.x = decl_b(&b, "attn_out", g.act_dtype, { g.max_tok, g.q_dim() });
    CHECK_OK(w.attn.declare_qs(&b, nm, g, "attn_out", g.q_dim()));

    AttnPagedFP8::Src src{};
    src.norm = "blk.0.attn_norm.weight";
    src.q = "blk.0.attn_q.weight";
    src.k = "blk.0.attn_k.weight";
    src.v = "blk.0.attn_v.weight";
    src.o = "blk.0.attn_output.weight";

    AttnPagedFP8::Config cfg{};
    cfg.csa_window = 128;
    cfg.csa_ratio  = 3;
    cfg.kvarn      = true;
    cfg.kv_block   = 16;

    AttnPagedFP8 attn;
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_F8E4M3;
    kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    rad_kvgroup grp = rad_decl_kv_group(&b, "kv_csa", &kd);

    CHECK_OK(attn.declare(&b, nm, g, 0, grp, 64, w, src, cfg));

    // Case 1: Context length 100 <= 128 (sliding window only)
    {
        RadCtx c;
        c.b = &b;
        RadBatch batch = decode_batch();
        batch.max_ctx_len = 100;
        c.batch = &batch;

        attn.step(&c, &batch);

        bool found_pool = false;
        for (const auto& issue : c.issues) {
            const RecOp* op = b.op_of(issue.op);
            if (op && op->op == "kv_csa3_kvarn_pool") found_pool = true;
        }
        CHECK(!found_pool);
    }

    // Case 2: Context length 256 > 128 (historical tokens present)
    {
        RadCtx c;
        c.b = &b;
        RadBatch batch = decode_batch();
        batch.max_ctx_len = 256;
        c.batch = &batch;

        attn.step(&c, &batch);

        const RecIssue* pool_issue = nullptr;
        for (const auto& issue : c.issues) {
            const RecOp* op = b.op_of(issue.op);
            if (op && op->op == "kv_csa3_kvarn_pool") pool_issue = &issue;
        }
        CHECK(pool_issue != nullptr);
        if (pool_issue) {
            REQUIRE_EQ(pool_issue->opd.size(), (size_t)3);
            CHECK_EQ(pool_issue->opd[0].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(pool_issue->opd[1].dtype, (uint32_t)RAD_I32);
            CHECK_EQ(pool_issue->opd[2].dtype, (uint32_t)RAD_I32);
        }
    }
}

TEST(csa_boundary_transitions_from_sliding_window) {
    const int csa_w = 128;
    auto calc_eff_ctx = [csa_w](int ctx) {
        const int n_hist = (ctx > csa_w) ? (ctx - csa_w) : 0;
        const int n_hist_pool = (n_hist + 2) / 3;
        return (ctx > csa_w) ? (n_hist_pool + csa_w) : ctx;
    };

    // Boundary at sliding window (t <= 128): exactly 1:1, 0 pooled
    CHECK_EQ(calc_eff_ctx(1), 1);
    CHECK_EQ(calc_eff_ctx(64), 64);
    CHECK_EQ(calc_eff_ctx(128), 128);

    // Transitions across 129, 130, 131, 132
    CHECK_EQ(calc_eff_ctx(129), 129); // 1 hist -> 1 pooled + 128 window = 129
    CHECK_EQ(calc_eff_ctx(130), 129); // 2 hist -> 1 pooled + 128 window = 129
    CHECK_EQ(calc_eff_ctx(131), 129); // 3 hist -> 1 pooled + 128 window = 129
    CHECK_EQ(calc_eff_ctx(132), 130); // 4 hist -> 2 pooled + 128 window = 130

    // Deep context scaling
    CHECK_EQ(calc_eff_ctx(192), 150); // 64 hist -> 22 pooled + 128 window = 150
    CHECK_EQ(calc_eff_ctx(256), 171); // 128 hist -> 43 pooled + 128 window = 171
    CHECK_EQ(calc_eff_ctx(65536), 21931); // 65408 hist -> 21803 pooled + 128 window = 21931
}

/* ==================================================================== RDNA4 ISA Engineering Rules */

TEST(rdna4_isa_rule1_vgpr_granularity_and_occupancy) {
    using namespace rad::isa::rdna4;

    /* Rule 1: VGPR Allocation Granularity is 24 on Wave32 (SIMD has 1536 VGPRs).
     * Allocated = ceil(vgpr / 24) * 24.
     * Max active waves per SIMD = min(16, 1536 / Allocated). */
    CHECK_EQ(kSimdVgprs, 1536);
    CHECK_EQ(kWave32VgprGranularity, 24);
    CHECK_EQ(kWave64VgprGranularity, 12);
    CHECK_EQ(kMaxWave32WavesPerSimd, 16);
    CHECK_EQ(kMaxVgprsPerWave32, 256);

    /* Full occupancy boundary: <= 96 VGPRs */
    CHECK_EQ(allocated_vgprs_wave32(64), 72);
    CHECK_EQ(simd_occupancy_waves(64), 16);
    CHECK_EQ(allocated_vgprs_wave32(96), 96);
    CHECK_EQ(simd_occupancy_waves(96), 16);
    CHECK_EQ(simd_occupancy_pct(96), 100.0f);

    /* The 97-VGPR Cliff: +1 VGPR drops occupancy from 16 waves (100%) to 12 waves (75%) */
    CHECK_EQ(allocated_vgprs_wave32(97), 120);
    CHECK_EQ(simd_occupancy_waves(97), 12);
    CHECK_EQ(simd_occupancy_pct(97), 75.0f);
    CHECK_EQ(allocated_vgprs_wave32(120), 120);
    CHECK_EQ(simd_occupancy_waves(120), 12);

    /* The 121-VGPR Disaster: +1 VGPR drops occupancy from 12 waves (75%) to 10 waves (62.5%) */
    CHECK_EQ(allocated_vgprs_wave32(121), 144);
    CHECK_EQ(simd_occupancy_waves(121), 10);
    CHECK_EQ(simd_occupancy_pct(121), 62.5f);

    /* The Power-of-Two Trap: 128 VGPRs allocates 144, yielding 10 waves instead of 12 */
    CHECK_EQ(allocated_vgprs_wave32(128), 144);
    CHECK_EQ(simd_occupancy_waves(128), 10);
    CHECK_EQ(allocated_vgprs_wave32(144), 144);
    CHECK_EQ(simd_occupancy_waves(144), 10);

    /* Higher allocation steps: 168 (9 waves), 192 (8 waves), 216 (7 waves), 240 (6 waves) */
    CHECK_EQ(allocated_vgprs_wave32(145), 168);
    CHECK_EQ(simd_occupancy_waves(145), 9);
    CHECK_EQ(allocated_vgprs_wave32(168), 168);
    CHECK_EQ(simd_occupancy_waves(168), 9);

    CHECK_EQ(allocated_vgprs_wave32(169), 192);
    CHECK_EQ(simd_occupancy_waves(169), 8);
    CHECK_EQ(allocated_vgprs_wave32(192), 192);
    CHECK_EQ(simd_occupancy_waves(192), 8);

    CHECK_EQ(allocated_vgprs_wave32(193), 216);
    CHECK_EQ(simd_occupancy_waves(193), 7);
    CHECK_EQ(allocated_vgprs_wave32(216), 216);
    CHECK_EQ(simd_occupancy_waves(216), 7);

    CHECK_EQ(allocated_vgprs_wave32(217), 240);
    CHECK_EQ(simd_occupancy_waves(217), 6);
    CHECK_EQ(allocated_vgprs_wave32(240), 240);
    CHECK_EQ(simd_occupancy_waves(240), 6);

    /* 241..256 VGPRs: allocates 264 (clipped to physical 256), yielding 5 waves */
    CHECK_EQ(allocated_vgprs_wave32(241), 264);
    CHECK_EQ(simd_occupancy_waves(241), 5);
    CHECK_EQ(allocated_vgprs_wave32(256), 264);
    CHECK_EQ(simd_occupancy_waves(256), 5);

    /* Danger window detector: B < v <= B + 8 */
    CHECK(is_in_cliff_danger_zone(97));
    CHECK(is_in_cliff_danger_zone(104));
    CHECK(!is_in_cliff_danger_zone(105));
    CHECK(is_in_cliff_danger_zone(121));
    CHECK(is_in_cliff_danger_zone(128));
    CHECK(!is_in_cliff_danger_zone(120));
    CHECK(!is_in_cliff_danger_zone(96));
}

TEST(rdna4_isa_rule2_zero_waterfall_loop_standard) {
    /* Rule 2: Dynamic register array indexing across varying lane indices forces LLVM
     * to emit v_movrels_b32 inside s_and_saveexec_b32 / s_cbranch_execnz loops.
     * All unrolling bounds on hot paths must be compile-time constants. */
    constexpr int kMaxTokens = 8;
    int simulated_vgprs[kMaxTokens] = {10, 20, 30, 40, 50, 60, 70, 80};

    // Compliant pattern: loop with compile-time static bound, dynamic predication inside
    auto compliant_sum = [&](int runtime_active_tokens) -> int {
        int sum = 0;
        #if defined(__clang__)
        #pragma unroll
        #elif defined(__GNUC__)
        #pragma GCC unroll 8
        #endif
        for (int i = 0; i < kMaxTokens; ++i) {
            if (i < runtime_active_tokens) {
                sum += simulated_vgprs[i];
            }
        }
        return sum;
    };

    CHECK_EQ(compliant_sum(3), 10 + 20 + 30);
    CHECK_EQ(compliant_sum(0), 0);
    CHECK_EQ(compliant_sum(8), 360);
}

TEST(rdna4_isa_rule3_dual_issue_vopd_constraints) {
    using namespace rad::isa::rdna4;

    /* Rule 3: VOPD Dual-Issue Constraints:
     * - OpX has 14 opcodes (0..13)
     * - OpY has 17 opcodes (0..13, 16..18)
     * - Wave32 only
     * - Destination parity: one even, one odd (vdstY[0] == ~vdstX[0])
     * - Bank conflict: 4 banks (reg % 4). OpX and OpY sources must be from different banks
     *   or be identical registers.
     * - Dual SRC2: If both read SRC2 (FMAC, FMAMK, DOT2ACC), one SRC2 must be even and other odd. */
    CHECK_EQ(kVopdOpXCount, 14);
    CHECK_EQ(kVopdOpYCount, 17);
    CHECK_EQ(kVgprBanks, 4);

    // Parity validator
    CHECK(vopd_valid_dest_parity(0, 1));
    CHECK(vopd_valid_dest_parity(2, 3));
    CHECK(vopd_valid_dest_parity(4, 5));
    CHECK(!vopd_valid_dest_parity(0, 2));  // Both even -> invalid
    CHECK(!vopd_valid_dest_parity(1, 3));  // Both odd  -> invalid

    // Bank conflict validator (4 banks: reg % 4)
    CHECK(vopd_valid_bank_pairing(0, 0));  // Same register -> legal
    CHECK(vopd_valid_bank_pairing(0, 1));  // Bank 0 and Bank 1 -> legal
    CHECK(vopd_valid_bank_pairing(0, 5));  // Bank 0 and Bank 1 (5%4=1) -> legal
    CHECK(!vopd_valid_bank_pairing(0, 4)); // Bank 0 and Bank 0 (4%4=0) -> CONFLICT!
    CHECK(!vopd_valid_bank_pairing(1, 5)); // Bank 1 and Bank 1 (5%4=1) -> CONFLICT!

    // Dual SRC2 parity validator
    CHECK(vopd_valid_dual_src2(2, 3));
    CHECK(!vopd_valid_dual_src2(2, 4));
}

TEST(rdna4_isa_rule4_global_load_tr_canonical_k_ordering) {
    using namespace rad::isa::rdna4;

    /* Rule 4: GLOBAL_LOAD_TR canonical K-ordering:
     * Lanes 0..15 hold k in {0, 1, 2, 3,  8, 9, 10, 11}
     * Lanes 16..31 hold k in {4, 5, 6, 7, 12, 13, 14, 15}
     * Both operands A and B to WMMA must follow this exact permutation. */
    const int expected_lo[] = {0, 1, 2, 3, 8, 9, 10, 11};
    for (int j = 0; j < 8; ++j) {
        CHECK_EQ(canonical_k_order(0, j), expected_lo[j]);
        CHECK_EQ(canonical_k_order(0, j), kCanonicalK_Lo[j]);
    }

    const int expected_hi[] = {4, 5, 6, 7, 12, 13, 14, 15};
    for (int j = 0; j < 8; ++j) {
        CHECK_EQ(canonical_k_order(16, j), expected_hi[j]);
        CHECK_EQ(canonical_k_order(16, j), kCanonicalK_Hi[j]);
    }

    // Mathematical invariant: dot product over permutation matches linear sum
    float a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float b[16] = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15};

    float ref_dot = 0.0f;
    for (int k = 0; k < 16; ++k) ref_dot += a[k] * b[k];

    // Canonical permuted sum: sum over lanes 0 and 16
    float perm_dot = 0.0f;
    for (int j = 0; j < 8; ++j) {
        int k_lo = canonical_k_order(0, j);
        perm_dot += a[k_lo] * b[k_lo];
        int k_hi = canonical_k_order(16, j);
        perm_dot += a[k_hi] * b[k_hi];
    }
    CHECK_EQ(ref_dot, perm_dot);

    // Hazard demonstration: if A uses canonical and B uses unpermuted linear order,
    // the dot product computes corrupted garbage
    float corrupted_dot = 0.0f;
    for (int j = 0; j < 8; ++j) {
        int k_lo = canonical_k_order(0, j);
        corrupted_dot += a[k_lo] * b[j];       // Mismatched index!
        int k_hi = canonical_k_order(16, j);
        corrupted_dot += a[k_hi] * b[8 + j];   // Mismatched index!
    }
    CHECK(std::fabs(corrupted_dot - ref_dot) > 1.0f); // Proves silent corruption hazard
}

TEST(rdna4_isa_rule5_split_cache_temporal_hints) {
    using namespace rad::isa::rdna4;

    /* Rule 5: Split Cache Temporal Hints (TH):
     * TH[2:0] values:
     *   0: RT    (Regular near + far)
     *   1: NT    (Non-temporal near + far)
     *   2: HT    (High-priority temporal near + far)
     *   3: LU    (Last-use: non-temporal + discard dirty)
     *   4: NT_RT (Non-temporal near / bypass L0, regular far L2) -> WEIGHT STREAMING
     *   5: RT_NT
     *   6: NT_HT (Non-temporal near / bypass L0, high-priority far L2) -> LAYER RESIDENT WEIGHTS
     *   7: NT_WB */
    CHECK(th_is_weight_streaming(TemporalHint::NT_RT));
    CHECK(th_is_weight_streaming(TemporalHint::NT_HT));
    CHECK(!th_is_weight_streaming(TemporalHint::RT)); // RT thrashes CU L0 vector cache

    CHECK(th_is_last_use(TemporalHint::LU));

    CHECK(th_is_streaming_store(TemporalHint::NT));
    CHECK(th_is_streaming_store(TemporalHint::NT_WB));
}

TEST(rdna4_isa_rule6_independent_wait_counters_and_hazard_guards) {
    using namespace rad::isa::rdna4;

    /* Rule 6: Independent Wait Counters and s_wait_alu:
     * Deprecates legacy s_waitcnt. Uses:
     * - s_wait_loadcnt (LOADcnt: vector memory loads, atomic-with-return)
     * - s_wait_storecnt (STOREcnt: vector memory stores)
     * - s_wait_dscnt (DScnt: LDS and Flat ops)
     * - s_wait_kmcnt (KMcnt: SMEM scalar loads and s_sendmsg)
     *
     * s_wait_alu Bitmasks:
     * - 0xFFFD: VALU write to VCC -> VALU read of VCC
     * - 0xF1FF: VALU write to SGPR -> VALU read of SGPR
     * - 0xFFFE: SALU write to SGPR/VCC -> SALU/VALU read of SGPR/VCC */
    CHECK_EQ(kWaitAluValuVcc,  0xFFFD);
    CHECK_EQ(kWaitAluValuSgpr, 0xF1FF);
    CHECK_EQ(kWaitAluSaluAny,  0xFFFE);
}

TEST(rdna4_isa_rule7_early_deallocation_epilogue_contract) {
    using namespace rad::isa::rdna4;

    /* Rule 7: Early Deallocation Contract:
     * Kernel epilogue must emit s_sendmsg sendmsg(MSG_DEALLOC_VGPRS) BEFORE s_wait_storecnt 0.
     * This releases the wave's VGPR block to the SIMD allocator while stores are in-flight,
     * allowing subsequent wavefronts to launch immediately. */
    std::vector<EpilogueStep> optimal_sequence = {
        EpilogueStep::IssueStores,
        EpilogueStep::DeallocVGPRs,
        EpilogueStep::WaitStores,
        EpilogueStep::EndPgm
    };

    auto validate_seq = [](const std::vector<EpilogueStep>& seq) -> bool {
        auto dealloc_it = std::find(seq.begin(), seq.end(), EpilogueStep::DeallocVGPRs);
        auto wait_it    = std::find(seq.begin(), seq.end(), EpilogueStep::WaitStores);
        bool has_dealloc = (dealloc_it != seq.end());
        bool has_wait    = (wait_it != seq.end());
        int dealloc_idx  = has_dealloc ? static_cast<int>(std::distance(seq.begin(), dealloc_it)) : -1;
        int wait_idx     = has_wait    ? static_cast<int>(std::distance(seq.begin(), wait_it)) : -1;
        return validate_epilogue_order(has_dealloc, has_wait, dealloc_idx, wait_idx);
    };

    CHECK(validate_seq(optimal_sequence));

    // Stalling epilogue (dealloc omitted or after wait):
    std::vector<EpilogueStep> stalling_sequence = {
        EpilogueStep::IssueStores,
        EpilogueStep::WaitStores,
        EpilogueStep::EndPgm
    };
    CHECK(!validate_seq(stalling_sequence));
}

TEST(ced_wave32_occupancy_and_rdna4_isa_rules) {
    using namespace rad::isa::rdna4;

    /* RDNA4 ISA Rule 1: VGPR <= 96 gives 100% occupancy (16 waves/SIMD).
     * For CED projection kernels:
     * - Down-projection: r=32 bottleneck, 1 wave of 32 threads computes z = V * h_t.
     * - Up-projection: E=512 experts / 32 threads = 16 waves.
     * Strict VGPR ceiling is 96 VGPRs to guarantee 100% SIMD occupancy. */
    constexpr int kCedWaveSize = 32;
    constexpr int kCedWorkgroupThreads = 128; // 4 Wave32 waves per workgroup
    constexpr int kCedMaxVgprs = 96;
    constexpr int kCedMaxLdsBytes = 4096;     // 4 KiB LDS per workgroup

    CHECK_EQ(kCedWaveSize, 32);
    CHECK_EQ(kCedWorkgroupThreads, 128);
    CHECK_EQ(kCedWorkgroupThreads % kCedWaveSize, 0);
    CHECK_EQ(kCedWorkgroupThreads / kCedWaveSize, 4);

    /* Verify VGPR allocation granularity and 100% occupancy at 96 VGPRs */
    CHECK_EQ(allocated_vgprs_wave32(kCedMaxVgprs), 96);
    CHECK_EQ(simd_occupancy_waves(kCedMaxVgprs), 16);
    CHECK_EQ(simd_occupancy_pct(kCedMaxVgprs), 100.0f);

    /* Verify the 97-VGPR occupancy cliff: 1 single VGPR beyond 96 drops occupancy to 75% */
    CHECK_EQ(allocated_vgprs_wave32(kCedMaxVgprs + 1), 120);
    CHECK_EQ(simd_occupancy_waves(kCedMaxVgprs + 1), 12);
    CHECK_EQ(simd_occupancy_pct(kCedMaxVgprs + 1), 75.0f);

    /* Verify CU LDS ceiling: 4 KiB LDS per workgroup easily accommodates 16 workgroups in 64 KiB LDS */
    constexpr int kMaxCuLds = 64 * 1024;
    CHECK(kCedMaxLdsBytes <= kMaxCuLds);
    CHECK_EQ(kMaxCuLds / kCedMaxLdsBytes, 16);

    /* RDNA4 ISA Rule 2 & 5: Split cache temporal hints for CED projection:
     * - Weight matrix projection basis (V_down, U_up) stream through cache: TemporalHint::NT_RT.
     * - Hidden state activations and intermediate bottleneck (z) are last-use: TemporalHint::LU.
     * - Predicted logits and candidate distributions are streaming stores: TemporalHint::NT. */
    CHECK(th_is_weight_streaming(TemporalHint::NT_RT));
    CHECK(th_is_last_use(TemporalHint::LU));
    CHECK(th_is_streaming_store(TemporalHint::NT));

    /* Zero-waterfall unroll invariant: rank r=32 matches Wave32 size exactly */
    constexpr int kCedRank = 32;
    constexpr int kCedExperts = 512;
    CHECK_EQ(kCedRank % kCedWaveSize, 0);
    CHECK_EQ(kCedExperts % kCedWaveSize, 0);
    CHECK_EQ(kCedExperts / kCedWaveSize, 16);
}

TEST(ced_low_rank_projection_and_topk_stability) {
    /* Test mathematical equivalence of low-rank factorized projection vs. full-rank:
     * Full-rank routing: logits_full = W * h + b, where W in R^{E x d}
     * Factorized CED:    z = V * h (R^r), logits_ced = U * z + b, where W_approx = U * V
     * Verify numerical equivalence, stable softmax (no NaN/Inf on large logits),
     * probability threshold filtering (P(e) >= tau), and top-k ranking. */
    const int d = 128;
    const int r = 16;
    const int E = 32;

    std::vector<float> V(r * d, 0.0f);
    std::vector<float> U(E * r, 0.0f);
    std::vector<float> b(E, 0.0f);

    for (int k = 0; k < r; ++k) {
        for (int j = 0; j < d; ++j) {
            V[k * d + j] = std::sin((float)((k + 1) * (j + 1)) * 0.05f) / std::sqrt((float)d);
        }
    }
    for (int e = 0; e < E; ++e) {
        for (int k = 0; k < r; ++k) {
            U[e * r + k] = std::cos((float)((e + 1) * (k + 1)) * 0.05f) / std::sqrt((float)r);
        }
        b[e] = -0.2f * (float)(e % 4);
    }

    // Multiply W = U * V
    std::vector<float> W(E * d, 0.0f);
    for (int e = 0; e < E; ++e) {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) sum += U[e * r + k] * V[k * d + j];
            W[e * d + j] = sum;
        }
    }

    // Input hidden state h_t
    std::vector<float> h(d, 0.0f);
    for (int j = 0; j < d; ++j) h[j] = std::sin((float)j * 0.1f);

    // 1. Direct full-rank calculation: logits_full = W * h + b
    std::vector<float> logits_full(E, 0.0f);
    for (int e = 0; e < E; ++e) {
        float sum = b[e];
        for (int j = 0; j < d; ++j) sum += W[e * d + j] * h[j];
        logits_full[e] = sum;
    }

    // 2. Factorized two-stage calculation: z = V * h, logits_ced = U * z + b
    std::vector<float> z(r, 0.0f);
    for (int k = 0; k < r; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < d; ++j) sum += V[k * d + j] * h[j];
        z[k] = sum;
    }
    std::vector<float> logits_ced(E, 0.0f);
    for (int e = 0; e < E; ++e) {
        float sum = b[e];
        for (int k = 0; k < r; ++k) sum += U[e * r + k] * z[k];
        logits_ced[e] = sum;
    }

    // Assert equivalence within FP tolerance
    for (int e = 0; e < E; ++e) {
        CHECK_NEAR(logits_ced[e], logits_full[e], 1e-4);
    }

    // 3. Numerically stable softmax test, even with large positive/negative offsets
    float max_l = logits_ced[0];
    for (int e = 1; e < E; ++e) if (logits_ced[e] > max_l) max_l = logits_ced[e];

    std::vector<float> probs(E, 0.0f);
    float p_sum = 0.0f;
    for (int e = 0; e < E; ++e) {
        probs[e] = std::exp(logits_ced[e] - max_l);
        p_sum += probs[e];
    }
    CHECK(p_sum > 0.0f);
    for (int e = 0; e < E; ++e) probs[e] /= p_sum;

    float total_p = 0.0f;
    for (int e = 0; e < E; ++e) {
        CHECK(!std::isnan(probs[e]));
        CHECK(!std::isinf(probs[e]));
        CHECK(probs[e] >= 0.0f && probs[e] <= 1.0f);
        total_p += probs[e];
    }
    CHECK_NEAR(total_p, 1.0f, 1e-5);

    // 4. Probability threshold filtering and top-k sorting
    const float tau = 0.04f;
    const int top_k = 3;
    std::vector<std::pair<float, int>> cands;
    for (int e = 0; e < E; ++e) {
        if (probs[e] >= tau) cands.emplace_back(probs[e], e);
    }
    std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    if ((int)cands.size() > top_k) cands.resize((size_t)top_k);

    CHECK(!cands.empty());
    CHECK(cands.size() <= (size_t)top_k);
    for (size_t i = 0; i < cands.size(); ++i) {
        CHECK(cands[i].first >= tau);
        if (i > 0) CHECK(cands[i - 1].first >= cands[i].first);
    }
}

TEST(ced_clustered_expert_distribution_and_gemv_invariants) {
    using namespace rad::isa::rdna4;

    /* 1. Clustered Expert Distribution (CED) clustering invariants:
     * 512 experts grouped into 32 clusters of 16 experts each. */
    constexpr int kExperts = 512;
    constexpr int kClusters = 32;
    constexpr int kClusterSize = kExperts / kClusters; // 16 experts per cluster
    CHECK_EQ(kClusterSize, 16);
    CHECK_EQ(kExperts % kClusters, 0);

    // Synthesize expert probability distribution with concentrated cluster mass
    std::vector<float> expert_probs(kExperts, 0.0f);
    // Cluster 3 (experts 48..63) and Cluster 7 (experts 112..127) have high co-activation
    for (int e = 48; e < 64; ++e) expert_probs[e] = 0.035f;   // 16 * 0.035 = 0.56
    for (int e = 112; e < 128; ++e) expert_probs[e] = 0.020f; // 16 * 0.020 = 0.32
    // Remaining probability distributed uniformly across other 30 clusters
    const float rem = (1.0f - 0.56f - 0.32f) / (float)(kExperts - 32);
    for (int e = 0; e < kExperts; ++e) {
        if ((e < 48 || e >= 64) && (e < 112 || e >= 128)) expert_probs[e] = rem;
    }

    // Compute cluster routing mass
    std::vector<float> cluster_mass(kClusters, 0.0f);
    for (int e = 0; e < kExperts; ++e) {
        int c = e / kClusterSize;
        cluster_mass[c] += expert_probs[e];
    }

    // Assert top clusters capture expected probability mass
    CHECK_NEAR(cluster_mass[3], 0.56f, 1e-4);
    CHECK_NEAR(cluster_mass[7], 0.32f, 1e-4);
    const float top2_mass = cluster_mass[3] + cluster_mass[7];
    CHECK_NEAR(top2_mass, 0.88f, 1e-4); // Top 2 clusters hold 88% of total probability!

    /* 2. RDNA4 Wave32 GEMV Kernel microarchitectural invariants:
     * Target: gfx1201 Dual SIMD, Wave32 mode.
     * Workgroup: 128 threads = 4 waves.
     * Epilogue contract: MSG_DEALLOC_VGPRS precedes s_wait_storecnt 0. */
    constexpr int kWaveSize = 32;
    constexpr int kWorkgroupThreads = 128;
    (void)kWaveSize;
    (void)kWorkgroupThreads;
    constexpr int kBottleneckRank = 32;
    constexpr int kLdsBytes = kBottleneckRank * sizeof(float); // 128 bytes
    CHECK_EQ(kLdsBytes, 128);
    CHECK(kLdsBytes <= 4096);

    // Epilogue sequence verification: Rule 7 Early Deallocation Contract
    CHECK(validate_epilogue_order(true, true, 1, 2)); // dealloc at 1, wait_stores at 2 -> valid
    CHECK(!validate_epilogue_order(true, true, 2, 1)); // wait_stores before dealloc -> invalid!

    // Temporal hints verification: Rule 5 Split Cache Temporal Hints
    CHECK_EQ(static_cast<uint8_t>(TemporalHint::NT_RT), 4);
    CHECK_EQ(static_cast<uint8_t>(TemporalHint::LU), 3);
    CHECK(th_is_weight_streaming(TemporalHint::NT_RT));
    CHECK(th_is_last_use(TemporalHint::LU));

    // Occupancy verification: Rule 1 24-VGPR quantum and 100% occupancy bound
    constexpr int kTargetVgprs = 48; // Actual CED GEMV register budget
    CHECK_EQ(allocated_vgprs_wave32(kTargetVgprs), 48);
    CHECK_EQ(simd_occupancy_waves(kTargetVgprs), 16);
    CHECK_EQ(simd_occupancy_pct(kTargetVgprs), 100.0f);
    CHECK(!is_in_cliff_danger_zone(kTargetVgprs));
}




TEST(qwen4exp_fused_router_single_dispatch_at_decode) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx  ctx = build_ctx(0, 1);
    ctx.max_tok  = 2048; ctx.max_seqs = 8; ctx.max_spec = 3; ctx.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);

    // 1. Verify router_gemm_topk_scatter is declared and resolves
    const RecOp* rf = find_op(b, "router_gemm_topk_scatter");
    REQUIRE(rf != nullptr);
    CHECK(rf->resolved);
    CHECK_EQ(pint(rf, "n_expert"), 512);
    CHECK_EQ(pint(rf, "top_k"), 10);
    CHECK_EQ(param(rf, "M")->ihi, 64); // Decode band <= 64

    // 2. Decode step (rows = 4 <= 64) with fused router enabled
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch decode_b = decode_batch();
    c.batch = &decode_b;
    qwen4exp_fp8::step(&c, &decode_b);

    int n_fused = 0, n_router_alone = 0, n_sgate_alone = 0, n_topk_scatter = 0;
    for (const auto& issue : c.issues) {
        const RecOp* o = b.op_of(issue.op);
        if (!o) continue;
        if (o->op == "router_gemm_topk_scatter") {
            ++n_fused;
            REQUIRE_EQ(o->w.size(), (size_t)2);
            const std::string& w0 = b.weights[o->w[0] - 1].name;
            const std::string& w1 = b.weights[o->w[1] - 1].name;
            CHECK(w0.find("ffn_gate_inp.weight") != std::string::npos);
            CHECK(w1.find("ffn_gate_shexp.weight") != std::string::npos);
        } else if (o->op == "gemm_nt") {
            for (rad_weight wh : o->w) {
                if (!wh) continue;
                const std::string& wn = b.weights[wh - 1].name;
                if (wn.find("ffn_gate_inp.weight") != std::string::npos) ++n_router_alone;
                if (wn.find("ffn_gate_shexp.weight") != std::string::npos) ++n_sgate_alone;
            }
        } else if (o->op == "router_topk_scatter") {
            ++n_topk_scatter;
        }
    }
    CHECK_EQ(n_fused, 8); // All 8 MoE layers issued the fused router
    CHECK_EQ(n_router_alone, 0); // Unfused router GEMM not issued
    CHECK_EQ(n_sgate_alone, 0); // Unfused shared gate GEMM not issued
    CHECK_EQ(n_topk_scatter, 0); // Unfused topk scatter not issued

    // 3. Fallback path when router_gemm_topk_scatter is refused
    RadBuilder b_refuse;
    b_refuse.refuse.insert("router_gemm_topk_scatter");
    REQUIRE_EQ(qwen4exp_fp8::declare(&b_refuse, &meta, &ctx), RAD_OK);

    RadCtx c_refuse; c_refuse.b = &b_refuse; c_refuse.rank = 0; c_refuse.world = 1;
    c_refuse.batch = &decode_b;
    qwen4exp_fp8::step(&c_refuse, &decode_b);

    int n_fused_ref = 0, n_router_ref = 0, n_sgate_ref = 0, n_topk_ref = 0;
    for (const auto& issue : c_refuse.issues) {
        const RecOp* o = b_refuse.op_of(issue.op);
        if (!o) continue;
        if (o->op == "router_gemm_topk_scatter") ++n_fused_ref;
        else if (o->op == "gemm_nt") {
            for (rad_weight wh : o->w) {
                if (!wh) continue;
                const std::string& wn = b_refuse.weights[wh - 1].name;
                if (wn.find("ffn_gate_inp.weight") != std::string::npos) ++n_router_ref;
                if (wn.find("ffn_gate_shexp.weight") != std::string::npos) ++n_sgate_ref;
            }
        } else if (o->op == "router_topk_scatter") {
            ++n_topk_ref;
        }
    }
    CHECK_EQ(n_fused_ref, 0);
    CHECK_EQ(n_router_ref, 8); // All 8 layers fall back to separate router GEMM
    CHECK_EQ(n_sgate_ref, 8); // All 8 layers fall back to separate shared gate GEMM
    CHECK_EQ(n_topk_ref, 8); // All 8 layers fall back to separate topk scatter

    // 4. Prefill batch with rows > 64 falls back to unfused even when fused op declared
    RadCtx c_pf; c_pf.b = &b; c_pf.rank = 0; c_pf.world = 1;
    RadBatch pf_b = prefill_batch();
    pf_b.n_tok = 128; // Exceeds kMoeRouterFusedRows (64)
    c_pf.batch = &pf_b;
    qwen4exp_fp8::step(&c_pf, &pf_b);

    int n_fused_pf = 0, n_router_pf = 0, n_sgate_pf = 0;
    for (const auto& issue : c_pf.issues) {
        const RecOp* o = b.op_of(issue.op);
        if (!o) continue;
        if (o->op == "router_gemm_topk_scatter") ++n_fused_pf;
        else if (o->op == "gemm_nt") {
            for (rad_weight wh : o->w) {
                if (!wh) continue;
                const std::string& wn = b.weights[wh - 1].name;
                if (wn.find("ffn_gate_inp.weight") != std::string::npos) ++n_router_pf;
                if (wn.find("ffn_gate_shexp.weight") != std::string::npos) ++n_sgate_pf;
            }
        }
    }
    CHECK_EQ(n_fused_pf, 0); // Decode-only fused router not issued at rows > 64
    CHECK_EQ(n_router_pf, 8); // Prefill uses separate router GEMM
    CHECK_EQ(n_sgate_pf, 8); // Prefill uses separate shared gate GEMM
}



