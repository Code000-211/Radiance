/* rad_build.h -- the declare phase (spec §3.1). The architecture plugin states everything it will
 * ever do; this is what receives it.
 *
 * Two properties are the whole point of the phase and everything here is arranged around them:
 *
 *   DECLARE ALWAYS COMPLETES. A missing kernel does not stop the walk. Every op, every band and
 *   every domain that resolved to nothing is accumulated with the constraint that refused it, and
 *   the full list is reported at the end. The first failure at the top tells you one thing; the
 *   whole list tells you whether you are missing a plugin or missing a model.
 *
 *   NOTHING IS DECIDED TWICE. Between declare and plan the model is known statically -- every op,
 *   every kernel that will service it, every weight it touches, and the order. The bucket table,
 *   the weight manifest, the buffer plan and the KV block sizes are all built once, here, so the
 *   step path indexes tables instead of asking questions.
 *
 * WHAT THIS COMPONENT DOES NOT DO. The kernel hierarchy, the op schemas, the constraint match and
 * the bucket boundaries are core/plugin/'s, and the Builder asks rather than reimplements: it
 * calls Registry::validate for the schema check and Registry::build_bands for the bucket table.
 * Two answers to one question is how a selector and a builder drift apart.
 *
 * LIFETIME. Every KernelRow, Resolved and RadOpSchema in the Program points into a plugin's static
 * tables, which the Registry dlclose()s. The Program must not outlive the Registry it was built
 * against, and neither may.
 */
#pragma once
#include "plugin/registry.h"

#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/* The handle the plugin holds. rad_builder.h only forward-declares it; core/build/ owns it, and
 * making it a base of rad::Builder means the C shims are a static_cast rather than a lookup. */
struct RadBuilder { };

namespace rad {

/* Process-lifetime storage for declared names.
 *
 * RadNameMap and RadKernelInfo hold raw `const char*`, and the Program outlives the Builder that
 * filled it -- so a std::string member would dangle and there is nowhere in the frozen struct to
 * put one. These are interned instead: a few thousand short strings, deduplicated, never freed.
 * Freeing them would mean making RadNameMap own its storage, which is an ABI change for a few
 * kilobytes. */
const char* intern(std::string_view s);

/* Names for the enums the graph dump prints. `bld_` prefixed because tier_name and site_name in
 * rad_core.h belong to the planner and these are the same kind of thing for a different axis. */
const char* bld_domain_name(int domain);
const char* bld_access_name(int access);
const char* bld_buf_kind_name(int kind);
const char* bld_kv_kind_name(int kind);

/* ------------------------------------------------------------------ the buffer plan */
/* What the packing produced, for the --debug-graph report. `sum_bytes` is what no sharing at all
 * would have cost and `peak_live` is what a perfect packer would reach, so the pair brackets the
 * arena and says whether the greedy pass left anything on the table. */
struct BufPlanReport {
    int64_t transient_bytes = 0;   /* the packed, shared region */
    int64_t persist_bytes   = 0;   /* PERSIST + DERIVED: their own non-overlapping region */
    int64_t host_bytes      = 0;   /* domain=HOST buffers; a separate address space (see below) */
    int64_t peak_live       = 0;   /* max over op index of the bytes live at it */
    int64_t sum_bytes       = 0;   /* every transient, unshared */
    int     n_unknown_live  = 0;   /* transients whose liveness nothing declared */
};

/* Liveness must already be in BufferInfo::first_def / last_use; a transient with first_def < 0 is
 * treated as live for the whole program, which is the sound assumption and not a guess.
 *
 * Assigns BufferInfo::arena_offset for every buffer and sets Program::arena_bytes. Returns RAD_OK,
 * or RAD_E_INVAL if a buffer could not be sized (a plugin-private dtype the core cannot measure). */
int plan_buffers(Program& p, BufPlanReport* rep = nullptr);

/* ONE SIZE OF STEP'S ARENA LAYOUT. The same buffers as the real plan, the fixed region where it
 * was, and the transients packed for a step of at most `rows` tokens -- so the arena past `end`
 * is unused by any such step and can be lent. `offset` and `decl` are indexed by buffer handle. */
struct ArenaLevel {
    int64_t rows = 0;
    int64_t end = 0;
    std::vector<int64_t>    offset;
    std::vector<RadBufDecl> decl;
};
/* Lay out `probe`'s buffers -- a sizing declare at `rows` -- over `full`'s. Refused, with the
 * reason in `why`, when the two declares do not describe the same buffers with only their rows
 * smaller: anything else cannot share the real plan's fixed region and kernels. */
int plan_level(const Program& full, const Program& probe, int64_t rows, ArenaLevel* out,
               std::string* why);

/* Bytes a declared buffer occupies. 0 means the core cannot size it -- a plugin-private dtype,
 * which is legal for a WEIGHT (the kernel's layout hook sizes those) and not for an activation. */
int64_t buffer_bytes(const RadBufDecl& d);

/* ------------------------------------------------------------------ the model behind a declare */
/* WHAT THE MODEL HOLDS FOR A LOGICAL WEIGHT: its encoding and its logical extents. This is what
 * rad_weight_encoding answers and what every declaration is checked against (spec §4.3). The
 * engine answers from the container or the checkpoint it loads, rad-convert from the recipe it
 * will quantise by; `prog` is the declare so far, whose name map is how a checkpoint is searched.
 * RAD_OK, or RAD_E_NOTFOUND for a weight the model does not have. */
struct WeightSource {
    RadEncoding enc{};
    uint32_t    rank = 0;
    int64_t     shape[RAD_MAX_RANK] = {0};
};
using WeightSourceFn = std::function<int(const std::string& name, const Program& prog,
                                         WeightSource* out)>;

/* ------------------------------------------------------------------ the builder */
class Builder : public RadBuilder {
public:
    /* The Registry must outlive the Builder AND the Program it produces: every KernelRow the
     * bands hold points into a plugin the Registry dlclose()s. */
    Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx);

    /* Where rad_weight_encoding's answers come from. Unset, every source is RAD_E_NOTFOUND and a
     * weight is read as plain in its declared dtype -- a test, a synthetic tool. */
    void set_sources(WeightSourceFn f) { sources_ = std::move(f); }

    /* The C surface, one method per rad_builder.h entry point. */
    rad_weight  decl_weight  (const char* name, const RadWeightDecl* d);
    int         weight_encoding(const char* source, RadEncoding* out, int64_t* shape,
                                uint32_t* rank);
    rad_buf     decl_buffer  (const char* name, const RadBufDecl* d);
    rad_kvgroup decl_kv_group(const char* name, const RadKVGroupDecl* d);
    int64_t     kv_block_size(rad_kvgroup g);
    int         bind_layer_kv(int layer, rad_kvgroup g);
    rad_op      decl_op      (const char* op, const RadParam* p, int n_p,
                              const rad_weight* w, int n_w);
    int         op_resolved  (rad_op h) const;
    int         decl_name_map(const RadNameMap* m);
    int         declare_logits(rad_buf b);
    void disable_branched_tree() { prog_.branched_tree_supported = false; }
    int         declare_drafter(const RadDrafterDecl* d);
    int         declare_drafter_tree(rad_buf tree, int64_t node_capacity);
    int         declare_encoder(const RadEncoderDecl* d);
    void        note_v       (const char* f, va_list ap);

    /* The buffer operands of an op.
     *
     * NOT in rad_builder.h, and it needs to be -- see the proposed ABI addition at the foot of this
     * header. rad_decl_op names an op's WEIGHT operands but not its buffer operands, so declare
     * cannot compute activation liveness from the ABI as frozen, and without liveness the arena is
     * the sum of every transient rather than the peak of them. Until the two functions below are added to the frozen header (purely additive,
     * no existing signature changes), a plugin cannot supply this and every transient is assumed
     * live for the whole program: correct, and as wasteful as it sounds. The --debug-graph buffer
     * report prints how many buffers fell into that case, so the cost is never invisible. */
    int op_reads (rad_op h, const rad_buf* b, int n);
    int op_writes(rad_op h, const rad_buf* b, int n);
    int buf_concurrent(rad_buf b);
    int weight_shard_span(rad_weight w, int64_t lo, int64_t hi);

    /* Finalise: weight first/last use, KV group geometry, buffer liveness and packing. Returns
     * RAD_OK, or the first structural error class. Resolution MISSES are not an error here -- an
     * op that did not resolve returns a null handle and the plugin is expected to have offered a
     * fallback (spec §2.3); it becomes an error when the run phase issues it. */
    int finish();

    /* A SIZING DECLARE (RadBuildCtx::shape_probe): every op takes the bands `ref` resolved for
     * the op at the same position instead of resolving its own, and finish() initialises nothing.
     * The plugin then sees exactly the answers the real declare gave -- which ops resolved, which
     * block size a group has -- so it declares the same graph, and nothing is selected or set up
     * a second time. An op that is not the one `ref` has at that position ends the probe with an
     * error: the plugin declared a different graph at the smaller size, and its buffers cannot be
     * laid over the real ones. */
    void set_reference(const Program* ref) { ref_ = ref; }

    /* WITHOUT INSTANCES: finish() selects every kernel and initialises none. For a caller that
     * launches nothing -- rad-convert plans a container's stored form, which is the kernels'
     * layouts and not their setup. A kernel's init is about the process it runs in (a ring, a
     * peer's buffers) and the conversion is often not that process: libavx's n-gram gather
     * refuses a process that may not make an io_uring ring, and a sandbox that forbids one still
     * has to be able to convert the model. */
    void set_instances(bool on) { instances_ = on; }

    Program&       program()       { return prog_; }
    const Program& program() const { return prog_; }

    /* Structural failures: a bad schema, two ranged parameters, a duplicate name. Every one of
     * them, not the first -- same rule as the miss list. */
    const std::vector<std::string>& errors() const { return errors_; }
    int  status() const { return status_; }
    const BufPlanReport& buf_report() const { return bufrep_; }

private:
    /* Record a structural error and remember its class. Returns 0 so callers can `return err(...)`
     * straight out of a decl_ function, which is what makes a failed decl_* falsy. */
    uint32_t err(int code, const char* f, ...) __attribute__((format(printf, 3, 4)));
    std::string scoped(const char* name) const;

    void resolve_bands(OpInfo& oi);
    void record_tune_request(const Resolved& r);
    /* init() for every surviving instance, run once the WHOLE graph has resolved. Running it
     * during resolution would let a failing init abort declare before the miss report was
     * assembled, and the miss report is the point of the phase. */
    void finish_instances();
    int32_t find_kv_consumer(rad_kvgroup g);
    void finish_layouts();
    void finish_kv_groups();
    void finish_weight_uses();
    void finish_buffer_liveness();

    Registry& reg_;
    Program prog_;
    const Program* ref_ = nullptr;
    bool instances_ = true;

    std::vector<std::string> errors_;
    int  status_ = RAD_OK;
    bool finished_ = false;

    std::string scope_;

    /* Parallel to prog_.ops. Empty until the ABI carries buffer operands. */
    struct OpBufUse { std::vector<rad_buf> reads, writes; };
    std::vector<OpBufUse> op_bufs_;

    /* Buffers the architecture said it touches from the second lane. See rad_buf_concurrent:
     * their declared op range does not bound when their bytes are in use, so liveness gives them
     * the whole program instead. */
    std::set<rad_buf> concurrent_bufs_;

    /* Parallel to prog_.kv_groups: where in the op list the group was declared, and which op was
     * found to consume it. Declaration order is the binding, because nothing else links an op to
     * a group -- see find_kv_consumer(). */
    std::vector<int32_t> kv_decl_at_;
    std::vector<int32_t> kv_consumer_;

    /* Declared name -> the index in prog_.name_map of its FIRST entry, so a repeated declaration
     * can be checked against it in constant time. A linear scan would be quadratic in the weight
     * count, and a routed model declares tens of thousands of them. */
    std::unordered_map<std::string, size_t> name_map_first_;

    /* Declared weight name -> its handle, for the same reason: Flash-Next declares about a hundred
     * thousand weights a rank, and a scan of the ones before each made declare take seconds. */
    std::unordered_map<std::string, rad_weight> weight_by_name_;

    /* The model's answers, asked once a source: a routed model declares each expert's codes and
     * scales as two weights over one source, and a checkpoint lookup is a name-map walk. */
    WeightSourceFn sources_;
    std::unordered_map<std::string, std::pair<int, WeightSource>> source_cache_;
    int lookup_source(const std::string& name, WeightSource* out);

    BufPlanReport bufrep_;
};

/* ------------------------------------------------------------------ --debug-graph (spec §16) */
/* The entire declared graph: every op in declared order, its bucket table with one line per band,
 * the kernel and variant that resolved each band and which plugin it came from, the constraints
 * that matched, the weights it touches and the buffer it writes; then the weight manifest, the
 * buffer plan, the KV groups, the tuning requests and the miss report.
 *
 * This is the answer to "it is unclear what even runs", so it is a first-class artifact and not a
 * debug print: the bucket-table block is byte-for-byte the format in spec §2.2. */
std::string dump_graph_str(const Program& p, const std::vector<std::string>* errors = nullptr);

/* The same walk, as JSON, for the server's Model view. Grouped by (op, band table): a 64-layer
 * model declares one op sixty-four times with one bucket table, and sixty-four identical rows are
 * not more information than one row saying x64. */
std::string dump_graph_json(const Program& p);
void        dump_graph    (const Program& p, FILE* out,
                           const std::vector<std::string>* errors = nullptr);

}  /* namespace rad */

/* ------------------------------------------------------------------ PROPOSED ABI ADDITION */
/* These two belong in include/rad/rad_builder.h beside rad_decl_op, and they are declared here
 * only because that header is frozen and this one is not. They are purely additive: no existing
 * signature changes and no existing plugin breaks.
 *
 * Without them, an op names its weight operands and not its buffer operands, so the declare phase
 * cannot compute activation liveness and the arena is the sum of every transient rather than the
 * peak of them -- on a 64-layer model with a per-layer residual and a per-layer FFN intermediate
 * that is the difference between one buffer and a hundred and twenty-eight. The declaration reads:
 *
 *     rad_op h = RAD_OP(b, "gemm_nt", RAD_PARAMS(...), RAD_WEIGHTS(w));
 *     rad_op_reads (b, h, RAD_BUFS(x));
 *     rad_op_writes(b, h, RAD_BUFS(y));
 *
 * Separate calls rather than more arguments to rad_decl_op, so the common declaration stays one
 * line and the buffer plan is opt-in per op. */
#ifdef __cplusplus
extern "C" {
#endif

int rad_op_reads (RadBuilder* b, rad_op h, const rad_buf* bufs, int n);
int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n);

#ifndef RAD_BUFS
#define RAD_BUFS(...)  ((const rad_buf[]){ __VA_ARGS__ }), \
                       (int)(sizeof((const rad_buf[]){ __VA_ARGS__ }) / sizeof(rad_buf))
#endif

#ifdef __cplusplus
}   /* extern "C" */
#endif
