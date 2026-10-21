/* rad_arch.h -- the scaffolding an architecture plugin sits on.
 *
 * spec.md §2.4 commits to one plugin per (architecture, quantisation). That is deliberate: a w4a8
 * model wants rmsnorm_had_quant_i8 where a bf16 model wants a plain rmsnorm, and pretending
 * otherwise means a plugin full of branches on a quant descriptor -- which is how you get back to
 * "unclear what actually runs". A declare function with no branches in it is a readable list of
 * what this model does, in order.
 *
 * The file-count mitigation for the quant axis is this library. It is header-only, it is C++
 * (nothing here crosses an .so boundary -- only the four exported C symbols do), and it holds one
 * small struct per transformer block with a declare() and a step(). A plugin is COMPOSITION.
 */
#ifndef RAD_ARCH_H
#define RAD_ARCH_H

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace arch {

/* One thread per tensor-parallel rank, all in one process (spec §1), and declare runs once per
 * rank. So a plugin's declared handles are per-rank state, not process state: a single global
 * would have rank 1's declare overwrite rank 0's. libr4d supports 2/4/8-rank all-reduce, so eight
 * is the bound worth carrying. */
enum { MAX_RANKS = 8 };


/* ================================================================== a debug residual probe
 *
 * THE ONE INSTRUMENT FOR A COMPOSITION FAULT, and it exists because nothing else in the tree can
 * see one: rad-kbench falsifies an op against the oracle on the same inputs and a graph that wires
 * the right ops to the wrong buffers -- or a block whose output is the wrong SIZE -- passes every
 * one of those checks and still serves a different model. What a reader needs then is the residual
 * stream's magnitude down the stack, which is smooth in a working model and steps in a broken one.
 *
 * It SYNCHRONISES THE STREAM and copies one row, so it is off unless RADIANCE_DEBUG_RESID is set,
 * and it prints each (layer, site) once. bf16 buffers only -- every activation in this tree is. */
struct DbgStat { double rms = -1.0, med = -1.0, max = -1.0; };

/* RMS, MEDIAN AND MAX TOGETHER, because a residual stream carries a handful of channels two orders
 * of magnitude above the rest (the "massive activations" every model in this family has) and the
 * RMS is theirs alone. The MEDIAN is the bulk, which is what a block's delta has to be compared
 * against; reading the RMS by itself says twenty layers did nothing when they did. */
inline DbgStat dbg_row_stat(RadCtx* c, rad_buf b, int64_t n) {
    DbgStat s;
    if (!b || n <= 0) return s;
    void* p = rad_buf_ptr(c, b);
    if (!p) return s;
    std::vector<uint16_t> h((size_t)n);
    if (rad_memcpy_async(h.data(), p, n * 2, rad_stream(c)) < 0) return s;
    rad_stream_sync(rad_stream(c));
    std::vector<float> v((size_t)n);
    double a = 0.0, mx = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t)h[(size_t)i] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        a += (double)f * (double)f;
        v[(size_t)i] = std::fabs(f);
        if (v[(size_t)i] > mx) mx = v[(size_t)i];
    }
    std::sort(v.begin(), v.end());
    s.rms = std::sqrt(a / (double)n);
    s.med = v[(size_t)(n / 2)];
    s.max = mx;
    return s;
}

inline bool dbg_resid_on() {
    static const bool on = std::getenv("RADIANCE_DEBUG_RESID") != nullptr;
    return on;
}

inline void dbg_resid(RadCtx* c, int layer, const char* site, int64_t n_embd,
                      rad_buf x, rad_buf delta) {
    if (!dbg_resid_on()) return;
    static int seen[512] = {0};
    const int slot = ((layer * 4) + (site[0] & 3)) & 511;
    if (seen[slot]++) return;
    const DbgStat sx = dbg_row_stat(c, x, n_embd), sd = dbg_row_stat(c, delta, n_embd);
    std::fprintf(stderr, "D resid layer %2d %-4s: x rms %8.4f med %8.5f max %9.3f | "
                         "delta rms %8.4f med %8.5f max %9.3f\n",
                 layer, site, sx.rms, sx.med, sx.max, sd.rms, sd.med, sd.max);
}
/* ================================================================== the GPTQ calibration tap
 *
 * A block that wants to characterise one of its own activations declares an f32 [K, K] PERSIST
 * buffer and a `gram_accum` op into it, then calls calib_write() when the corpus is done. These
 * three helpers are the parts that are not block-specific: whether it is on, when to stop, and
 * the file.
 *
 * PERSIST AND NOT TRANSIENT, and that is the whole reason this is not a debug probe: a transient
 * buffer is liveness-analysed and may share storage with any other transient, so an accumulator
 * declared transient would be overwritten between the launches it is meant to sum.
 *
 * THE CONSUMER IS ANOTHER PROCESS. rad-convert reads these files; the engine writes them. That
 * is not an architectural preference but the only order that works -- GPTQ needs the Hessian of
 * a layer's real input, which needs the model running, and a container is written before it can
 * be served. The bootstrap is: convert with plain absmax, serve THAT, calibrate on it, convert
 * again with the Hessians. A second moment is robust to the small perturbation the first
 * container's own error puts on the activation.
 *
 * PER RANK, AND THE CONSUMER SUMS. Under expert parallelism a rank's `down` input is only the
 * tokens routed to ITS experts, so no single rank holds the layer's distribution. Its `gate_up`
 * input is the whole token set on every rank, so summing those double-counts by `world` -- which
 * is a positive scale on H, and GPTQ is invariant to one (the damping is a fraction of the trace
 * and the update uses ratios). So one rule, sum everything, and the gate_up case is harmless. */
inline const char* calib_dir() {
    static const char* d = std::getenv("RADIANCE_CALIB_DIR");
    return d && *d ? d : nullptr;
}

/* The token count at which the tap dumps and switches itself off. Off by default in the sense
 * that calib_dir() gates it; this only says how much is enough. */
inline long long calib_target() {
    static const long long n = [] {
        const char* s = std::getenv("RADIANCE_CALIB_TOKENS");
        const long long v = s ? std::atoll(s) : 0;
        return v > 0 ? v : 65536LL;
    }();
    return n;
}

/* <dir>/gram.r<rank>.<what>.bin, as a 32-byte header and k*k f32 in row-major order:
 *
 *   char     magic[8] = "RADGRAM2"
 *   uint32_t k          the dimension, on both axes
 *   uint32_t rank       which rank wrote it
 *   uint64_t rows       activation rows accumulated, for provenance only
 *   uint32_t domain     0: the model's own activation; N: that activation rotated by the
 *                       unnormalised Hadamard of width N, block by block -- what a rotated
 *                       weight's kernel multiplies
 *   uint32_t pad
 *
 * `rows` is not a normaliser: nothing downstream divides by it, because H is used only up to a
 * positive scale. It is there so a reader can tell a full calibration from one that was killed
 * early -- and, for one expert's file (CalibExperts), an expert the corpus barely routed to from
 * one it exercised: libquant reads a per-expert file only when it holds several times K rows.
 * `domain` is what the tap actually summed, so a reader quantising in another one -- a rotated
 * recipe over a bf16 model's Hessian -- brings the file there itself (libquant's calib_sum). */
inline bool calib_write_host(const float* h, int64_t k, long long rows, const char* what,
                             uint32_t domain, int rank) {
    const char* dir = calib_dir();
    if (!dir || !h || k <= 0) return false;
    char path[512];
    std::snprintf(path, sizeof path, "%s/gram.r%d.%s.bin", dir, rank, what);
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "radiance: calibration cannot write %s\n", path);
        return false;
    }
    char hdr[32] = {};
    std::memcpy(hdr, "RADGRAM2", 8);
    const uint32_t kk = (uint32_t)k, rk = (uint32_t)rank;
    const uint64_t nr = (uint64_t)rows;
    std::memcpy(hdr + 8, &kk, 4);
    std::memcpy(hdr + 12, &rk, 4);
    std::memcpy(hdr + 16, &nr, 8);
    std::memcpy(hdr + 24, &domain, 4);
    const size_t n = (size_t)(k * k);
    const bool okw = std::fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr &&
                     std::fwrite(h, 4, n, f) == n;
    std::fclose(f);
    if (!okw) std::fprintf(stderr, "radiance: calibration write to %s was short\n", path);
    return okw;
}

/* The same file from a device accumulator: copied back on the block's stream, then written. */
inline bool calib_write(RadCtx* c, rad_buf b, int64_t k, long long rows, const char* what,
                        uint32_t domain) {
    if (!calib_dir() || !b || k <= 0) return false;
    void* p = rad_buf_ptr(c, b);
    if (!p) return false;
    std::vector<float> h((size_t)(k * k));
    if (rad_memcpy_async(h.data(), p, (int64_t)h.size() * 4, rad_stream(c)) < 0) return false;
    rad_stream_sync(rad_stream(c));
    return calib_write_host(h.data(), k, rows, what, domain, rad_rank(c));
}

/* A pinned host buffer of at least `bytes`, one a rank and held for the life of the process, so a
 * tap that brings a scratch back every pass copies it at the link's speed and not a pageable
 * copy's. Null, said once, when the allocation fails. */
inline float* calib_staging(RadCtx* c, int64_t bytes) {
    static std::mutex mu;
    static std::map<int, std::pair<void*, int64_t>> per_rank;
    std::lock_guard<std::mutex> lk(mu);
    std::pair<void*, int64_t>& s = per_rank[rad_rank(c)];
    if (s.second >= bytes) return (float*)s.first;
    if (s.first) rad_dev_free(s.first, RAD_MEM_HOST_PINNED);
    s.first = rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
    s.second = s.first ? bytes : 0;
    if (!s.first)
        std::fprintf(stderr, "radiance: calibration could not pin %lld MiB of host staging on "
                             "rank %d\n", (long long)(bytes >> 20), rad_rank(c));
    return (float*)s.first;
}

/* ONE HESSIAN AN EXPERT, which is what a routed expert's `down` projection needs and the layer's
 * pooled Gram is not. That projection's input is the expert's OWN intermediate units: channel i
 * of one expert has nothing to do with channel i of another, so a Gram summed over the layer's
 * experts describes none of them. It is dominated by whichever expert carries the most energy --
 * on Qwen3.8-Flash-Next one expert holds a quarter of some layers' -- and GPTQ fitted to it spends
 * every other expert's error budget on that expert's channels.
 *
 * ON THE HOST, because the card has no room for it: 640^2 f32 for each of 256 experts a rank is
 * 419 MB a layer, 20 GB over 48. A pass sums each expert's rows into one layer-sized scratch on
 * the card; the scratch comes back through calib_staging() and its upper triangles are added
 * here, packed -- a Gram is symmetric, so half the bytes, 10 GB a rank for that model. The cost is
 * a stream sync a layer, which a calibration run reading its experts from disk does not notice.
 *
 * Accumulated in f32: each pass adds one partial sum an expert, so a 10M-token corpus is a few
 * hundred adds deep, and the consumer damps H by a percent of its trace. */
struct CalibExperts {
    int64_t ne = 0, k = 0;
    std::vector<float>   acc;    /* `ne` upper triangles of k*(k+1)/2, row by row */
    std::vector<int64_t> rows;   /* rows each expert was routed */

    void init(int64_t n_expert, int64_t width) {
        ne = n_expert;
        k = width;
        acc.assign((size_t)(ne * (k * (k + 1) / 2)), 0.0f);
        rows.assign((size_t)ne, 0);
    }

    /* `full` is `ne` row-major k x k Grams, back to back -- the scratch as the card wrote it. */
    void add(const float* full) {
        const int64_t tri = k * (k + 1) / 2;
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned nt = std::max(1u, std::min(8u, hw / 4));
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([this, full, tri, nt, t] {
                for (int64_t e = t; e < ne; e += nt) {
                    const float* s = full + (size_t)(e * k * k);
                    float* d = acc.data() + (size_t)(e * tri);
                    for (int64_t i = 0; i < k; ++i) {
                        const float* sr = s + (size_t)(i * k + i);
                        for (int64_t j = 0; j < k - i; ++j) d[j] += sr[j];
                        d += k - i;
                    }
                }
            });
        for (std::thread& x : th) x.join();
    }

    /* Expert `e`'s Gram, whole and symmetric, into `h` (k x k row-major). */
    void unpack(int64_t e, float* h) const {
        const float* d = acc.data() + (size_t)(e * (k * (k + 1) / 2));
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = i; j < k; ++j, ++d) {
                h[(size_t)(i * k + j)] = *d;
                h[(size_t)(j * k + i)] = *d;
            }
    }

    /* One file an expert, `<stem>.<global expert>`: the weight it will pack minus `.weight`, which
     * is the name libquant looks for before the pooled `<stem>`. Every expert gets one, a
     * never-routed expert's saying `rows` 0. */
    bool write(const char* stem, int64_t expert_base, uint32_t domain, int rank) const {
        std::vector<float> h((size_t)(k * k));
        bool ok = true;
        for (int64_t e = 0; e < ne; ++e) {
            unpack(e, h.data());
            char what[256];
            std::snprintf(what, sizeof what, "%s.%lld", stem, (long long)(expert_base + e));
            ok = calib_write_host(h.data(), k, rows[(size_t)e], what, domain, rank) && ok;
        }
        return ok;
    }
};

/* ================================================================== names */
/* Declared names outlive the call that built them -- the weight manifest, the graph dump and the
 * checkpoint name map all hold the pointer. Interning in a deque keeps every c_str() stable for
 * the life of the plugin, which a vector<string> would not (it reallocates).
 *
 * The scope prefix comes from RadBuildCtx: a drafter or an encoder tower declared into the same
 * builder must not collide with the language model's names (spec §10, §11). */
class Names {
public:
    explicit Names(const char* scope = "")
        : scope_(scope && *scope ? std::string(scope) + "." : std::string()) {}

    /* A declared name, scope-prefixed. */
    const char* f(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[256];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
        pool_.emplace_back(scope_ + buf);
        return pool_.back().c_str();
    }
    /* A checkpoint-side name. Never scope-prefixed: the container says what it says. */
    const char* ckpt(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[256];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
        pool_.emplace_back(buf);
        return pool_.back().c_str();
    }

private:
    std::deque<std::string> pool_;
    std::string             scope_;
};

/* Errors are values. Declare walks a long list and any step of it can refuse; this propagates the
 * first refusal without an early-return ladder swallowing the reason. */
#define RAD_ARCH_TRY(expr)  do { int _s = (expr); if (_s < 0) return _s; } while (0)

/* ================================================================== C++-side decl helpers */
/* rad_group_layer() and friends are C compound literals, which compile in C++ only as a GNU
 * extension -- rad_builder.h says so and points here. A named function also reads better in an
 * initialiser list than a cast-shaped macro does. */
inline RadWeightGroup grp_model()               { RadWeightGroup g{}; g.layer = -1; g.expert = -1; return g; }
inline RadWeightGroup grp_layer(int l)          { RadWeightGroup g{}; g.layer = l;  g.expert = -1; return g; }
inline RadWeightGroup grp_expert(int l, int e)  { RadWeightGroup g{}; g.layer = l;  g.expert = e;  return g; }
/* ...AND THE SLOT, WHICH IS NOT OPTIONAL FOR AN EXPERT WITH MORE THAN ONE TENSOR. The container
 * writes a layer's experts as fixed-stride units and checks that each unit's slots arrive
 * 0..n-1 (core/format/radfile.cpp, check_unit_invariant) -- so two tensors of one expert that
 * both leave `slot` at 0 are refused by name at convert time. A routed fp8 expert has four: the
 * gate_up codes, their scale plane, the down codes and theirs. */
inline RadWeightGroup grp_expert_slot(int l, int e, int s) {
    RadWeightGroup g{}; g.layer = l; g.expert = e; g.slot = s; return g;
}

/* `row_parts` is rows of THIS RANK per stacked projection, see RadWeightDecl. A single part is
 * the same thing as none, so it is not recorded and the load path stays on its plain-rows
 * branch. The pointer form exists because the fp8 linear derives its scale plane's parts by
 * dividing the weight's, which an initializer_list cannot express. */
inline rad_weight decl_w_n(RadBuilder* b, const char* name, uint32_t dtype,
                           std::initializer_list<int64_t> shape,
                           int access, int shard, RadWeightGroup group, int optional,
                           const int64_t* row_parts, int n_row_parts) {
    RadWeightDecl d{};
    d.dtype = dtype;
    d.rank  = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) d.shape[i++] = s;
    d.access   = access;
    d.shard    = shard;
    d.group    = group;
    d.optional = optional;
    if (row_parts && n_row_parts > 1) {
        d.n_row_parts = n_row_parts < RAD_MAX_ROW_PARTS ? n_row_parts : RAD_MAX_ROW_PARTS;
        for (int j = 0; j < d.n_row_parts; ++j) d.row_parts[j] = row_parts[j];
    }
    return rad_decl_weight(b, name, &d);
}

inline rad_weight decl_w(RadBuilder* b, const char* name, uint32_t dtype,
                         std::initializer_list<int64_t> shape,
                         int access, int shard, RadWeightGroup group, int optional = 0,
                         std::initializer_list<int64_t> row_parts = {}) {
    int64_t p[RAD_MAX_ROW_PARTS] = {0};
    int     n = 0;
    for (int64_t v : row_parts) if (n < RAD_MAX_ROW_PARTS) p[n++] = v;
    return decl_w_n(b, name, dtype, shape, access, shard, group, optional, p, n);
}

/* A VIEW: `planes` of the logical weight `source` (null: the weight named `name` itself), declared
 * at what they select -- one plane's own dtype and extents, several planes' codes dtype and the
 * logical extents (RadWeightDecl::source, spec §4.3). A block-fp8 linear is ONE stored weight
 * that gemm_nt_q takes as two operands, so it is declared twice over the same source: its codes
 * as `<base>.weight` and its scales as `<base>.scale`, `source` = `<base>.weight`. Every view of
 * a weight takes the weight's movement group. */
inline rad_weight decl_view(RadBuilder* b, const char* name, const char* source,
                            const char* planes, uint32_t dtype,
                            std::initializer_list<int64_t> shape, int access, int shard,
                            RadWeightGroup group, const int64_t* row_parts = nullptr,
                            int n_row_parts = 0) {
    RadWeightDecl d{};
    d.dtype = dtype;
    d.rank  = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) d.shape[i++] = s;
    d.access = access;
    d.shard  = shard;
    d.group  = group;
    if (row_parts && n_row_parts > 1) {
        d.n_row_parts = n_row_parts < RAD_MAX_ROW_PARTS ? n_row_parts : RAD_MAX_ROW_PARTS;
        for (int j = 0; j < d.n_row_parts; ++j) d.row_parts[j] = row_parts[j];
    }
    d.source = source;
    d.planes = planes;
    return rad_decl_weight(b, name, &d);
}

/* What the model holds `source` as -- its encoding, from the container, the checkpoint, or the
 * recipe rad-convert is applying -- or false when nothing is known about it: a builder with no
 * model behind it, or a name the name map has not been told about yet (declare the map first). */
inline bool weight_enc(RadBuilder* b, const char* source, RadEncoding* out) {
    return rad_weight_encoding(b, source, out, nullptr, nullptr) == RAD_OK;
}

/* The dtype a view of every plane of `source` is declared at: the codes', when the encoding is
 * known, else `dflt` -- the dtype a weight with nothing behind it is read as. */
inline uint32_t weight_codes_dtype(RadBuilder* b, const char* source, uint32_t dflt) {
    RadEncoding e{};
    return weight_enc(b, source, &e) && e.n_planes > 0 ? e.plane[0].dtype : dflt;
}

inline rad_buf decl_b(RadBuilder* b, const char* name, uint32_t dtype,
                      std::initializer_list<int64_t> shape,
                      int kind = RAD_BUF_TRANSIENT, int domain = RAD_DOMAIN_DEVICE) {
    RadBufDecl d{};
    d.dtype = dtype;
    d.rank  = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) d.shape[i++] = s;
    d.kind   = kind;
    d.domain = domain;
    return rad_decl_buffer(b, name, &d);
}

/* Every map starts with src_index all -1: "the whole tensor", which is what all but the stacked
 * expert case wants. Zero would mean sub-tensor 0 of every source, which is a silently wrong map. */
inline RadNameMap map_base(const char* declared, int mode) {
    RadNameMap m{};
    m.declared = declared;
    m.mode     = mode;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    return m;
}

/* A checkpoint tensor copied through unchanged. */
inline int map_copy(RadBuilder* b, const char* declared, const char* src) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src    = 1;
    m.src[0]   = src;
    return rad_decl_name_map(b, &m);
}

/* A checkpoint tensor copied through, with alternatives: the first source the container actually
 * holds wins. This is how a tied lm_head is expressed -- Qwen3 0.6B and 1.7B ship no
 * output.weight and the embedding table serves as both, while 8B and up ship one. Without it the
 * plugin would need two builds of the same architecture. */
inline int map_copy_alt(RadBuilder* b, const char* declared, const char* a, const char* c) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src    = 2;
    m.src[0]   = a;
    m.src[1]   = c;
    return rad_decl_name_map(b, &m);
}

/* One expert's slice of a stacked checkpoint tensor. GGUF stores a MoE layer's experts as one
 * ffn_gate_exps of shape [.., n_expert] while placement needs one declared weight per expert,
 * because the movement unit is an expert and not a tensor (spec §4.1). src_index carries the
 * expert; without it the index would survive only inside the declared name, as a convention
 * rad-convert would have to parse. */
inline int map_slice(RadBuilder* b, const char* declared, const char* src, int index) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src       = 1;
    m.src[0]      = src;
    m.src_index[0]= index;
    return rad_decl_name_map(b, &m);
}

/* Two stacked tensors, the same expert out of each, concatenated: gate and up into gate_up. */
inline int map_slice_concat(RadBuilder* b, const char* declared,
                            std::initializer_list<const char*> src, int index, int dim = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
    m.n_src = (int)src.size();
    int i = 0;
    for (const char* s : src) { m.src[i] = s; m.src_index[i] = index; ++i; }
    m.concat_dim = dim;
    return rad_decl_name_map(b, &m);
}

/* A convert-time fusion: q/k/v into one qkv, gate and up into one gate_up. The plugin is the only
 * component that knows both sides, and rad-convert reads the map from it (spec §2.4). The fusion
 * is worth a comment on its cost: it buys one GEMM launch instead of three and one movement unit
 * instead of three, and it costs the ability to place q, k and v on different tiers, which
 * nothing wants to do. */
inline int map_concat(RadBuilder* b, const char* declared, std::initializer_list<const char*> src,
                      int dim = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
    m.n_src      = (int)src.size();
    int i = 0;
    for (const char* s : src) m.src[i++] = s;
    m.concat_dim = dim;
    return rad_decl_name_map(b, &m);
}

/* ONE LOGICAL TENSOR THE CHECKPOINT STORES AS N SHARDS, named `<prefix><i><suffix>` for i in
 * [0, n). Qwen3.8-Flash-Next's n-gram embedding is 320,001,536 rows in 128 tensors called
 * `...ngram_embedding.shard_0.weight` ... `shard_127.weight`, which is thirty times what
 * RadNameMap's `src[8]` holds.
 *
 * So this emits ceil(n/8) CONCAT maps and the core joins them in call order -- see the merge rule
 * in core/build/rad_builder.cpp for why the pieces are the answer rather than a bigger array. The
 * ORDER IS THE CONCATENATION and shard 0 must come first: a hashed row id addresses the whole
 * table and a permuted table is a model that loads, runs, and looks up the wrong embeddings.
 *
 * Returns the first refusal, or RAD_OK. */
inline int map_concat_shards(RadBuilder* b, const char* declared, const char* prefix,
                             const char* suffix, int n, int dim = 0) {
    if (n <= 0) return RAD_E_INVAL;
    for (int i = 0; i < n; i += 8) {
        RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
        m.concat_dim = dim;
        m.n_src = 0;
        char buf[8][256];
        for (int k = 0; k < 8 && i + k < n; ++k) {
            std::snprintf(buf[k], sizeof buf[k], "%s%d%s", prefix, i + k, suffix);
            m.src[m.n_src++] = buf[k];
        }
        const int rc = rad_decl_name_map(b, &m);
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

/* ================================================================== buffer liveness */
/* An op's WEIGHT operands are named at declare; its BUFFER operands are not, and without them a
 * transient has no declared use and the planner must assume it live for the whole program --
 * sound, because a plugin may issue any handle in any order, and as wasteful as it sounds.
 *
 * So every op this library emits states its read and write sets. Returning the handle rather than
 * a status is what keeps the declaration one statement: a refusal from rad_op_reads is recorded by
 * the builder, with the op named, and surfaces in declare's final report -- the same way a refusal
 * from rad_decl_op does. */
inline rad_op rw(RadBuilder* b, rad_op h,
                 std::initializer_list<rad_buf> reads,
                 std::initializer_list<rad_buf> writes) {
    if (h == RAD_NULL_HANDLE) return h;   /* an unresolved op has no liveness to declare */
    rad_op_reads (b, h, reads.begin(),  (int)reads.size());
    rad_op_writes(b, h, writes.begin(), (int)writes.size());
    return h;
}

/* ================================================================== geometry */
/* Dimensions come from model metadata, so one plugin covers every size in a family. Everything
 * here is ALREADY DIVIDED by world_size where it is sharded: the plugin declares divided
 * dimensions and emits the collectives itself, because a sharding inference pass is a compiler
 * pass whose failure mode is a model that produces fluent wrong text on a rank count nobody
 * tested (spec §9). */
/* THE PAGED BLOCK, IN TOKENS. `attn_paged` declares `block_size` RAD_DERIVED -- a value the
 * KERNEL supplies -- and libr4d's rows all carry C_EQ("block_size", 16), so on the shipped path
 * this is the value that would have been derived anyway and pinning it changes no resolution.
 *
 * It has to be pinned because DERIVED ONLY WORKS WHEN SOMETHING DERIVES IT. A geometry libr4d has
 * no attention kernel for -- Qwen3.5-0.8B is head_dim 256 at gqa 4, and libr4d has 256/6 and
 * 128/4 -- resolves to libref, which serves any block size and therefore names none; the KV
 * group then cannot be sized and declare refuses the model by name. The block size is a property
 * of the CACHE the model is asking for, so the model is the right place to say it. */
enum { RAD_KV_BLOCK = 16 };

struct Geom {
    int64_t n_layer    = 0;
    int64_t n_embd     = 0;    /* the residual stream. Replicated on every rank. */
    int64_t head_dim   = 0;
    int64_t n_vocab    = 0;    /* per rank: lm_head is vocab-sharded (spec §9) */
    int64_t n_vocab_all= 0;
    int64_t vocab_off  = 0;    /* this rank's first vocab row, for the sampler's candidate merge */
    int64_t n_head     = 0;    /* per rank */
    int64_t n_head_kv  = 0;    /* per rank */
    int64_t n_head_all = 0;
    int64_t n_head_kv_all = 0;
    /* THIS RANK'S FIRST KV ROW inside the checkpoint's UNSHARDED k/v projection rows, used only
     * when the KV heads are replicated over ranks instead of split (world > n_head_kv_all, see
     * geom_from). It is in container rows, the same unit a LinearFP8 span takes, so a block that
     * declares a row-sharded k/v passes it straight through as `span_lo`. Zero on every split
     * deployment, which is what keeps TP2 reading the projection exactly as it always did. */
    int64_t kv_off   = 0;
    int64_t n_ff       = 0;    /* per rank */
    int64_t n_ff_all   = 0;
    int64_t max_tok    = 0;
    int64_t max_seqs   = 0;
    int64_t max_ctx    = 0;
    int     max_spec   = 0;
    /* The most rows the sampler can want in one step: one per sequence, or one per draft position
     * per sequence at a speculative verify. It bounds the logits buffer, which at a 248K vocab is
     * the difference between 63 MiB and 8 GiB. */
    int64_t max_logit_rows = 0;
    int     rank       = 0;
    int     world      = 1;
    /* 1 = the exact all-reduce, 0 = this deployment accepts a lossy wire. Straight through to
     * every `all_reduce` this library emits; RadBuildCtx says what it costs and buys. */
    int     wire_exact = 1;
    /* The byte floor under which the lossy wire is refused and the exact kernel serves the
     * message anyway. Straight through to `all_reduce`'s `min_bytes`; 0 leaves the choice to
     * the kernel plugin, which has the measured default. */
    int64_t wire_min_bytes = 0;
    float   eps        = 1e-6f;
    /* GemmaRMSNorm's `1 +`. Qwen3.5, Qwen3-Next and Gemma normalise with `x_hat * (1 + w)` and
     * store `w` around 0.1, not around 1.0 -- so reading the gain as a plain RMSNorm gain shrinks
     * every normed activation by five to ten times, at every norm, in every layer. It is a
     * property of the ARCHITECTURE and not of a block, which is why it lives here: `rmsnorm`,
     * `rmsnorm_add`, `qk_norm_rope_gate` and `rmsnorm_had_quant_i8` all carry a `wadd` parameter
     * for it (libref/ref_registry.cpp), every block that declares one passes this, and a
     * family whose norms are plain leaves it at zero.
     *
     * NOT the gated delta net's own `ssm_norm`: RMSNormGated is a plain RMS norm in this family
     * too, its gains are around 0.88 in the checkpoint, and `gdn_gated_rmsnorm` has no such
     * parameter. Two conventions in one model, which is exactly why this is not a build flag. */
    float   wadd       = 0.0f;
    /* The same question for the per-head q/k norms, separately: they are a different
     * tensor with a different distribution (this checkpoint's are PARTLY NEGATIVE, which
     * only `1 + w` explains) and a family could answer the two differently. */
    float   wadd_qk    = 0.0f;
    float   theta      = 1e6f;
    float   rope_scale = 1.0f;
    /* M-RoPE: the checkpoint's `mrope_section`, when it has one. A model whose rotary has several
     * position components declares `rope` with `rope_mode` and these sections and hands it
     * RadBatch::rope_pos; empty, the rotary is one-component NeoX over `positions`. Stored here
     * rather than pointed at, because the container's strings are not guaranteed to outlive
     * declare. */
    bool    rope_mc    = false;
    char    rope_mode[8]      = "neox";
    char    rope_sections[24] = "";
    const char* dtype  = "bf16";   /* the OPERAND COMBINATION, spec §2.3's dtype key */
    uint32_t act_dtype = RAD_BF16; /* what activation buffers are made of */
    /* THE KV CACHE'S OWN WIDTH, and it is separate from `dtype` because it is a separate decision:
     * `dtype` says what the WEIGHTS are and follows the checkpoint, while this follows the cache
     * and is a deployment choice the checkpoint has no opinion about. They were one field, which
     * read as "the cache is bf16 because the weights are fp8-with-bf16-activations" -- a sentence
     * that is true of this checkpoint and is not a rule. Whatever this says must be the spelling
     * `rad_dtype_name` gives, because the attention and kv_store rows band on it verbatim. */
    const char* kv_dtype = "bf16";
    /* THE LINEARS ARE PLAIN bf16 WEIGHTS, read by `gemm_nt` over the bf16 activation, in a model
     * whose blocks are the fp8 family's (rad_fp8.h). A checkpoint served as it ships -- a dense bf16
     * release, converted without a quantising recipe -- has no scale planes and no use for E4M3
     * activations, so in this mode the producers in front of a linear quantise nothing and declare
     * no codes: an `ActFP8` is its bf16 plane alone. It is the MODEL's answer and not a linear's,
     * because the producer and its consumers have to agree on it; the plugin reads it off a trunk
     * weight's encoding and every linear checks its own against it. */
    bool    w_bf16 = false;

    int64_t q_dim()   const { return n_head * head_dim; }
    int64_t kv_dim()  const { return n_head_kv * head_dim; }
    int64_t qkv_dim() const { return q_dim() + 2 * kv_dim(); }
    int64_t gqa()     const { return n_head_kv ? n_head / n_head_kv : 0; }
};


/* THE KV CACHE'S WIDTH, RESOLVED ONCE FOR EVERY ARCHITECTURE. RadBuildCtx::kv_dtype is
 * --kv-cache-dtype; RAD_DT_INVALID means the operator did not state one and the plugin's own
 * default stands. Every architecture in this tree passes RAD_BF16 as that default.
 *
 * It is one flag on the build context rather than a per-plugin environment switch: a switch read
 * at declare is invisible in the startup log and gives "is this cache fp8" a different answer per
 * architecture. It changes what the model computes, so it is a flag. */
static inline uint32_t rad_kv_cache_dtype(const RadBuildCtx* c, uint32_t dflt) {
    return (c && c->kv_dtype != RAD_DT_INVALID) ? (uint32_t)c->kv_dtype : dflt;
}
/* Divide, refusing rather than rounding. A rank count that does not divide the head count is not
 * a configuration to approximate: it is a configuration to reject at declare, by name, before any
 * weight is mapped. */
inline int divide_or_fail(int64_t v, int world, const char* what, int64_t* out) {
    if (world <= 0 || v % world != 0) {
        fprintf(stderr, "radiance: %s = %lld is not divisible by world_size = %d\n",
                what, (long long)v, world);
        return RAD_E_INVAL;
    }
    *out = v / world;
    return RAD_OK;
}

/* The vocabulary is the one dimension that does not divide -- 151669 and 248320 are not multiples
 * of anything convenient -- and refusing tensor parallel over an awkward vocab size would be
 * absurd. So the last rank holds fewer rows. The sampler merges candidate sets rather than
 * gathering full logits (spec §9, §13), so a ragged shard costs it nothing; a padded one would
 * cost a fake token id that has to be masked everywhere forever. */
/* THE VOCABULARY SPLIT: ceil(total / world) rows a rank, ROUNDED UP TO A WHOLE 128-ROW BLOCK, the
 * last rank short. The block is RAD_FP8_BLOCK: a block-scaled fp8 lm_head is stored in 128x128
 * tiles and a row span that cuts a tile is refused (rad_weight_shard_span), so the cut has to land
 * on one. At two ranks on 248320 or 151936 the rounding changes nothing -- both halves are block
 * multiples -- and a vocabulary the world divides is never rounded; at three it is what makes the
 * head shardable at all (248320 / 3 = 82773.3; the ranks take 82816, 82816, 82688). core/engine.cpp
 * RESTATES this rule and checks it against the width the plugin declared; the two must move together. */
inline int64_t vocab_per_rank(int64_t total, int world) {
    /* A vocabulary the world divides keeps the even share, whatever the world: every container
     * served today splits that way (248320 at 2 and 4, 151936 at 2 -- the last halves mid-block and
     * is served with an untiled head). Only a vocabulary the world does NOT divide takes the
     * block-aligned ceil, which is what makes a tiled head shardable at all. */
    if (total % world == 0) return total / world;
    const int64_t per = (total + world - 1) / world;
    return (per + 127) / 128 * 128;
}
inline int64_t shard_extent(int64_t total, int world, int rank) {
    const int64_t per = vocab_per_rank(total, world);
    const int64_t beg = per * rank;
    const int64_t end = beg + per < total ? beg + per : total;
    return end > beg ? end - beg : 0;
}
inline int64_t shard_begin(int64_t total, int world, int rank) {
    const int64_t per = vocab_per_rank(total, world);
    const int64_t beg = per * rank;
    return beg < total ? beg : total;
}

/* A container-header integer under either of the two spellings a key reaches us in: the GGUF one
 * (`qwen35.ssm.state_size`) and the HuggingFace one (`linear_key_head_dim`). One plugin serves a
 * .rad converted from either source, and the alternative is a second plugin that differs only in
 * its string literals. `found` says whether the value was read or defaulted, because a geometry
 * that silently defaults is a model that silently becomes a different model. */
inline long long meta_int2(const RadModelMeta* m, const char* gguf_key, const char* hf_key,
                           long long dflt, bool* found = nullptr) {
    const long long sentinel = -0x7FFFFFFFll;
    long long v = rad_meta_geti(m, gguf_key, sentinel);
    if (v == sentinel) v = rad_meta_geti(m, hf_key, sentinel);
    if (found) *found = (v != sentinel);
    return v == sentinel ? dflt : v;
}

/* `arch_splits_ff`: the plugin divides the feed-forward width across the ranks ITSELF (qwen4exp:
 * the routed experts' width is cut by whole fp8 blocks, by parity, at any world up to the block
 * count). With it set, a width the world does not divide is left whole in g.n_ff and the plugin
 * takes its slice from there; without it the refusal stands, because for every other plugin g.n_ff
 * IS the per-rank width. A width that does divide is unaffected either way. */
inline int geom_from(Geom& g, const RadModelMeta* m, const RadBuildCtx* c,
                     const char* dtype, uint32_t act_dtype, bool arch_splits_ff = false) {
    if (!m || !c) return RAD_E_INVAL;
    if (c->is_draft) {
        /* A drafter declared into this plugin would need a second set of per-rank handles and a
         * second name-map namespace. Nothing unimplemented returns success (spec §17). */
        fprintf(stderr, "radiance: this plugin has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }
    if (c->rank < 0 || c->rank >= MAX_RANKS) return RAD_E_INVAL;

    g.n_layer   = m->n_layers;
    g.n_embd    = m->n_embd;
    g.head_dim  = m->head_dim ? m->head_dim : (m->n_head ? m->n_embd / m->n_head : 0);
    g.n_head_all    = m->n_head;
    g.n_head_kv_all = m->n_head_kv ? m->n_head_kv : m->n_head;
    g.n_ff_all      = m->n_ff;
    g.n_vocab_all   = m->n_vocab;
    g.max_tok   = c->max_tok;
    g.max_seqs  = c->max_seqs;
    g.max_ctx   = c->max_ctx;
    g.max_spec  = c->max_spec;
    const int64_t spec_factor = c->max_spec > 0 ? (c->max_spec >= 16 ? c->max_spec + 1 : 16) : 1;
    g.max_logit_rows = c->max_seqs * spec_factor;
    if (g.max_logit_rows > c->max_tok) g.max_logit_rows = c->max_tok;
    /* The engine's KL mode reads a row at every prompt position (RadBuildCtx::max_out_rows), which
     * is more rows than tokens: the sampler's own rows come first and repeat a scored one. */
    if (c->max_out_rows > g.max_logit_rows) g.max_logit_rows = c->max_out_rows;
    g.rank      = c->rank;
    g.world     = c->world_size ? c->world_size : 1;
    g.wire_exact= c->tp_wire_lossy ? 0 : 1;
    g.wire_min_bytes = c->tp_wire_min_bytes;
    g.eps       = m->rms_eps;
    g.theta     = m->rope_theta;
    g.rope_scale= m->rope_scale ? m->rope_scale : 1.0f;
    g.dtype     = dtype;
    g.act_dtype = act_dtype;
    {
        const char* sec = rad_meta_gets(m, "rope_parameters.mrope_section",
                                        rad_meta_gets(m, "text_config.rope_parameters.mrope_section",
                                                      ""));
        if (sec && *sec) {
            const bool inter =
                rad_meta_geti(m, "rope_parameters.mrope_interleaved",
                              rad_meta_geti(m, "text_config.rope_parameters.mrope_interleaved",
                                            0)) != 0;
            g.rope_mc = true;
            std::snprintf(g.rope_mode, sizeof g.rope_mode, "%s", inter ? "imrope" : "mrope");
            std::snprintf(g.rope_sections, sizeof g.rope_sections, "%s", sec);
        }
    }

    if (g.n_layer <= 0 || g.n_embd <= 0 || g.head_dim <= 0 || g.n_head_all <= 0 ||
        g.n_ff_all <= 0 || g.n_vocab_all <= 0 || g.max_tok <= 0)
        return RAD_E_INVAL;
    /* eps and theta are read, not defaulted. A container that omits them and gets 1e-6 and 1e6
     * anyway is a container that silently becomes a different model, and the symptom is fluent
     * wrong text. */
    if (!(g.eps > 0.0f) || !(g.theta > 0.0f)) {
        fprintf(stderr, "radiance: container gives rms_eps=%g rope_theta=%g; both are required\n",
                (double)g.eps, (double)g.theta);
        return RAD_E_INVAL;
    }

    int s;
    if ((s = divide_or_fail(g.n_head_all,    g.world, "n_head",    &g.n_head))    < 0) return s;
    /* THE KV HEADS ARE REPLICATED WHEN THERE ARE NOT ENOUGH TO SPLIT. There is one row-span per KV
     * head and no way to hand four ranks half a head, so `divide_or_fail` on n_head_kv is the
     * wrong verb once world exceeds the head count -- the question is not rounding, it is whether
     * the declaration is allowed to say one KV head per rank. It is, when world is a multiple of
     * the KV head count: query heads 0-11 belong to KV head 0 and 12-23 to KV head 1, so ranks
     * {0,1} need only KV head 0 and ranks {2,3} only KV head 1. Each rank then declares
     * n_head_kv 1 and reads its own head's row-span of k/v, which `kv_off` locates; the attention
     * declaration comes out at gqa 6 over a page of 4 instead of gqa 12 over a page of 2, and
     * kv_dim(), gqa() and every block that reads them need no new branch because from here down a
     * replicated rank and a split rank look the same.
     *
     * THE COST IS HONEST AND IT IS MEMORY, NOT MATH: the replicating ranks hold BYTE-IDENTICAL KV
     * weights, a replicated read that costs no extra VRAM per rank versus the TP2 split it
     * replaces (Flash-Next: 2 KV heads x 5120 x 256 rows read twice instead of once). What it buys
     * is that world 4 is declarable at all.
     *
     * When world <= n_head_kv_all the split below is untouched, so a world that divides the KV
     * heads -- TP2 on this family -- declares exactly as it did before replication existed. */
    if (g.world > g.n_head_kv_all) {
        if (g.world % g.n_head_kv_all != 0) {
            fprintf(stderr, "radiance: world_size = %d cannot replicate n_head_kv = %lld; "
                            "replication needs world to be a multiple of the KV head count\n",
                    g.world, (long long)g.n_head_kv_all);
            return RAD_E_INVAL;
        }
        g.n_head_kv = 1;   /* world / n_head_kv_all ranks share each KV head */
        g.kv_off    = (g.rank / (g.world / g.n_head_kv_all)) * g.head_dim;
    } else if ((s = divide_or_fail(g.n_head_kv_all, g.world, "n_head_kv", &g.n_head_kv)) < 0) {
        return s;
    } else {
        g.kv_off = 0;   /* a split rank reads the rows the plain row-shard already gives it */
    }
    if (arch_splits_ff && g.world > 1 && g.n_ff_all % g.world != 0) {
        g.n_ff = g.n_ff_all;   /* the plugin slices it (see the parameter's comment) */
    } else if ((s = divide_or_fail(g.n_ff_all, g.world, "n_ff", &g.n_ff)) < 0) {
        return s;
    }
    /* THE VOCABULARY IS SPLIT. r4d_sample_topk_f32 takes a vocabulary offset and packs its
     * candidates as (u32 GLOBAL id, f32 logit) pairs, and the all-gather places every rank's
     * pairs for one sampled position side by side, which is the layout sample_merge_topk reads.
     * lm_head is therefore ROW-SHARDED and a rank reads half of it: on Qwen3.8-27B that is
     * 1.27 GiB of the 16.1 GiB a rank reads per decode step, which makes the logits GEMM a
     * large single item in the decode budget.
     *
     * The last rank is SHORT rather than padded -- 248320 and 151669 divide by nothing convenient
     * -- because the sampler merges candidate sets rather than gathering full logits, so a ragged
     * shard costs it nothing and a padded one would cost a fake token id masked everywhere
     * forever. core/engine.cpp restates this split and checks its restatement against the width
     * the plugin declared. */
    g.n_vocab   = shard_extent(g.n_vocab_all, g.world, g.rank);
    g.vocab_off = shard_begin (g.n_vocab_all, g.world, g.rank);
    if (g.n_vocab <= 0) return RAD_E_INVAL;
    return RAD_OK;
}

/* ================================================================== a vocab-sharded draft head */
/* WHETHER A DRAFTER'S OWN HEAD CAN BE VOCAB-SHARDED, and it lives here because both the block that
 * declares the op and the plugin that sizes the buffer under it have to give the SAME answer -- a
 * buffer that disagreed with the op declared over it is a read past its end. Every drafter asks
 * the identical question, so the test lives in one place: two copies of a rule this exact are two
 * things to keep in step.
 *
 * IT REFUSES A RAGGED OR UNALIGNED SHARD rather than working around it. The packed w2a8 layout is
 * TILE-MAJOR at 16 output rows a block, so a rank whose share is not a multiple of 16 has a block
 * spanning two ranks and there is no row shard of it at all; and the candidate merge's global ids
 * are `rank * n_vocab + local`, which is only true of a contiguous, equally-sized split. Either
 * one wrong is fluent text with the wrong tokens in it. */
inline bool vocab_head_sharded(const Geom& g) {
    return g.world > 1 && g.n_vocab > 0 && (g.n_vocab % 16) == 0 &&
           g.n_vocab * (int64_t)g.world == g.n_vocab_all &&
           g.vocab_off == (int64_t)g.rank * g.n_vocab;
}
/* Rows of such a head, and therefore the width of its logit plane, on THIS rank. */
inline int64_t vocab_head_rows(const Geom& g) {
    return vocab_head_sharded(g) ? g.n_vocab : g.n_vocab_all;
}

/* ================================================================== the KV cache operand */
/* attn_paged and kv_store take the group's cache as an operand. RadKVGroupBatch carries the slot
 * mapping, the block table and the sequence lengths, but the pool base belongs to the block
 * manager and not to the batch -- so the operand names the GROUP and the LAYER and the core
 * resolves it, the same way it resolves a weight handle. */
inline RadOperand kv_cache(rad_kvgroup group, int layer) { return RAD_KV(group, layer); }

/* ================================================================== step-sized operands */
/* EVERY OPERAND A STEP ISSUES CARRIES THE STEP'S ROW COUNT, NOT THE BUFFER'S.
 *
 * A ranged parameter is collapsed to a BAND when the instance is resolved (spec §2.2) and
 * `rad_issue` carries the actual value as an integer rather than by rewriting the parameter list,
 * so a kernel cannot read the row count out of `M`: it would get the band's upper bound.
 * libref therefore reads every extent off the operands (libref/ref_ops.h), and an
 * operand handed over at its declared extent means a single-row decode computed over all 8192 rows
 * of an arena buffer -- reading uninitialised memory and writing over the next op's input.
 *
 * So the run phase narrows, and these three helpers are how. `rows` is dim 0, which is tokens for
 * every activation buffer in this library; the strides are products of the trailing dimensions, so
 * shortening the outermost one leaves all of them correct (core/runtime/issue.cpp says the same).
 */
inline RadOperand brows(rad_buf h, int64_t rows) {
    RadOperand o = RAD_B(h);
    o.rows = rows;
    return o;
}

/* A column slice narrowed to this step's rows: the two together, because every strided view this
 * library takes is also row-narrowed and spelling it as two calls invites forgetting one. */
inline RadOperand bcol(rad_buf h, int64_t first, int64_t width, int64_t rows) {
    RadOperand o = RAD_B_COL(h, first, width);
    o.rows = rows;
    return o;
}


/* A COLUMN SLICE OF A RANGE OF ROWS. `offset` is a flat element offset and `cols` narrows the last
 * dimension with the declared width kept as the stride, so the two compose: first_row * pitch
 * picks the row, first_col picks the column within it (core/runtime/issue.cpp says the same).
 *
 * The mixed step is what needs it. A block that issues the recurrent path over the decode rows and
 * the chunked path over the prefill ones is issuing each over a RANGE of tokens, and the operands
 * it hands them are already column views of a fused projection. */
inline RadOperand bcol_at(rad_buf h, int64_t first_row, int64_t pitch,
                          int64_t first_col, int64_t width, int64_t rows) {
    RadOperand o = RAD_B(h);
    o.offset = first_row * pitch + first_col;
    o.rows   = rows;
    o.cols   = width;
    return o;
}

/* A RANGE OF ROWS, offset and count. The element offset is `first * pitch` because a buffer's
 * rows are its declared width apart and RadOperand::offset is in elements; dim 0 is then narrowed
 * to `n`, which is what makes the slice a tensor the kernel sees as [n, pitch].
 *
 * It exists because one op can cap the whole step. A libr4d kernel is constrained to a band, and
 * an op whose widest kernel stops at M=64 forces every op in the step down to 64 tokens -- on this
 * model that is the delta net's 96-column a|b projection, 0.5% of the layer's projection bytes,
 * holding the prefill chunk of a 27B model to 64. Issuing that one op in slices costs a launch per
 * 64 rows and lets everything else run at the chunk the scheduler wanted. */
inline RadOperand brow_slice(rad_buf h, int64_t first, int64_t n, int64_t pitch) {
    RadOperand o = RAD_B(h);
    o.offset = first * pitch;
    o.rows   = n;
    return o;
}

/* A batch field: a pointer the core did not allocate, so it has no declared shape and the row
 * count has to come from here. Without it the operand arrives rank 0 and a kernel that sizes its
 * loop by numel(positions) runs one token. */
/* A dense two-dimensional batch field: [rows, cols], packed. The block table is the one that
 * needs it -- see RAD_P_T2 in rad_runtime.h for why a rank-1 spelling is not enough. */
inline RadOperand praw2(const void* p, uint32_t dtype, int64_t rows, int64_t cols) {
    return RAD_P_T2(p, dtype, rows, cols);
}

inline RadOperand praw(const void* p, uint32_t dtype, int64_t rows) {
    RadOperand o = RAD_P_T(p, dtype);
    o.rows = rows;
    return o;
}

/* THE ROTARY POSITIONS OF A PASS, in the two forms its consumers take.
 *
 * `rope_pos1` is one component a token: RadBatch::rope_pos's plane 0 when the batch carries the
 * planes, `positions` when it does not. It is what a cos/sin table is filled and read at, and it
 * is exact for every token of a pass whose components agree (RadBatch::rope_mixed 0) -- so a
 * fused path that reads it is taken only then.
 *
 * `rope_posmc` is what a multi-component `rope` takes: the dense [3, T] planes for an M-RoPE model
 * with a batch that carries them, and the one-component form otherwise -- which every mode reads
 * as NeoX, so a model without M-RoPE and a text-only deployment rotate over `positions` alone. */
inline RadOperand rope_pos1(const RadBatch* b, int64_t T) {
    return praw(b->rope_pos ? b->rope_pos : b->positions, RAD_I32, T);
}
inline RadOperand rope_posmc(const Geom& g, const RadBatch* b, int64_t T) {
    if (g.rope_mc && b->rope_pos) return praw2(b->rope_pos, RAD_I32, 3, T);
    return praw(b->positions, RAD_I32, T);
}
/* Does a pass have a token whose rotary components differ -- which a one-component fused path
 * cannot rotate. */
inline bool rope_mixed(const RadBatch* b) { return b->rope_pos && b->rope_mixed; }

/* `rope` declared for the geometry's rotary: M-RoPE's mode and sections when the model has them,
 * `mode` otherwise. */
inline rad_op decl_rope(RadBuilder* b, const Geom& g, int64_t rows, int64_t head_dim,
                        int64_t n_head, int64_t n_head_kv, int64_t rot, const char* mode) {
    if (g.rope_mc)
        return RAD_OP(b, "rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", head_dim),
                                 RAD_INT("n_head", n_head), RAD_INT("n_head_kv", n_head_kv),
                                 RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                 RAD_STR("mode", g.rope_mode), RAD_INT("rotary_dim", rot),
                                 RAD_STR("sections", g.rope_sections)),
                      RAD_NOWEIGHTS);
    return RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", head_dim),
                             RAD_INT("n_head", n_head), RAD_INT("n_head_kv", n_head_kv),
                             RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                             RAD_STR("mode", mode), RAD_INT("rotary_dim", rot)),
                  RAD_NOWEIGHTS);
}


/* HOW MANY OF THIS STEP'S SEQUENCES ARE DECODE ROWS, AND WHERE THEIR TOKENS END.
 *
 * A block that issues one path over the decode rows and another over the prefill chunks needs the
 * boundary, and for a MIXED step RadBatch::n_seq_decode is the only thing that carries it. For a
 * pure step it is not: PREFILL means none of them and DECODE means all of them, by definition of
 * the phase, and asking the phase first is what makes the two fields optional there. A batch that
 * sets phase and leaves them zero -- a test, a bench harness, an out-of-tree caller -- would
 * otherwise be handed the prefill path by a block that read them alone, with nothing to say so.
 * tests/arch_test.cpp covers exactly that case.
 *
 * The core sets both fields on every step including the pure ones, so this agrees with them; it
 * is the fallback that matters, not the answer. */
inline void batch_split(const RadBatch* b, int64_t* n_seq_dec, int64_t* n_tok_dec) {
    switch (b->phase) {
        case RAD_PHASE_PREFILL: *n_seq_dec = 0;              *n_tok_dec = 0;              break;
        case RAD_PHASE_DECODE:  *n_seq_dec = b->n_seq;       *n_tok_dec = b->n_tok;       break;
        default:                *n_seq_dec = b->n_seq_decode; *n_tok_dec = b->n_tok_decode; break;
    }
}

/* The step batch's entry for one declared KV group. The batch is built once per step by the core
 * and shared by every layer and every group (spec §8); this is the lookup, not a computation. */
inline const RadKVGroupBatch* kv_batch(const RadBatch* b, rad_kvgroup group) {
    for (int i = 0; i < b->n_kv_groups; ++i)
        if (b->kv[i].group == group) return &b->kv[i];
    return nullptr;
}

/* ================================================================== the plugin exports */
/* The four C symbols spec §2.4 names, wired to a namespace. Defining them through a macro keeps
 * the plugin source free of extern "C" boilerplate, and RAD_ARCH_NO_EXPORTS lets the unit test
 * include two plugin sources into one translation unit without the exports colliding -- which is
 * what makes the declared graph testable with no GPU and no model file. */
/* QUANT is the container's quantisation descriptor this plugin serves, "" for an unquantised one.
 * It is the second half of the selection key (spec §2.4, rad_builder.h): qwen35_bf16 and
 * qwen35_fp8 both answer "qwen35" to rad_arch_id and differ only here. */
#define RAD_ARCH_PLUGIN(NS, ID, QUANT, VERSION, DESC)                                          \
    extern "C" const char* rad_arch_id(void) { return ID; }                                    \
    extern "C" const char* rad_arch_quant(void) { return QUANT; }                              \
    extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m,                      \
                                    const RadBuildCtx* c) { return NS::declare(b, m, c); }     \
    extern "C" void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {            \
        NS::step(c, batch);                                                                    \
    }                                                                                          \
    extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }               \
    extern "C" const RadPluginInfo* rad_plugin_info(void) {                                    \
        /* The PLUGIN's name, not the architecture id: rad_abi.h spells the convention out       \
         * ("qwen3_dense_bf16"), the registry requires plugin names to be unique, and a family    \
         * with one plugin per weight format has several plugins per architecture. Passing ID     \
         * here would make qwen35_bf16 and qwen35_fp8 both "qwen35", and the second would be      \
         * refused at load. */                                                                    \
        static const RadPluginInfo info = { RAD_PLUGIN_ARCH, #NS, VERSION, DESC, "" };         \
        return &info;                                                                          \
    }


/* THE FIFTH EXPORT, AND IT IS OPT-IN. A plugin that has nothing to say before declare does not
 * define it; the core reads the absence as "no opinion" and every field of RadArchProbe keeps its
 * zero. Separate from RAD_ARCH_PLUGIN rather than folded into it so that adding a question to
 * RadArchProbe never makes an existing plugin fail to compile. */
#define RAD_ARCH_PROBE(NS)                                                                     \
    extern "C" int rad_arch_probe(const RadModelMeta* m, RadArchProbe* out) {                  \
        return NS::probe(m, out);                                                              \
    }

/* THE REPLY FORMAT, ALSO OPT-IN. A plugin whose model spells its reasoning and tool calls in a way
 * the chat template's analysis does not find -- or finds wrongly -- states it here once, and the
 * reply parser, the lazy call grammar and its trigger words all follow from it. NS::chat_format
 * returns a pointer to static storage with struct_size = sizeof(RadChatFormat), or null to let
 * the template decide for this particular model. */
#define RAD_ARCH_CHAT_FORMAT(NS)                                                               \
    extern "C" const RadChatFormat* rad_arch_chat_format(const RadModelMeta* m) {              \
        return NS::chat_format(m);                                                             \
    }

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_ARCH_H */
