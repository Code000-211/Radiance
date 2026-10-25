/* kfixture.cpp -- reading and writing the recorded reference (see kfixture.h).
 *
 * Binary and little-endian, since the payload is float samples and text would be lossy or about
 * 3x larger. Versioned; a reader refuses any version it does not know.
 */
#include "kfixture.h"
#include "opshapes.h"
#include "format/encoding.h"
#include "rad_plugin.h"
#include "rad_sample.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace rad {
namespace {

const char KF_MAGIC[4] = { 'R', 'K', 'B', '1' };

/* Primitive writer and its mirrored reader; save() and load() must stay field-for-field in step. */
struct W {
    FILE* f;
    bool  ok = true;
    void raw(const void* p, size_t n) { if (ok && std::fwrite(p, 1, n, f) != n) ok = false; }
    void u8(uint8_t v)   { raw(&v, 1); }
    void u32(uint32_t v) { raw(&v, 4); }
    void i64(int64_t v)  { raw(&v, 8); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f64(double v)   { raw(&v, 8); }
    void str(const std::string& s) { u32((uint32_t)s.size()); raw(s.data(), s.size()); }
    void f32v(const std::vector<float>& v) {
        u32((uint32_t)v.size()); raw(v.data(), v.size() * sizeof(float));
    }
    void i64v(const std::vector<int64_t>& v) {
        u32((uint32_t)v.size()); raw(v.data(), v.size() * sizeof(int64_t));
    }
};

struct R {
    FILE* f;
    bool  ok = true;
    void raw(void* p, size_t n) { if (ok && std::fread(p, 1, n, f) != n) ok = false; }
    uint8_t  u8()  { uint8_t v = 0;  raw(&v, 1); return v; }
    uint32_t u32() { uint32_t v = 0; raw(&v, 4); return v; }
    int64_t  i64() { int64_t v = 0;  raw(&v, 8); return v; }
    uint64_t u64() { uint64_t v = 0; raw(&v, 8); return v; }
    double   f64() { double v = 0;   raw(&v, 8); return v; }
    std::string str() {
        const uint32_t n = u32();
        /* Bound lengths read from a possibly corrupt file before allocating. */
        if (!ok || n > (1u << 20)) { ok = false; return {}; }
        std::string s(n, '\0');
        raw(s.data(), n);
        return s;
    }
    std::vector<float> f32v() {
        const uint32_t n = u32();
        if (!ok || n > (1u << 24)) { ok = false; return {}; }
        std::vector<float> v(n);
        raw(v.data(), (size_t)n * sizeof(float));
        return v;
    }
    std::vector<int64_t> i64v() {
        const uint32_t n = u32();
        if (!ok || n > RAD_MAX_RANK) { ok = false; return {}; }
        std::vector<int64_t> v(n);
        raw(v.data(), (size_t)n * sizeof(int64_t));
        return v;
    }
};

/* The largest element count an operand may describe: past every real tensor (the largest is a
 * 51.2 GB n-gram table, ~2^35 bf16 elements), well short of an overflow in any size product. */
constexpr int64_t KF_MAX_NUMEL = (int64_t)1 << 40;

/* Whether a loaded operand can describe a case (Kfixture::load). Absent operands carry nothing to
 * check. */
bool operand_valid(const KfOperand& o) {
    if (o.role < RAD_OPD_IN || o.role > RAD_OPD_WTAB) return false;
    if (o.absent) return true;
    if (rad_dtype_bits(o.dtype) <= 0) return false;
    if (o.shape.empty() || o.shape.size() > (size_t)RAD_MAX_RANK) return false;
    int64_t n = 1;
    for (int64_t x : o.shape) {
        if (x <= 0 || x > KF_MAX_NUMEL || n > KF_MAX_NUMEL / x) return false;
        n *= x;
    }
    if (o.fill < KF_FILL_NORMAL || o.fill > KF_FILL_SAMPLE_PARAMS) return false;
    if (o.idx_max < 0 || o.fill_chunk < 0 || o.idx_const < -1) return false;
    if (o.has_enc) {
        if (o.sel.empty() || o.sel.size() != o.sel_rows.size() ||
            o.sel.size() != o.sel_cols.size())
            return false;
        for (size_t k = 0; k < o.sel.size(); ++k)
            if (o.sel_rows[k] <= 0 || o.sel_cols[k] <= 0 || o.sel_rows[k] > KF_MAX_NUMEL ||
                o.sel_cols[k] > KF_MAX_NUMEL / o.sel_rows[k])
                return false;
    }
    if (o.have_ref) {
        if (o.ref_n != n || o.ref_nonfinite < 0 || o.ref_nonfinite > n) return false;
        if ((int64_t)o.sample.size() != std::min<int64_t>(n, KF_SAMPLES)) return false;
    }
    return true;
}

}  /* namespace */

uint64_t kf_operand_seed(uint64_t case_seed, int k) {
    uint64_t x = case_seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(k + 1));
    return rad_splitmix(x);
}

void kf_sample_positions(uint64_t seed, int64_t n, std::vector<int64_t>* out) {
    out->clear();
    if (n <= 0) return;
    /* Operands no larger than the sample count are sampled exhaustively. */
    if (n <= KF_SAMPLES) {
        out->reserve((size_t)n);
        for (int64_t i = 0; i < n; ++i) out->push_back(i);
        return;
    }
    out->reserve(KF_SAMPLES);
    uint64_t s = seed;
    for (int i = 0; i < KF_SAMPLES; ++i)
        out->push_back((int64_t)(rad_splitmix(s) % (uint64_t)n));
}

int Kfixture::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return RAD_E_IO;
    W w{ f };
    w.raw(KF_MAGIC, 4);
    w.u32(KF_VERSION);
    w.u32((uint32_t)KF_SAMPLES);
    w.str(source);
    w.str(oracle);
    w.str(recorded);
    w.u32((uint32_t)cases.size());
    for (const KfCase& c : cases) {
        w.str(c.op);
        w.u64(c.seed);
        w.i64(c.m);
        w.str(c.band);
        w.u32((uint32_t)c.params.size());
        for (const KfParam& p : c.params) {
            w.str(p.key);
            w.u32((uint32_t)p.kind);
            w.u64((uint64_t)p.i);
            w.f64(p.d);
            w.str(p.s);
        }
        w.u32((uint32_t)c.opd.size());
        for (const KfOperand& o : c.opd) {
            w.u32((uint32_t)o.role);
            w.u32(o.dtype);
            w.i64v(o.shape);
            w.u8(o.absent ? 1 : 0);
            w.u32((uint32_t)o.fill);
            w.u8((uint8_t)((o.is_index ? 1 : 0) | (o.idx_unique ? 2 : 0) | (o.idx_cu ? 4 : 0) |
                           (o.idx_prefix ? 8 : 0) | (o.idx_history ? 16 : 0) |
                           (o.idx_ordered ? 32 : 0)));
            w.i64(o.idx_max);
            w.i64(o.idx_const);
            w.i64(o.fill_chunk);
            w.u8(o.has_enc ? 1 : 0);
            if (o.has_enc) {
                w.raw(&o.enc, sizeof o.enc);
                w.u32((uint32_t)o.sel.size());
                for (size_t k = 0; k < o.sel.size(); ++k) {
                    w.u32((uint32_t)o.sel[k]);
                    w.i64(o.sel_rows[k]);
                    w.i64(o.sel_cols[k]);
                }
            }
            w.u8(o.have_ref ? 1 : 0);
            if (o.have_ref) {
                w.i64(o.ref_n);
                w.f64(o.ref_sumsq);
                w.i64(o.ref_nonfinite);
                w.f32v(o.sample);
                w.u32((uint32_t)o.sketch.size());
                w.raw(o.sketch.data(), o.sketch.size() * sizeof(double));
            }
        }
    }
    const bool ok = w.ok;
    std::fclose(f);
    return ok ? RAD_OK : RAD_E_IO;
}

int Kfixture::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return RAD_E_NOTFOUND;
    R r{ f };
    char magic[4];
    r.raw(magic, 4);
    if (!r.ok || std::memcmp(magic, KF_MAGIC, 4) != 0) { std::fclose(f); return RAD_E_FORMAT; }
    version = r.u32();
    const uint32_t samples = r.u32();
    /* Sample positions are derived from the count, so a file with a different count cannot be
     * read. */
    if (version < 4 || version > KF_VERSION || samples != (uint32_t)KF_SAMPLES) {
        std::fclose(f);
        return RAD_E_FORMAT;
    }
    const bool v4 = version == 4;
    /* The sampler laws are version 6's; an older file naming one is corrupt. */
    const bool sampler_laws = version >= 6;
    const bool ordered_law = version >= 7;
    source   = r.str();
    oracle   = r.str();
    recorded = r.str();
    const uint32_t n = r.u32();
    if (!r.ok || n > (1u << 20)) { std::fclose(f); return RAD_E_FORMAT; }
    cases.clear();
    cases.reserve(n);
    for (uint32_t ci = 0; ci < n && r.ok; ++ci) {
        KfCase c;
        c.op   = r.str();
        c.seed = r.u64();
        c.m    = r.i64();
        c.band = r.str();
        if (v4) (void)r.u8();     /* a flag no recording set */
        const uint32_t np = r.u32();
        if (!r.ok || np > 256) { r.ok = false; break; }
        for (uint32_t i = 0; i < np && r.ok; ++i) {
            KfParam p;
            p.key  = r.str();
            p.kind = (int)r.u32();
            p.i    = (long long)r.u64();
            p.d    = r.f64();
            p.s    = r.str();
            c.params.push_back(std::move(p));
        }
        const uint32_t no = r.u32();
        if (!r.ok || no > 64) { r.ok = false; break; }
        for (uint32_t i = 0; i < no && r.ok; ++i) {
            KfOperand o;
            o.role   = (int)r.u32();
            o.dtype  = r.u32();
            o.shape  = r.i64v();
            o.absent = r.u8() != 0;
            o.fill   = (int)r.u32();
            const uint8_t flags = r.u8();
            o.is_index    = (flags & 1) != 0;
            o.idx_unique  = (flags & 2) != 0;
            o.idx_cu      = (flags & 4) != 0;
            o.idx_prefix  = (flags & 8) != 0;
            o.idx_history = (flags & 16) != 0;
            o.idx_ordered = (flags & 32) != 0;
            /* A flag this reader does not know is a law it would draw as another. */
            if ((flags & ~0x3Fu) != 0) r.ok = false;
            if (!sampler_laws && (o.idx_history || o.fill == KF_FILL_SAMPLE_PARAMS)) r.ok = false;
            if (!ordered_law && o.idx_ordered) r.ok = false;
            o.idx_max    = r.i64();
            o.idx_const  = r.i64();
            o.fill_chunk = r.i64();
            o.has_enc    = r.u8() != 0;
            if (o.has_enc) {
                r.raw(&o.enc, sizeof o.enc);
                const uint32_t ns = r.u32();
                if (!r.ok || ns == 0 || ns > RAD_ENC_MAX_PLANES || !rad_enc_valid(&o.enc)) {
                    r.ok = false;
                    break;
                }
                for (uint32_t k = 0; k < ns; ++k) {
                    const uint32_t ix = r.u32();
                    if (ix >= (uint32_t)o.enc.n_planes) r.ok = false;
                    o.sel.push_back((int)ix);
                    o.sel_rows.push_back(r.i64());
                    o.sel_cols.push_back(r.i64());
                }
            }
            o.have_ref   = r.u8() != 0;
            if (o.have_ref) {
                o.ref_n         = r.i64();
                o.ref_sumsq     = r.f64();
                o.ref_nonfinite = r.i64();
                o.sample        = r.f32v();
                if (!v4) {
                    const uint32_t nk = r.u32();
                    if (!r.ok || (nk != 0 && nk != (uint32_t)KF_SKETCH)) { r.ok = false; break; }
                    o.sketch.resize(nk);
                    r.raw(o.sketch.data(), (size_t)nk * sizeof(double));
                }
            }
            /* Version 4 drew no weight table: its references were computed on zeros. */
            if (v4 && o.role == RAD_OPD_WTAB) o.undrawn = true;
            if (r.ok && !operand_valid(o)) r.ok = false;
            c.opd.push_back(std::move(o));
        }
        cases.push_back(std::move(c));
    }
    const bool ok = r.ok;
    std::fclose(f);
    return ok ? RAD_OK : RAD_E_FORMAT;
}

void kf_reduce(const float* v, int64_t n, uint64_t seed, double* sumsq, int64_t* nonfinite,
               std::vector<float>* sample) {
    /* Threaded (outputs reach tens of millions of elements, which would dominate a replay's CPU
     * time). Per-block partial sums are added in block order, so the result does not depend on the
     * thread count and recorded references are machine-independent. */
    const int64_t nb = (n + KF_REDUCE_BLOCK - 1) / KF_REDUCE_BLOCK;
    std::vector<double> part((size_t)std::max<int64_t>(nb, 1), 0.0);
    std::vector<int64_t> partnf((size_t)std::max<int64_t>(nb, 1), 0);
    auto do_block = [&](int64_t b) {
        const int64_t lo = b * KF_REDUCE_BLOCK;
        const int64_t hi = std::min<int64_t>(lo + KF_REDUCE_BLOCK, n);
        double ss = 0;
        int64_t nf = 0;
        for (int64_t i = lo; i < hi; ++i) {
            if (!std::isfinite(v[i])) { ++nf; continue; }
            ss += (double)v[i] * (double)v[i];
        }
        part[(size_t)b] = ss;
        partnf[(size_t)b] = nf;
    };
    rad_for_blocks(nb, do_block);
    double ss = 0;
    int64_t nf = 0;
    for (int64_t b = 0; b < nb; ++b) { ss += part[(size_t)b]; nf += partnf[(size_t)b]; }
    *sumsq = ss;
    *nonfinite = nf;

    std::vector<int64_t> pos;
    kf_sample_positions(seed, n, &pos);
    sample->clear();
    sample->reserve(pos.size());
    for (int64_t p : pos) sample->push_back(v[p]);
}

void kf_sketch(const float* v, int64_t n, uint64_t seed, std::vector<double>* out) {
    out->assign(KF_SKETCH, 0.0);
    if (n <= 0) return;
    const int64_t nb = (n + KF_REDUCE_BLOCK - 1) / KF_REDUCE_BLOCK;
    std::vector<double> part((size_t)nb * KF_SKETCH, 0.0);
    const uint64_t key = seed ^ 0xD6E8FEB86659FD93ull;
    rad_for_blocks(nb, [&](int64_t b) {
        const int64_t lo = b * KF_REDUCE_BLOCK;
        const int64_t hi = std::min<int64_t>(lo + KF_REDUCE_BLOCK, n);
        double acc[KF_SKETCH] = {};
        for (int64_t i = lo; i < hi; ++i) {
            if (!std::isfinite(v[i])) continue;
            /* Element i's signs: the low KF_SKETCH bits of a hash of (seed, i). Stateless, so a
             * block needs nothing from the blocks before it. */
            uint64_t z = key + (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            const double x = (double)v[i];
            for (int j = 0; j < KF_SKETCH; ++j) acc[j] += ((z >> j) & 1u) ? -x : x;
        }
        for (int j = 0; j < KF_SKETCH; ++j) part[(size_t)b * KF_SKETCH + (size_t)j] = acc[j];
    });
    for (int64_t b = 0; b < nb; ++b)
        for (int j = 0; j < KF_SKETCH; ++j)
            (*out)[(size_t)j] += part[(size_t)b * KF_SKETCH + (size_t)j];
}

namespace {

/* A uniform in [0, 1) from 24 bits, as rad_sample_uniform01 forms one, and an integer in [0, n).
 * Arithmetic only: a libm call could round differently on the machine that replays. */
float unit24(uint64_t& s) { return (float)(rad_splitmix(s) >> 40) * (1.0f / 16777216.0f); }
int64_t below(uint64_t& s, int64_t n) {
    return n > 0 ? (int64_t)(rad_splitmix(s) % (uint64_t)n) : 0;
}

/* Row r's own stream, so a row's draw depends on neither the rows before it nor the row count. */
uint64_t row_stream(uint64_t seed, int64_t r) {
    return seed ^ (0xD1B54A32D192ED03ull * (uint64_t)(r + 1));
}

/* RAD_FILL_SAMPLE_PARAMS: `rows` RadSampleParams, each what a request stages (core/sample/
 * sampler.cpp), flags agreeing with fields. Row 0 switches every stage on, so a one-row case does
 * work in each; on the others each stage is on in seven rows of eight and otherwise at its
 * identity, so both paths of every stage run in a batch. One row in eight is greedy and one in
 * sixteen a greedy request asking for logprobs, which the engine stages at top_k 1 without the
 * flag.
 *
 * Every value is in the range a request uses, except XTC's threshold: candidates are drawn as
 * normals, a flat distribution whose largest probability among 1024 is about 0.015, so llama.cpp's
 * 0.1 would never find the two candidates above it that the stage needs. It is drawn in
 * [1e-4, 1e-2), and below 1e-3 on row 0, where it must fire. The window lies inside the history
 * plane (`fill_chunk` wide) and covers at least half of what follows its start; `mask_row` names a
 * row of the bitmask plane (`idx_max` rows). DRY's penalty is its multiplier times
 * base^(L - allowed) for a repeat of L tokens; the base stays below 2, so over the repeats the
 * history law draws a penalty is a logit or two at the median and rarely past 10^3, and an error
 * elsewhere in the plane is not lost in the norm of one huge value. */
void draw_sample_params(const KfOperand& d, uint64_t seed, int64_t rows, uint8_t* out) {
    const int64_t hw = d.fill_chunk > 0 ? d.fill_chunk : 0;
    const int64_t mask_rows = d.idx_max > 0 ? d.idx_max : 0;
    /* What a positive last-n counts in when the op has no history plane: the engine's default. */
    const int64_t span = hw > 0 ? hw : 64;
    for (int64_t r = 0; r < rows; ++r) {
        uint64_t s = row_stream(seed, r);
        const bool all = r == 0;
        auto u = [&] { return unit24(s); };
        auto pick = [&](int64_t n) { return below(s, n); };
        auto on = [&] { return all || pick(8) != 0; };

        RadSampleParams p;
        std::memset(&p, 0, sizeof p);
        p.seed = rad_splitmix(s);
        p.pos = (uint64_t)pick(4096);
        const int64_t kind = all ? 15 : pick(16);
        const bool greedy = kind < 2, top1 = kind == 2;

        /* [0.5, 1) or (1, 1.5]: 1 is the identity, drawn on its own. */
        const float t = u();
        p.temp      = greedy || top1 ? 0.0f : on() ? (t < 0.5f ? 0.5f + t : 2.0f - t) : 1.0f;
        p.top_k     = top1 ? 1 : on() ? (int32_t)(20 + pick(81)) : 0;
        p.top_p     = on() ? 0.8f + 0.19f * u() : 1.0f;
        p.min_p     = on() ? 0.01f + 0.09f * u() : 0.0f;
        p.typical_p = on() ? 0.7f + 0.29f * u() : 1.0f;

        /* Penalties: each of the three at its identity in one row of four, never all three. */
        const bool pen = on();
        p.rep_penalty = 1.0f;
        p.penalty_last_n = 64;
        if (pen) {
            if (all || pick(4)) p.rep_penalty  = 1.0f + 0.3f * (1.0f - u());
            if (all || pick(4)) p.freq_penalty = 0.5f * (1.0f - u());
            if (all || pick(4)) p.pres_penalty = 0.5f * (1.0f - u());
            if (p.rep_penalty == 1.0f && p.freq_penalty == 0.0f && p.pres_penalty == 0.0f)
                p.rep_penalty = 1.1f;
            const int64_t w = all ? 0 : pick(3);
            p.penalty_last_n = w == 0 ? -1 : w == 1 ? 64 : (int32_t)(1 + pick(span));
            p.flags |= RAD_SP_PENALTY;
        } else if (pick(2)) {
            p.penalty_last_n = 0;
        }

        /* DRY. On one row in four a breaker in the window caps the repeat, at no less than the
         * allowed length (the host turns DRY off below it). */
        const bool dry = on();
        p.dry_base = 1.75f;
        p.dry_allowed_length = 2;
        p.dry_penalty_last_n = -1;
        if (dry) {
            p.dry_multiplier = 0.25f + 0.75f * u();
            p.dry_base = 1.5f + 0.5f * u();
            if (!all && pick(4) == 0) p.dry_allowed_length = (int32_t)(1 + pick(3));
            if (!all && pick(2))
                p.dry_penalty_last_n = (int32_t)(8 + pick(span > 8 ? span - 7 : 1));
            if (!all && pick(4) == 0) p.dry_rep_limit = p.dry_allowed_length + (int32_t)pick(7);
            p.flags |= RAD_SP_DRY;
        }

        /* XTC's coin comes up in about seven rows of eight that set it. */
        p.xtc_threshold = 0.1f;
        if (on()) {
            p.xtc_probability = all ? 1.0f : 0.75f + 0.25f * u();
            p.xtc_threshold = (all || pick(2) ? 1e-4f : 1e-3f) * (1.0f + 9.0f * u());
            p.flags |= RAD_SP_XTC;
        }

        /* The engine's window runs to the newest token at the plane's end; half the windows do,
         * the rest end earlier. */
        if (hw > 0 && (pen || dry)) {
            const int64_t off = all || pick(2) ? 0 : pick(hw / 2 + 1);
            const int64_t rem = hw - off;
            p.hist_off = (int32_t)off;
            p.hist_len = (int32_t)(all || pick(2) ? rem : (rem + 1) / 2 + pick(rem / 2 + 1));
        }

        p.mask_row = -1;
        if (mask_rows > 0 && on()) {
            p.mask_row = (int32_t)pick(mask_rows);
            p.flags |= RAD_SP_MASK;
        }

        if (greedy) p.flags |= RAD_SP_GREEDY;
        if (top1 || (!greedy && on())) p.flags |= RAD_SP_LOGPROBS;
        std::memcpy(out + (size_t)r * sizeof p, &p, sizeof p);
    }
}

/* RAD_OPD_F_IDX_HISTORY: `rows` rows of `w` ids in [0, range), each text that repeats itself. A row
 * draws on 16 ids spread over the range, in runs of one to three, and four times in five copies 3
 * to 12 of its own earlier tokens. A fresh id follows every copy, so a repeat is rarely much longer
 * than the copy that made it, which keeps DRY's exponent small; a window ending inside a 64-token
 * row ends inside a repeat about three times in four. The row closes in a loop, its last 2 to 4
 * tokens repeated three more times, so a window reaching its end always ends in a repeat, and one
 * that DRY finds at three earlier positions. */
void draw_history(int64_t rows, int64_t w, int64_t range, uint64_t seed, float* v) {
    for (int64_t r = 0; r < rows; ++r) {
        uint64_t s = row_stream(seed, r);
        int64_t ids[16];
        for (int64_t& x : ids) x = below(s, range);
        float* row = v + r * w;
        int64_t i = 0;
        while (i < w) {
            if (i >= 4 && below(s, 5) < 4) {
                const int64_t src = below(s, i - 2), len = 3 + below(s, 10);
                for (int64_t t = 0; t < len && i < w; ++t, ++i) row[i] = row[src + t];
                if (i < w) row[i++] = (float)ids[below(s, 16)];
            } else {
                const int64_t len = 1 + below(s, 3);
                for (int64_t t = 0; t < len && i < w; ++t, ++i) row[i] = (float)ids[below(s, 16)];
            }
        }
        const int64_t p = 2 + below(s, 3);
        if (w >= 8 * p)
            for (int64_t j = w - 3 * p; j < w; ++j) row[j] = row[j - p];
    }
}

}  /* namespace */

void kf_draw(const std::vector<KfOperand>& opd, size_t k, int64_t M, int64_t vocab_rows,
             uint64_t seed, std::vector<uint8_t>* out) {
    const KfOperand& d = opd[k];
    int64_t n = 1;
    for (int64_t x : d.shape) n *= x;
    out->assign((size_t)rad_dtype_bytes(d.dtype, n), 0);
    if (n <= 0 || d.absent || out->empty()) return;   /* empty: a dtype the core cannot size */
    if (d.role != RAD_OPD_IN && d.role != RAD_OPD_INOUT && d.role != RAD_OPD_WEIGHT &&
        d.role != RAD_OPD_WTAB)
        return;
    if (d.undrawn) return;
    const bool weightish = d.role == RAD_OPD_WEIGHT || d.role == RAD_OPD_WTAB;
    if (d.role == RAD_OPD_WTAB && d.has_enc) {
        std::string why;
        if (kf_draw_table(d, k, out, &why) != RAD_OK) out->assign(out->size(), 0);
        return;
    }

    /* A weight's seed comes from its own identity (position, dtype, extents), not the case, so
     * every point of an M sweep uses the same bytes and the caller can draw them once
     * (kf_weight_seed is the cache key). */
    if (weightish) seed = kf_weight_seed(d, k);

    /* Sampler rows are structs, written whole; a row that does not fit is left zero. */
    if (!weightish && !d.is_index && d.fill == KF_FILL_SAMPLE_PARAMS) {
        draw_sample_params(d, seed, (int64_t)(out->size() / sizeof(RadSampleParams)), out->data());
        return;
    }

    std::vector<float> tmp;
    /* Large plain-normal operands (every weight and ordinary activation) are drawn block by block,
     * each block on its own thread and narrowed into its own byte range, avoiding a full-size
     * float intermediate (713 MB for one fp8 GEMM weight). The bytes do not depend on the thread
     * count: the stream is seeded per RAD_FILL_BLOCK block, and every dtype the draw narrows to
     * packs whole elements into a byte. The shaped fills below read across the whole tensor and
     * are small, so they are not blocked.
     *
     * A weight-role operand may hold indices (e.g. ngram_ids sizes and offsets, which the oracle
     * refuses when <= 0 or negative), so it is drawn as an index, not as normals. Index values
     * pass through the float `tmp`, exact only up to 2^24. */
    const bool plain = (weightish && !d.is_index) || (!d.is_index && d.fill == KF_FILL_NORMAL);
    if (plain) {
        /* Sigma 0.02 matches real checkpoint weights; unit variance would put an fp8 quantiser's
         * amax far from where it sits on real data. */
        const float sigma = weightish ? 0.02f : 1.0f;
        const int64_t nb = (n + RAD_FILL_BLOCK - 1) / RAD_FILL_BLOCK;
        uint8_t* dst = out->data();
        rad_for_blocks(nb, [&](int64_t b) {
            thread_local std::vector<float> blk;
            const int64_t lo = b * RAD_FILL_BLOCK;
            const int64_t len = std::min<int64_t>(RAD_FILL_BLOCK, n - lo);
            blk.resize((size_t)len);
            rad_fill_normal_block(blk.data(), len, seed, b, sigma);
            rad_narrow(blk.data(), d.dtype, len, dst + rad_dtype_bytes(d.dtype, lo));
        });
        return;
    }
    tmp.assign((size_t)n, 0.0f);
    if (d.is_index && d.idx_prefix) {
        /* A routing histogram's offsets [0, ..., T]: non-decreasing, first 0 and last T, so each
         * of the n - 1 buckets holds a valid run. T is idx_max - 1 (idx_max bounds a value
         * exclusively). The runs come from T uniform choices of a bucket. */
        const int64_t buckets = n > 1 ? n - 1 : 1;
        const int64_t T = d.idx_max > 0 ? d.idx_max - 1 : 0;
        std::vector<int64_t> cnt((size_t)buckets, 0);
        uint64_t s2 = seed;
        for (int64_t t = 0; t < T; ++t) ++cnt[(size_t)(rad_splitmix(s2) % (uint64_t)buckets)];
        int64_t acc = 0;
        for (int64_t i = 1; i < n; ++i) {
            acc += cnt[(size_t)(i - 1)];
            tmp[(size_t)i] = (float)acc;
        }
    } else if (d.is_index && d.idx_ordered) {
        /* A list in block order (see OpdShape::idx_ordered): slot j holds `start + j`, so a row
         * ascends strictly. The start walks up with the row, so a tile's rows disagree about which
         * blocks they hold and the mask filters what the union carries for the others -- which is
         * the whole point of the table op. It stops short of idx_max so no entry leaves the block
         * space. */
        const int64_t w = d.shape.empty() ? n : d.shape.back();
        const int64_t rows_n = w > 0 ? n / w : 0;
        const int64_t span = d.idx_max > 0 ? d.idx_max : w;
        for (int64_t r = 0; r < rows_n; ++r) {
            const int64_t start = std::min<int64_t>(r, span > w ? span - w : 0);
            for (int64_t j = 0; j < w; ++j) tmp[(size_t)(r * w + j)] = (float)(start + j);
        }
    } else if (d.is_index && d.idx_cu) {
        /* Cumulative sequence lengths [0 .. T]. GDN kernels read cu[n] before any null test, so
         * a wrong one faults. */
        const int64_t S = n > 1 ? n - 1 : 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)((i * M) / S);
    } else if (d.is_index && d.idx_const >= 0) {
        /* A fixed length (see OpdShape::idx_const). */
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)d.idx_const;
    } else if (d.is_index && d.idx_unique) {
        /* Distinct scatter destinations (see OpdShape::idx_unique): partial Fisher-Yates. */
        uint64_t s2 = seed;
        const int64_t range = d.idx_max > 0 ? d.idx_max : n;
        std::vector<int32_t> pool((size_t)range);
        for (int64_t i = 0; i < range; ++i) pool[(size_t)i] = (int32_t)i;
        for (int64_t i = 0; i < n && i < range; ++i) {
            const int64_t j = i + (int64_t)(rad_splitmix(s2) % (uint64_t)(range - i));
            std::swap(pool[(size_t)i], pool[(size_t)j]);
            tmp[(size_t)i] = (float)pool[(size_t)i];
        }
        for (int64_t i = range; i < n; ++i) tmp[(size_t)i] = -1.0f;
    } else if (d.is_index) {
        /* Range: idx_max if the recipe set it, else the previous operand's leading extent, else
         * vocab_rows. */
        int64_t rows = vocab_rows > 0 ? vocab_rows : 32000;
        if (k > 0 && !opd[k - 1].shape.empty()) rows = opd[k - 1].shape[0];
        if (d.idx_max > 0) rows = d.idx_max;
        if (rows <= 0) rows = 1;
        if (d.idx_history) {
            const int64_t w = d.shape.empty() ? n : d.shape.back();
            draw_history(n / w, w, rows, seed, tmp.data());
        } else {
            uint64_t s2 = seed;
            for (int64_t i = 0; i < n; ++i)
                tmp[(size_t)i] = (float)(rad_splitmix(s2) % (uint64_t)rows);
        }
    } else {
        rad_fill_normal(tmp.data(), n, seed, 1.0f);
        /* Map into the operand's domain where normals fall outside it. The maps run in double and
         * round once, as the normals do: glibc's float routines differ between releases, and a
         * fixture recorded on one host is replayed on another against its exact answers. */
        if (d.fill == KF_FILL_SIGMOID) {
            for (int64_t i = 0; i < n; ++i)
                tmp[(size_t)i] = (float)(1.0 / (1.0 + std::exp(-(double)tmp[(size_t)i])));
        } else if (d.fill == KF_FILL_TRI_LOWER) {
            /* Unit diagonal, small off-diagonal, upper part zero. `A` is the chunked delta rule's
             * I + tril(diag(beta) K K^T, -1), unit lower triangular with a bounded inverse. A
             * normal diagonal would make it ill-conditioned (gdn_kkt_solve outputs ~1e20), and the
             * comparison would measure the condition number, not the kernel. */
            const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : 1;
            const int64_t heads = d.shape.size() >= 3 ? d.shape[1] : 1;
            const int64_t rows = (heads * ch) != 0 ? n / (heads * ch) : n;
            for (int64_t t = 0; t < rows; ++t)
                for (int64_t h = 0; h < heads; ++h)
                    for (int64_t j = 0; j < ch; ++j) {
                        const size_t i = (size_t)((t * heads + h) * ch + j);
                        const int64_t diag = t % ch;
                        if (j > diag)       tmp[i] = 0.0f;
                        else if (j == diag) tmp[i] = 1.0f;
                        else                tmp[i] *= 0.25f;
                    }
        } else if (d.fill == KF_FILL_GATE_CUMSUM) {
            /* Rows are [tokens, heads]; the cumsum runs down tokens and resets at each chunk
             * boundary, as chunk_local_cumsum produces. */
            const int64_t heads = d.shape.size() >= 2 ? d.shape.back() : 1;
            const int64_t rows = heads > 0 ? n / heads : n;
            const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : rows;
            for (int64_t h = 0; h < heads; ++h) {
                float acc = 0.0f;
                for (int64_t t = 0; t < rows; ++t) {
                    if (t % ch == 0) acc = 0.0f;
                    acc -= (float)std::log1p(std::exp((double)tmp[(size_t)(t * heads + h)]));
                    tmp[(size_t)(t * heads + h)] = acc;
                }
            }
        } else if (d.fill == KF_FILL_UNIT_ROW) {
            /* Unit L2 norm along the last dimension, as the model's qk_norm provides. Plain normals
             * over a 128-wide head have norm ~11, making the solve ill-conditioned (outputs
             * ~1e18). */
            const int64_t w = d.shape.empty() ? n : d.shape.back();
            if (w > 0) {
                for (int64_t r = 0; r + w <= n; r += w) {
                    double ss = 0;
                    for (int64_t i = 0; i < w; ++i)
                        ss += (double)tmp[(size_t)(r + i)] * (double)tmp[(size_t)(r + i)];
                    const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
                    for (int64_t i = 0; i < w; ++i) tmp[(size_t)(r + i)] *= inv;
                }
            }
        }
    }
    rad_narrow(tmp.data(), d.dtype, n, out->data());
}

uint64_t kf_weight_seed(const KfOperand& d, size_t k) {
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(k + 1) ^ ((uint64_t)d.dtype << 17);
    for (int64_t x : d.shape) { s ^= (uint64_t)x; s = rad_splitmix(s); }
    return rad_splitmix(s);
}

int kf_draw_table(const KfOperand& d, size_t k, std::vector<uint8_t>* out, std::string* why) {
    out->clear();
    if (!d.has_enc || d.shape.size() < 2 || d.shape[0] <= 0) {
        if (why) *why = "a weight table needs an encoding and a leading expert dimension";
        return RAD_E_SHAPE;
    }
    const int64_t ne = d.shape[0];
    /* One expert: the table's selection over its shape without the leading dimension. */
    KfOperand one = d;
    one.role = RAD_OPD_WEIGHT;
    one.shape.assign(d.shape.begin() + 1, d.shape.end());
    const int64_t per = rad_dtype_bytes(d.dtype, rad_numel(one.shape));
    if (per <= 0) {
        if (why) *why = "a dtype the core cannot size";
        return RAD_E_DTYPE;
    }
    out->assign((size_t)(per * ne), 0);
    const uint64_t base = kf_weight_seed(d, k);
    std::vector<std::vector<uint8_t>> pl;
    std::vector<uint8_t> logical;
    for (int64_t e = 0; e < ne; ++e) {
        uint64_t es = base ^ (0xC2B2AE3D27D4EB4Full * (uint64_t)(e + 1));
        kf_draw_planes(one, rad_splitmix(es), &pl);
        std::string w;
        const int rc = kf_planes_logical(one, pl, &logical, &w);
        if (rc != RAD_OK || (int64_t)logical.size() != per) {
            if (why) *why = rc != RAD_OK ? w : "an expert decodes to a different size than the "
                                               "table holds";
            out->assign((size_t)(per * ne), 0);
            return rc != RAD_OK ? rc : RAD_E_SHAPE;
        }
        std::memcpy(out->data() + per * e, logical.data(), (size_t)per);
    }
    return RAD_OK;
}

void kf_operand_enc(const WeightInfo& w, KfOperand* d) {
    d->has_enc = true;
    d->enc = w.enc;
    d->sel.assign(w.sel, w.sel + w.n_sel);
    d->sel_rows.assign(w.sel_rows, w.sel_rows + w.n_sel);
    d->sel_cols.assign(w.sel_cols, w.sel_cols + w.n_sel);
}

namespace {

/* The entries of the codebook `codes` index, or 0 when they index none. */
int64_t codebook_entries(const RadEncoding& e) {
    if (const RadEncPlane* g = rad_enc_plane(&e, "grid")) return g->extent[0];
    if (const RadEncPlane* t = rad_enc_plane(&e, "table")) return t->extent[1];
    return 0;
}

/* One plane, rows of `rb` bytes. */
void draw_plane(const RadEncoding& e, const RadEncPlane& p, int64_t rows, int64_t cols,
                int64_t rb, uint64_t seed, uint8_t* out) {
    const int64_t n = rows * cols;
    uint64_t s = seed;
    if (!std::strcmp(p.role, "t.perm")) {
        /* a per-row permutation of the columns: Fisher-Yates over the identity */
        for (int64_t r = 0; r < rows; ++r) {
            std::vector<int64_t> v((size_t)cols);
            for (int64_t j = 0; j < cols; ++j) v[(size_t)j] = j;
            for (int64_t j = cols - 1; j > 0; --j)
                std::swap(v[(size_t)j], v[(size_t)(rad_splitmix(s) % (uint64_t)(j + 1))]);
            for (int64_t j = 0; j < cols; ++j)
                rad_store_f32(out + r * rb, p.dtype, j, (float)v[(size_t)j]);
        }
        return;
    }
    int sgn = 0;
    const int pb = rad_dtype_packed_int(p.dtype, &sgn);
    const bool ints = pb || p.dtype == RAD_I8 || p.dtype == RAD_U8 || p.dtype == RAD_I16 ||
                      p.dtype == RAD_U16 || p.dtype == RAD_I32 || p.dtype == RAD_U32;
    const bool codes = !std::strcmp(p.role, "codes");
    const bool minifloat = p.dtype == RAD_FP4E2M1 || p.dtype == RAD_FP6E2M3 ||
                           p.dtype == RAD_FP6E3M2;
    if (ints || minifloat) {
        /* Every bit pattern is a value, so draw uniformly over patterns, except codebook codes
         * (kept inside the codebook) and wide integers (capped at 12 bits so a reference's sums
         * stay exact in f32). The sub-byte floats have no infinity or NaN, so their patterns are
         * stored as drawn: rad_store_f32 does not encode them. */
        const int bits = rad_dtype_bits(p.dtype);
        uint64_t range = bits >= 16 ? 1u << 12 : 1ull << bits;
        const int64_t entries = codes ? codebook_entries(e) : 0;
        if (entries > 0 && (uint64_t)entries < range) range = (uint64_t)entries;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < cols; ++j) {
                uint32_t c = (uint32_t)(rad_splitmix(s) % range);
                if (pb || minifloat) { rad_store_bits(out + r * rb, bits, j, c); continue; }
                const bool neg = (p.dtype == RAD_I8 || p.dtype == RAD_I16 || p.dtype == RAD_I32);
                const float v = neg ? (float)((int64_t)c - (int64_t)(range / 2)) : (float)c;
                rad_store_f32(out + r * rb, p.dtype, j, v);
            }
        return;
    }
    if (p.dtype == RAD_E8M0) {
        /* 2^-4 .. 2^4; a draw over every exponent would overflow */
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < cols; ++j)
                out[r * rb + j] = (uint8_t)(123 + rad_splitmix(s) % 9);
        return;
    }
    /* Float plane: normal draw; scale planes get a positive magnitude, since no quantiser writes a
     * negative scale. Rows are stored independently, so they go to threads in chunks. */
    std::vector<float> f((size_t)n);
    rad_fill_normal(f.data(), n, seed, 1.0f);
    const bool mag = rad_enc_role_in_chain(p.role, "scale") || !std::strcmp(p.role, "t.scale");
    const int64_t chunk = std::max<int64_t>(1, RAD_FILL_BLOCK / std::max<int64_t>(cols, 1));
    rad_for_blocks((rows + chunk - 1) / chunk, [&](int64_t b) {
        const int64_t r1 = std::min<int64_t>(rows, (b + 1) * chunk);
        for (int64_t r = b * chunk; r < r1; ++r)
            for (int64_t j = 0; j < cols; ++j) {
                float v = f[(size_t)(r * cols + j)];
                if (mag) v = 0.25f + std::fabs(v);
                rad_store_f32(out + r * rb, p.dtype, j, v);
            }
    });
}

}  /* namespace */

void kf_draw_planes(const KfOperand& d, uint64_t seed, std::vector<std::vector<uint8_t>>* out) {
    out->assign(d.sel.size(), {});
    for (size_t k = 0; k < d.sel.size(); ++k) {
        const RadEncPlane& p = d.enc.plane[d.sel[k]];
        const int64_t rows = d.sel_rows[k], cols = d.sel_cols[k];
        const int64_t rb = rad_enc_row_bytes(p.dtype, cols);
        std::vector<uint8_t>& b = (*out)[k];
        b.assign((size_t)(rows * rb), 0);
        uint64_t ps = seed ^ (0xA24BAED4963EE407ull * (uint64_t)(d.sel[k] + 1));
        draw_plane(d.enc, p, rows, cols, rb, rad_splitmix(ps), b.data());
    }
}

int kf_planes_logical(const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                      std::vector<uint8_t>* out, std::string* why) {
    const int64_t n = rad_numel(d.shape);
    out->assign((size_t)rad_dtype_bytes(d.dtype, n), 0);
    const void* data[RAD_ENC_MAX_PLANES] = {};
    for (size_t k = 0; k < planes.size() && k < RAD_ENC_MAX_PLANES; ++k)
        data[k] = planes[k].data();
    return enc_selection_logical(d.enc, d.sel.data(), (int)d.sel.size(), d.sel_rows.data(),
                                 d.sel_cols.data(), data, d.dtype, n, out->data(), why);
}

int kf_lay_weight(const RadKernelInfo* info, const RadParam* p, int n_p, int operand,
                  const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                  RadLayout* lay, std::vector<uint8_t>* out) {
    if (!info || !info->layout || !d.has_enc) return RAD_E_UNSUPPORTED;
    RadTensor pt[RAD_ENC_MAX_PLANES];
    const int n = (int)d.sel.size();
    for (int k = 0; k < n; ++k) {
        pt[k] = RadTensor{};
        pt[k].data  = const_cast<uint8_t*>(planes[(size_t)k].data());
        pt[k].dtype = d.enc.plane[d.sel[(size_t)k]].dtype;
        pt[k].rank  = 2;
        pt[k].shape[0] = d.sel_rows[(size_t)k];
        pt[k].shape[1] = d.sel_cols[(size_t)k];
        rad_tensor_pack(&pt[k]);
    }
    *lay = RadLayout{};
    const int rc = info->layout(p, n_p, operand, &d.enc, d.sel.data(), pt, n, lay);
    if (rc != RAD_OK) return rc;
    /* A layout without a relayout would fail at load; report it rather than return unwritten
     * bytes. */
    if (lay->bytes <= 0 || !info->relayout) return RAD_E_STATE;
    out->assign((size_t)lay->bytes, 0);
    return info->relayout(p, n_p, operand, &d.enc, d.sel.data(), pt, n, out->data(), lay->bytes);
}

int64_t kf_vocab_rows(const std::vector<KfOperand>& opd) {
    int64_t v = 0;
    for (const KfOperand& d : opd)
        if (!d.shape.empty() && d.shape[0] > v) v = d.shape[0];
    return v;
}

/* Describe a re-laid weight as Ctx::refresh_weights does: stored dtype, declared shape, not the
 * RadLayout's shape. They differ when a layout changes the width: libr4d's fp8 lm_head appends each
 * row's 128-block scales, storing [N, K + 2*sk], while the kernel is still handed [N, K] of E4M3
 * and finds the scales past the end of the row. */
void kf_describe_laid(const KfOperand& d, const RadLayout& lay, void* data, RadTensor* out) {
    *out = RadTensor{};
    out->data  = data;
    out->dtype = lay.dtype;
    out->rank  = (uint32_t)d.shape.size();
    for (size_t i = 0; i < d.shape.size() && i < RAD_MAX_RANK; ++i)
        out->shape[i] = d.shape[i];
    rad_tensor_pack(out);
}

}  /* namespace rad */
