/* kfixture.h -- a recorded reference to check kernel libraries against without a model.
 *
 * Checking against a real container needs a 27B model file and runs libref (straight host loops)
 * at model shapes, which takes tens of minutes. Instead the oracle runs once when a fixture is
 * recorded, and the result is checked in. Libraries are then checked in seconds on any machine,
 * and the fixture doubles as a regression test of libref itself (a live comparison would move
 * both sides together).
 *
 * Full outputs at model shapes would be tens of gigabytes, so per output operand it stores:
 *
 *   - the sum of squares over every element, and the count: the rel_l2 denominator, and the check
 *     a sample cannot make (a correct first tile followed by zeros);
 *   - the number of non-finite elements (a NaN against a number is a failure, checked first);
 *   - the values at KF_SAMPLES seed-derived positions. 1024 samples catch an error rate of 1e-3
 *     with probability 64% and 1e-2 with probability > 1 - 3e-5;
 *   - from version 5, KF_SKETCH seeded random-sign projections of the whole output. Their
 *     difference estimates the full-tensor rel_l2 to about +-25%, so a localised error the samples
 *     miss and the norm cannot see (one permuted or sign-flipped tile of a large output) fails.
 *
 * Inputs are not stored: both sides draw them from the stored seed with the same code.
 */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* Positions sampled per output operand. Fixed because it is baked into every recorded file. */
constexpr int KF_SAMPLES = 1024;

/* Block size of the threaded full-tensor reduction. Part of the contract, like RAD_FILL_BLOCK:
 * partial sums are combined in block order, so the result is independent of thread count. */
constexpr int64_t KF_REDUCE_BLOCK = 1 << 18;

/* Projections per output in a version-5 recording. 32 gives the rel_l2 estimate a relative spread
 * of sqrt(2/32), about 25%. Fixed because it is baked into every recorded file. */
constexpr int KF_SKETCH = 32;

/* The format version this tree writes. Versions 4 to 6 still load: version 4 lacks the sketch, the
 * prefix fill law and drawn weight tables (KfOperand::undrawn), version 5 the sampler laws (the
 * request-row fill and the token history), version 6 the block-order index law. */
constexpr uint32_t KF_VERSION = 7;

/* The fill laws, mirroring OpdShape::Fill. Separate because these values are on disk and must not
 * be renumbered. */
enum {
    KF_FILL_NORMAL = 0,
    KF_FILL_SIGMOID = 1,
    KF_FILL_GATE_CUMSUM = 2,
    KF_FILL_TRI_LOWER = 3,
    KF_FILL_UNIT_ROW = 4,
    KF_FILL_SAMPLE_PARAMS = 5,
};

/* A geometry key/value, stored typed, so loading needs no type-guessing parser of
 * `Geometry::str()`. */
struct KfParam {
    std::string key;
    int         kind = 0;        /* RAD_P_INT | RAD_P_F64 | RAD_P_STR */
    long long   i = 0;
    double      d = 0;
    std::string s;
};

struct KfOperand {
    int      role = 0;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    bool     absent = false;

    /* How the input was drawn (copied from OpdShape at record time), so replay reproduces the
     * exact bytes; it cannot be recovered from the shape. */
    int      fill = 0;
    bool     is_index = false, idx_unique = false, idx_cu = false, idx_prefix = false;
    bool     idx_history = false, idx_ordered = false;
    int64_t  idx_max = 0, idx_const = -1, fill_chunk = 0;
    /* A weight table (RAD_OPD_WTAB) in a version-4 recording, which drew it as zeros. Its
     * reference was computed on zeros, so the replay draws zeros too; such a case checks nothing
     * and is reported as vacuous until the fixture is recorded again. */
    bool     undrawn = false;

    /* A weight operand is planes of an encoding (spec §4.3), which kernels lay out themselves.
     * Recorded: the encoding, the selected plane indices, and each plane's extents in its own
     * elements. The planes are drawn from the operand's seed; the reference reads what they decode
     * to (kf_planes_logical). Absent for other operands and for hand-written test weights, which
     * are drawn as plain tensors.
     *
     * A weight table (RAD_OPD_WTAB) records one expert's encoding and selection: its shape is
     * [n_expert, ...] and each expert's planes are drawn from its own seed (kf_draw_table). */
    bool        has_enc = false;
    RadEncoding enc{};
    std::vector<int>     sel;
    std::vector<int64_t> sel_rows, sel_cols;

    /* The reference, for OUT and INOUT operands of a case that has one. */
    bool     have_ref = false;
    int64_t  ref_n = 0;
    double   ref_sumsq = 0;
    int64_t  ref_nonfinite = 0;
    std::vector<float> sample;   /* KF_SAMPLES values at seed-derived positions */
    std::vector<double> sketch;  /* KF_SKETCH projections (kf_sketch); empty in version 4 */
};

struct KfCase {
    std::string op;
    std::vector<KfParam>   params;
    std::vector<KfOperand> opd;
    uint64_t    seed = 0;
    /* The ranged parameter's value, recorded because its key varies by op ("M", "q_len", ...).
     * The draw needs it, e.g. to divide a token count across sequences. */
    int64_t     m = 1;
    /* The band this point came from, for the report only. */
    std::string band;
};

struct Kfixture {
    /* Version 4 carries each weight operand's encoding and selected planes (KfOperand::enc).
     * Version 5 adds the sketch, the prefix fill law and drawn weight tables, version 6 the
     * sampler laws, version 7 the block-order index law. Earlier files are refused and must be
     * re-recorded. */
    uint32_t    version = KF_VERSION;
    std::string source;        /* the container it was recorded from */
    std::string oracle;        /* the plugin and version that produced the references */
    std::string recorded;      /* an ISO date, so a stale fixture is visible */
    std::vector<KfCase> cases;

    /* load() refuses, as RAD_E_FORMAT, a file it cannot read whole or whose contents cannot
     * describe a case: an unknown role, fill law or index flag, a law newer than the file's
     * version, a dtype the core cannot size, an extent that is not positive or an element count
     * past 2^40, a plane selection without extents, a sample count other than the operand's. A
     * corrupt file must not reach an allocation, and an unknown law must not be drawn as another.
     */
    int load(const std::string& path);
    /* Writes KF_VERSION whatever `version` holds. */
    int save(const std::string& path) const;
};

/* The sampled positions in an operand of `n` elements; derived identically by recorder and reader,
 * so only values are stored. */
void kf_sample_positions(uint64_t seed, int64_t n, std::vector<int64_t>* out);

/* The seed for operand `k` of a case, shared by record and replay. Inputs use k; an output's
 * sample positions and sketch use kf_output_seed. */
uint64_t kf_operand_seed(uint64_t case_seed, int k);
inline uint64_t kf_output_seed(uint64_t case_seed, int k) {
    return kf_operand_seed(case_seed, 0x5000 + k);
}

/* ================================================================== the weight a kernel reads
 *
 * Kernels may want weights arranged differently (libr4d's fp8 GEMMs read WMMA fragment order, its
 * int4 experts reordered nibbles; libref reads rows). Like a container, the fixture carries the
 * encoding's canonical planes; each library lays them out with its own hooks, and the reference
 * reads what the planes decode to. One recording thus serves libraries with different layouts.
 */
/* The operand's encoding and selection, from the declaration it was recorded against. */
void kf_operand_enc(const WeightInfo& w, KfOperand* d);

/* A weight operand's selected planes drawn from `seed`: dense in their own elements, rows starting
 * on a byte, and every value valid for its role (codes inside the codebook, real permutations,
 * exponent scales that neither overflow nor vanish). */
void kf_draw_planes(const KfOperand& d, uint64_t seed, std::vector<std::vector<uint8_t>>* out);

/* What a reference reads for those planes: the operand's dtype and shape (enc_selection_logical).
 * RAD_OK, or the refusal with `why`. */
int kf_planes_logical(const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                      std::vector<uint8_t>* out, std::string* why);

/* The bytes kernel `info` reads for `operand`, laid out from the planes. Returns RAD_OK with `out`
 * and `lay` filled; RAD_E_UNSUPPORTED when the kernel reads the plane as is (the
 * kf_planes_logical bytes, widened if the operand is wider); RAD_E_DTYPE when the kernel does not
 * read this encoding. Any other error is a decline the caller must report. */
int kf_lay_weight(const RadKernelInfo* info, const RadParam* p, int n_p, int operand,
                  const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                  RadLayout* lay, std::vector<uint8_t>* out);

/* Describe a laid-out weight operand as the engine does: stored dtype, declared shape (not the
 * RadLayout's; see the definition). */
void kf_describe_laid(const KfOperand& d, const RadLayout& lay, void* data, RadTensor* out);

/* The seed a weight operand is drawn from: its own identity, not the case's. Also the cache key for
 * drawn bytes (see kf_draw). */
uint64_t kf_weight_seed(const KfOperand& d, size_t k);

/* A weight table's logical bytes, [n_expert, ...] in the operand's dtype: expert e's planes drawn
 * from the table's identity seed and e, and decoded as kf_planes_logical does. RAD_OK, or the
 * refusal with `why`. */
int kf_draw_table(const KfOperand& d, size_t k, std::vector<uint8_t>* out, std::string* why);

/* ================================================================== drawing an input
 *
 * One implementation for recorder and reader, which must produce byte-identical inputs. An index
 * operand's default range is the leading extent of the previous operand, `opd[k - 1]`; the first
 * operand falls back to `vocab_rows`.
 */
/* The fallback index range, derived from the case alone so recorder (which knows the model's
 * vocabulary) and reader (which does not) agree. */
int64_t kf_vocab_rows(const std::vector<KfOperand>& opd);

/* Draws operand `k`'s bytes. Large plain operands are drawn on every core: the stream is cut into
 * RAD_FILL_BLOCK blocks seeded by index, so the bytes do not depend on the thread count. A weight
 * table with an encoding goes through kf_draw_table (and an undecodable one comes back zeroed;
 * call kf_draw_table to learn why). */
void kf_draw(const std::vector<KfOperand>& opd, size_t k, int64_t M, int64_t vocab_rows,
             uint64_t seed, std::vector<uint8_t>* out);

/* Reduce a widened output to the reference's three quantities; used on the oracle's output when
 * recording and the kernel's when replaying. */
void kf_reduce(const float* v, int64_t n, uint64_t seed, double* sumsq, int64_t* nonfinite,
               std::vector<float>* sample);

/* KF_SKETCH projections of the finite elements of `v`, each element's signs derived from `seed`
 * and its index. Summed in double, block by block in block order, so the result is independent of
 * thread count. */
void kf_sketch(const float* v, int64_t n, uint64_t seed, std::vector<double>* out);

}  /* namespace rad */
