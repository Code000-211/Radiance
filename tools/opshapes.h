/* opshapes.h -- the extent of operand i of an op, given its resolved geometry.
 *
 * A schema names operands and roles but not extents, which the run phase supplies at issue. Tools
 * that call an op cold (rad-kbench, rad-tune) need them to size buffers. The kernel's
 * RadKernelInfo::opd_shape hook is asked first (rad_operand_shapes_hook); the fallback recipes
 * (rad_operand_shapes) cover the docs/OPS.md ops whose shapes follow from their parameters. Ops covered by neither are
 * skipped and counted by the caller, never guessed: an invented extent tests bounds handling, not
 * correctness.
 *
 * Shared by rad-kbench and rad-tune so a tuning case and its correctness check use the same shape.
 */
#pragma once
#include "iface.h"

#include <functional>
#include <string>
#include <vector>

namespace rad {

struct OpdShape {
    uint32_t             dtype = RAD_F32;
    std::vector<int64_t> shape;
    bool                 is_index = false;   /* token ids / row indices, not normals */
    /* Index range when it is not the previous operand's leading extent: a KV slot indexes
     * n_blocks * block_size positions, a block-table entry n_blocks. 0 keeps the default. */
    int64_t              idx_max = 0;
    /* A constant for an index operand that is a length, not a choice (`seqused`): a random
     * context shorter than the query would make causal attention read a nonexistent prefix. -1
     * keeps the draw. */
    int64_t              idx_const = -1;
    /* Distinct indices for a scatter (`slot_mapping`): two tokens on one slot race, and ref and
     * the kernel resolve the race differently. Read-side indices may repeat. */
    bool                 idx_unique = false;
    /* Cumulative sequence lengths [0, ..., T] (`cu` of the GDN ops): the batch shape, not a draw. */
    bool                 idx_cu = false;
    /* A routing histogram's offsets [0, ..., idx_max - 1] (RAD_OPD_F_IDX_PREFIX). */
    bool                 idx_prefix = false;
    /* A token history whose rows repeat themselves (RAD_OPD_F_IDX_HISTORY). */
    bool                 idx_history = false;
    /* A list in block order (RAD_OPD_F_IDX_ORDERED): `[rows, width]`, row r holding `start + j` at
     * slot j. `qsa_select` emits a selection sorted by block and the ops that read it refuse
     * anything else, so uniform ids are not a case at all. */
    bool                 idx_ordered = false;
    /* Value domain of a non-index operand when normals fall outside it. `beta` is a sigmoid
     * output in (0, 1); `g` is an intra-chunk cumulative sum of log-decays, non-increasing and
     * non-positive, so every GDN decay factor e^{g_i - g_j} (j <= i) is <= 1. With normals the
     * oracle overflows. fill_chunk is the chunk width where the cumsum resets (RAD_GDN_CHUNK).
     * FillSampleParams writes sampler rows (RAD_FILL_SAMPLE_PARAMS); for it fill_chunk is the
     * history width and idx_max the bitmask's row count. */
    enum Fill { FillNormal = 0, FillSigmoid, FillGateCumsum, FillTriLower, FillUnitRow,
                FillSampleParams };
    int                  fill = FillNormal;
    int64_t              fill_chunk = 0;
    /* An optional operand this case does not pass; it is a null RadTensor (rad_abi.h), and the
     * caller allocates and compares nothing. Recipes still emit one entry per schema operand, so
     * the count check catches schema drift. Used for `quant_act_i8`'s third output. */
    bool                 absent = false;
};

/* Model facts no op parameter carries. The GDN ops read head counts off the operands, so the caller
 * passes them from the container metadata; without them the gdn_* recipes decline (skipped and
 * counted). */
struct ShapeCtx {
    int64_t n_head_k = 0;   /* linear_num_key_heads */
    int64_t n_head_v = 0;   /* linear_num_value_heads */
    int64_t n_seq    = 1;   /* sequences in the checked batch */
};

/* ================================================================== the hook, which wins
 *
 * Ask the kernel first via `RadKernelInfo::opd_shape`; rad_operand_shapes is the fallback for
 * plugins without the hook. The caller chooses which kernel to ask:
 *
 *   - a correctness check asks the oracle, so both implementations get the same bytes under the
 *     reference semantics;
 *   - a benchmark asks the kernel under test, since a device-only fusion has no oracle.
 *
 * Returns RAD_E_UNSUPPORTED if the kernel has no hook or declines the geometry; the caller skips
 * and counts the case. */
int rad_operand_shapes_hook(const RadKernelInfo* k, const Geometry& g, int n_opd,
                            std::vector<OpdShape>* out);

/* Fills one entry per schema operand, in order. RAD_E_UNSUPPORTED when this op has no recipe or
 * the geometry does not carry the parameters the recipe needs. `M` is the value of the ranged
 * parameter for the band being sized. */
int rad_operand_shapes(const std::string& op, const Geometry& g, int n_opd, int64_t M,
                       std::vector<OpdShape>* out, const ShapeCtx* ctx = nullptr);

inline int64_t rad_numel(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
}

/* Values are materialised in f32 for filling and comparison; narrow outputs are widened first, so
 * a last-bit bf16 difference falls within tolerance. */
void rad_widen (const void* src, uint32_t dtype, int64_t n, std::vector<float>* out);
void rad_narrow(const float* src, uint32_t dtype, int64_t n, void* dst);

/* splitmix64, keyed per case so one case's inputs do not depend on how many ran before it. */
inline uint64_t rad_splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t rad_case_seed(uint64_t base, std::string_view op, int band, int domain, int operand);
void     rad_fill_normal(float* p, int64_t n, uint64_t seed, float sigma);
/* The block width is part of the contract: a caller streaming an operand block by block must use
 * the width rad_fill_normal splits at, or the same seed gives different values. */
extern const int64_t RAD_FILL_BLOCK;
void     rad_fill_normal_block(float* p, int64_t n, uint64_t seed, int64_t block, float sigma);

/* Runs fn(b) for each block b in [0, nb), round-robin over the hardware threads. A block's result
 * must not depend on which thread runs it. */
void rad_for_blocks(int64_t nb, const std::function<void(int64_t)>& fn);

}  /* namespace rad */
