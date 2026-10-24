/* dflash_k_norm_rope_store.cpp -- schema, description, call validation and the fusion oracle for
 * the DFlash2 layer's fused K chain.
 *
 * WHAT THE KERNEL IS, AND WHY IT IS NOT libref's ARITHMETIC.
 *
 * The chain this replaces is the DEVICE chain the engine ran:
 *
 *   libr4d/r4d_model_bf16.hip:440-455  rmsnorm_kernel<wf,THR>   sum of squares over the row,
 *                                        block_sum, rsqrtf, bf16 store
 *   libr4d/r4d_model_bf16.hip:544-570  rope_kernel             NeoX pair, powf, sincosf, bf16
 *   libr4d/r4d_model_bf16.hip:589-700  kv_store_kernel         paged scatter, k then v
 *
 * PARITY STATUS: TARGET, NOT RESULT. The kernel is written to reproduce that chain byte for byte
 * and the argument for each rounding is stated below, but nothing has been built, compiled or run.
 * Equality is established only by the dedicated mine-vs-libr4d comparator (the `replaces` hook
 * cannot do it -- see its comment block); until that passes, every claim here is an inspection.
 *
 * libref differs from that chain in two places, both documented in libref itself and both
 * deliberately NOT reproduced here:
 *
 *   * ref_rmsnorm (libref/ref_elementwise.cpp:27-44) divides `ss / (float)n + eps` and takes
 *     `1.0f / sqrtf(...)`; the device kernel takes `rsqrtf(...)`. Two different last bits.
 *   * ref_rmsnorm folds the sum of squares in ASCENDING index order over one element at a time;
 *     the device kernel folds it over `r4d_act_threads(M, n)` threads with a wave butterfly and
 *     then sums the wave partials in ascending order. A different partition, a different rounding.
 *
 * The device chain wins because it is what produced the KV cache the model actually served from.
 * The fusion oracle names `rmsnorm -> rope -> kv_store` because that is the conventional vocabulary
 * for the same function. Note that stock rad-kbench resolves each step only within the fused row's
 * OWN plugin, so it cannot execute this chain; see the boundary note below and in
 * reports/dflash2-k-store-handoff.md.
 *
 * THE REDUCTION IS INTENDED TO BE REPRODUCIBLE, NOT MERELY CLOSE -- the argument follows, and the
 * measurement does not exist yet. For head_dim 128 with `r4d_act_threads(M, 128)` in
 * {1024, 512, 256} (it returns nothing smaller: libr4d/r4d_common.h:250-254), the reference
 * loop `for (i = threadIdx.x; i < n; i += THR)` gives every thread AT MOST ONE element, because
 * THR >= 256 > 128. A thread with no element contributes a zero to block_sum, and adding 0.0f to a
 * finite f32 accumulator is exact, so the fold is the same for all three widths. The fold that
 * remains is: a wave butterfly over each aligned group of 32 columns (xor 16, 8, 4, 2, 1 -- the same
 * DPP sequence, in the same order, over the same lanes: libr4d/r4d_common.h:129-136), then the wave
 * partials summed in ascending wave index. A 128-thread block has exactly four such groups and its
 * four partials are the reference's first four; the rest are zeros. That is why the kernel uses 128
 * lanes and owns no case for other widths. THIS IS A SOURCE ARGUMENT; the comparator must confirm
 * it byte for byte before anything is called equal.
 *
 * THE TWO BF16 ROUNDINGS STAY SEPARATE. rmsnorm stores bf16 and rope reads it back, so the value
 * that is rotated is the ROUNDED norm -- not the f32 product. The kernel re-derives that rounded
 * value for both ends of a rotary pair rather than keeping f32 across the seam, which is the same
 * thing libr4d/r4d_qk_norm_rope_gate.hip does and the same thing ref_qk_norm_rope documents.
 *
 * NO COS/SIN BUFFER. `powf` and `sincosf` are called at rope_kernel's own arguments, so there is
 * no table to fill, no lifetime to own and no second definition of the angles.
 *
 * THE SLOT RULE IS THE DEVICE STORE'S. A negative slot stores nothing, and so does a slot at or
 * past the last block (libr4d/r4d_model_bf16.hip:591). libref's ref_kv_store instead REFUSES an
 * out-of-range slot up front (ref_attn.cpp:335-337) -- it can, because it iterates host memory;
 * this op cannot, because the mapping is device memory and scanning it would need a copy and a
 * synchronisation the launch path forbids. The kernel skips exactly as the row it replaces does.
 * The norm and the rotation still run for that token: they are separate ops upstream and their
 * effect on `k` exists whatever the slot says.
 */
#include "dflash_k_norm_rope_store.h"

#include <cstdint>
#include <cstring>

namespace {

/* The one geometry the device row is compiled for. `head_dim` is 128 because the DFlash2 block's
 * K norm is declared at g.head_dim and this candidate is scoped to it; a second width needs the
 * reduction argument above re-derived, not a template argument bolted on. */
constexpr int64_t kHeadDim = 128;

int64_t span_elements(const RadTensor& t) {
    int64_t e = 1;
    for (uint32_t i = 0; i < t.rank; ++i) e += (t.shape[i] - 1) * t.stride[i];
    return e;
}

/* Do two tensors share any byte? Both are bf16 here, but the span is measured in ELEMENTS and
 * widened by the operand's own width so a call mixing widths is still caught. */
bool overlaps(const RadTensor& a, const RadTensor& b) {
    if (!a.data || !b.data) return false;
    const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
    const uint64_t xa = (uint64_t)span_elements(a) * 2u, xb = (uint64_t)span_elements(b) * 2u;
    return x < y + xb && y < x + xa;
}

/* A row-major rank-2 plane whose rows are `width` wide at `pitch`, checked the way rmsnorm and
 * kv_store check the operand they were given: unit column step, a pitch that cannot make rows
 * overlap, and extents that cannot overflow the address arithmetic. */
int check_plane(const RadTensor& t, int64_t width, int64_t rows, int64_t* pitch) {
    if (t.rank != 2) return RAD_E_UNSUPPORTED;
    if (t.shape[0] != rows || t.shape[1] != width) return RAD_E_SHAPE;
    if (t.stride[1] != 1) return RAD_E_STRIDE;
    if (t.stride[0] < 0) return RAD_E_SHAPE;
    if (rows > 1) {
        if (t.stride[0] < width) return RAD_E_SHAPE;
        if (rows > INT64_MAX / t.stride[0]) return RAD_E_SHAPE;
    }
    if (span_elements(t) <= 0) return RAD_E_SHAPE;
    *pitch = rows > 1 ? t.stride[0] : (t.stride[0] > 0 ? t.stride[0] : width);
    return RAD_OK;
}

}  /* namespace */

int rad_dflash_k_store_schema_count(void) { return 1; }

/* ---- schema ---------------------------------------------------------------------------------- */
#define P_INT(k)  { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_F64(k)  { k, RAD_P_F64, RAD_REQUIRED, 0 }
#define P_STR(k)  { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(n)     { n, RAD_OPD_IN,  0 }
#define INOUT(n)  { n, RAD_OPD_INOUT, 0 }
#define WGT(n)    { n, RAD_OPD_WEIGHT, 0 }

/* PARAMETER SPELLINGS ARE libref's WHEREVER THE CONCEPT EXISTS, because the loader refuses a plugin
 * whose schema for an op disagrees with the one already registered -- and these names are also what
 * the two steps of the oracle must be given. `wadd` and `scale` are `rmsnorm`'s and `rope`'s
 * spellings; `n_head_kv`, `block_size` and `kv_dtype` are `kv_store`'s. */
static const RadParamSpec kParams[] = {
    P_INT("M"), P_INT("n_head_kv"), P_INT("head_dim"), P_F64("eps"), P_F64("wadd"),
    P_F64("theta"), P_F64("scale"), P_INT("block_size"), P_STR("kv_dtype"),
};
static const RadOperandSpec kOperands[] = {
    INOUT("k"), IN("v"), WGT("w"), IN("positions"), IN("slot_mapping"), INOUT("kv_cache"),
};
static const RadOpSchema kSchema = {
    "dflash_k_norm_rope_store", kParams, (int)(sizeof(kParams) / sizeof(kParams[0])),
    kOperands, (int)(sizeof(kOperands) / sizeof(kOperands[0])),
    "k = bf16(rope_neox(bf16(rmsnorm(k)))); then scatter k and v into "
    "kv_cache[n_blocks, n_head_kv, block_size, 2*head_dim] at each token's slot, K at [.., d] and "
    "V at [.., head_dim + d]. The norm is rounded to bf16 before the rotation and the rotation "
    "again before the store. A negative or out-of-range slot stores nothing; k still leaves normed "
    "and rotated, which is the unfused pair's effect."
};
#undef P_INT
#undef P_F64
#undef P_STR
#undef IN
#undef INOUT
#undef WGT

const RadOpSchema* rad_dflash_k_store_schema_at(int i) { return i == 0 ? &kSchema : nullptr; }

/* ---- operand description --------------------------------------------------------------------- */
/* SIX DESCRIPTORS, ALWAYS, and the count is checked against the schema it just read. `M` is the
 * band ceiling and the REAL row count comes from the operand at launch (rad_arch.h's brows note:
 * a ranged parameter has been collapsed, so a kernel that read rows from `M` would walk an arena).
 *
 * THE CACHE'S BLOCK COUNT is the same rule ref_kv_store_shape states: a scatter whose only written
 * block is block 0 agrees with anything about the block index, so the description leaves two
 * blocks of slack beyond what a dense prefix of `M` slots could need. */
int rad_dflash_k_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= RAD_DKNS_OPERANDS) return RAD_E_INVAL;
    const int64_t M   = rad_param_getdim(p, n_p, "M", 0);
    const int64_t nkv = rad_param_getdim(p, n_p, "n_head_kv", 0);
    const int64_t hd  = rad_param_getdim(p, n_p, "head_dim", 0);
    const int64_t bs  = rad_param_getdim(p, n_p, "block_size", 0);
    if (M <= 0 || nkv <= 0 || hd <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    if (hd > INT64_MAX / nkv) return RAD_E_UNSUPPORTED;

    /* THE DESCRIPTION IS THE GEOMETRY THE LAUNCH SERVES. Describing a head_dim this row refuses,
     * or a cache dtype it refuses, would turn a case this row cannot run into a FAILURE instead of
     * the SKIP it is -- the rule ref_shapes.cpp states for its own hook. */
    if (hd != kHeadDim) return RAD_E_UNSUPPORTED;
    const char* kvd = rad_param_gets(p, n_p, "kv_dtype", "bf16");
    if (!kvd || std::strcmp(kvd, "bf16") != 0) return RAD_E_UNSUPPORTED;

    RadOpdDesc d[6]{};
    for (int i = 0; i < 6; ++i) { d[i].fill = (uint32_t)RAD_FILL_NORMAL; d[i].idx_const = -1; }

    /* k and v: the step's bf16 activations, one row a token, n_head_kv heads of head_dim. */
    d[RAD_DKNS_K].dtype = (uint32_t)RAD_BF16;
    d[RAD_DKNS_K].rank = 2;
    d[RAD_DKNS_K].shape[0] = M;
    d[RAD_DKNS_K].shape[1] = nkv * hd;

    d[RAD_DKNS_V] = d[RAD_DKNS_K];

    /* The gain is f32: rad_block_dflash2.h declares every norm gain RAD_F32, and the kernel reads
     * the width off the operand so a bf16 gain is served too -- the description names the one the
     * graph actually passes. */
    d[RAD_DKNS_W].dtype = (uint32_t)RAD_F32;
    d[RAD_DKNS_W].rank = 1;
    d[RAD_DKNS_W].shape[0] = hd;

    /* Positions are rope's: one component a token, i32, one a row. */
    d[RAD_DKNS_POS].dtype = (uint32_t)RAD_I32;
    d[RAD_DKNS_POS].rank = 1;
    d[RAD_DKNS_POS].shape[0] = M;
    d[RAD_DKNS_POS].fill = (uint32_t)RAD_FILL_INDEX;
    /* THE DRAW BOUND IS NAMED RATHER THAN INHERITED. The ABI's 0 means "the leading extent of the
     * previous operand", and the previous operand here is the GAIN -- head_dim wide. A tool would
     * then draw positions from [0, 128) and exercise one band of the rotation. M is this op's own
     * ceiling and is what a fixture can defend; the real positions are the model's. */
    d[RAD_DKNS_POS].idx_max = M;

    /* THE SLOT IS A SCATTER DESTINATION: distinct, and drawn over the whole cache in SLOTS, which
     * is no operand's dim 0. Two tokens on one slot make the store a race and the two
     * implementations resolve it differently. */
    const int64_t nb = (M + bs - 1) / bs + 2;
    d[RAD_DKNS_SLOT].dtype = (uint32_t)RAD_I32;
    d[RAD_DKNS_SLOT].rank = 1;
    d[RAD_DKNS_SLOT].shape[0] = M;
    d[RAD_DKNS_SLOT].fill = (uint32_t)RAD_FILL_INDEX;
    d[RAD_DKNS_SLOT].idx_max = nb * bs;
    d[RAD_DKNS_SLOT].flags = (uint32_t)RAD_OPD_F_IDX_UNIQUE;

    /* kv_store's paged layout, and only this op's `kv_dtype` reaches it: k and v are bf16 whatever
     * the cache stores (the unfused pair's activation is bf16 too). The dtype itself was checked
     * above, so the descriptor names bf16 rather than re-parsing the string. */
    d[RAD_DKNS_KV].dtype = (uint32_t)RAD_BF16;
    d[RAD_DKNS_KV].rank = 4;
    d[RAD_DKNS_KV].shape[0] = nb;
    d[RAD_DKNS_KV].shape[1] = nkv;
    d[RAD_DKNS_KV].shape[2] = bs;
    d[RAD_DKNS_KV].shape[3] = 2 * hd;

    *out = d[operand];
    return RAD_OK;
}

/* ---- the call ---------------------------------------------------------------------------------- */
int rad_dflash_k_store_args(const RadArgs* a, RadDflashKStoreCall* call) {
    if (!a || !call || !a->t || a->n_t != RAD_DKNS_OPERANDS) return RAD_E_INVAL;
    const RadTensor& k    = a->t[RAD_DKNS_K];
    const RadTensor& v    = a->t[RAD_DKNS_V];
    const RadTensor& w    = a->t[RAD_DKNS_W];
    const RadTensor& pos  = a->t[RAD_DKNS_POS];
    const RadTensor& slot = a->t[RAD_DKNS_SLOT];
    const RadTensor& kv   = a->t[RAD_DKNS_KV];
    if (!k.data || !v.data || !w.data || !pos.data || !slot.data || !kv.data) return RAD_E_INVAL;

    const int64_t nkv = rad_param_getdim(a->p, a->n_p, "n_head_kv", 0);
    const int64_t hd  = rad_param_getdim(a->p, a->n_p, "head_dim", 0);
    const int64_t bs  = rad_param_getdim(a->p, a->n_p, "block_size", 0);
    /* `block_size`, `n_head_kv` and the cache's block count are the values the launch NARROWS to
     * `int` for the argument struct, so they are bounded here rather than left to wrap. The pitches
     * stay int64 all the way to the kernel and are bounded at launch. */
    if (nkv <= 0 || nkv > INT32_MAX || hd <= 0 || bs <= 0 || bs > INT32_MAX) return RAD_E_SHAPE;
    const float eps  = (float)rad_param_getf(a->p, a->n_p, "eps", 1e-6);
    const float wadd = (float)rad_param_getf(a->p, a->n_p, "wadd", 0.0);
    const float theta = (float)rad_param_getf(a->p, a->n_p, "theta", 10000.0);
    const float rscale = (float)rad_param_getf(a->p, a->n_p, "scale", 1.0);
    if (!(theta > 0.0f)) return RAD_E_INVAL;

    const char* kvdt = rad_param_gets(a->p, a->n_p, "kv_dtype", "bf16");
    if (!kvdt || std::strcmp(kvdt, "bf16") != 0) return RAD_E_DTYPE;

    if (k.dtype != RAD_BF16 || v.dtype != RAD_BF16 || kv.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (pos.dtype != RAD_I32 || slot.dtype != RAD_I32) return RAD_E_DTYPE;

    /* THE WIDTH IS THE SPECIALISATION. head_dim 128 is what the row was derived for; anything else
     * returns a reason instead of a differently-partitioned reduction. */
    if (hd != kHeadDim) return RAD_E_UNSUPPORTED;
    if (nkv > INT32_MAX || hd > INT64_MAX / nkv) return RAD_E_SHAPE;
    /* RANK BEFORE ANY EXTENT IS READ. `check_plane` rejects a non-rank-2 operand, but it reads
     * shape[0] to do it; a rank-0 tensor's shape entries are whatever the caller left there. */
    if (k.rank != 2 || v.rank != 2) return RAD_E_UNSUPPORTED;
    const int64_t lane = nkv * hd;
    if (k.shape[0] <= 0 || k.shape[0] > INT32_MAX) return RAD_E_SHAPE;
    const int64_t rows = k.shape[0];

    int64_t k_pitch = 0, v_pitch = 0;
    int rc = check_plane(k, lane, rows, &k_pitch);
    if (rc != RAD_OK) return rc;
    rc = check_plane(v, lane, rows, &v_pitch);
    if (rc != RAD_OK) return rc;
    /* THE NORM'S ROW ENUMERATION IS FLAT: rows*n_head_kv rows of head_dim at one pitch, which is
     * the same bytes as one row a token only when the plane is packed. r4d_rows_pitch agrees for
     * the n_head_kv 1 case -- it returns the plane's own stride[0], any width >= head_dim -- and
     * returns -1 for a pitched plane with more than one head (libr4d/r4d_args.h:171-185), so the
     * unfused `rmsnorm` refuses exactly the call this refuses. For one row the pitch is never used,
     * which is what several callers hand a single-row slice. */
    if (rows > 1 && lane != hd && k.stride[0] != lane) return RAD_E_STRIDE;

    const int64_t band = rad_param_getdim(a->p, a->n_p, "M", 0);
    if (band > 0 && rows > band) return RAD_E_SHAPE;

    /* The gain: head_dim values, dense. It is read one element a lane by index, so the last stride
     * has to be 1 -- a transposed gain would be read as a different vector. */
    if (w.rank != 1 || w.shape[0] < hd || w.stride[0] != 1) return RAD_E_SHAPE;

    /* Position and slot: one i32 a row, dense, and as many rows as the call has tokens. */
    if (pos.rank != 1 || pos.shape[0] != rows || pos.stride[0] != 1) return RAD_E_SHAPE;
    if (slot.rank != 1 || slot.shape[0] != rows || slot.stride[0] != 1) return RAD_E_SHAPE;

    /* The cache, exactly the paged layout the store walks, packed: every inner offset the kernel
     * forms is a product of these extents. */
    if (kv.rank != 4) return RAD_E_SHAPE;
    if (kv.shape[1] != nkv || kv.shape[2] != bs || kv.shape[3] != 2 * hd) return RAD_E_SHAPE;
    if (kv.shape[0] <= 0 || kv.shape[0] > INT32_MAX) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(&kv)) return RAD_E_STRIDE;

    /* ALIASING. `k` is written in place, so a call that handed the cache or the v plane as `k`
     * would overwrite its own input mid-launch -- and the two would then disagree about every
     * byte after the first head. The pair it replaces has the same three buffers distinct. */
    if (overlaps(k, kv) || overlaps(v, kv) || overlaps(k, v)) return RAD_E_INVAL;

    call->k = k.data;
    call->v = v.data;
    call->w = w.data;
    call->pos = (const int32_t*)pos.data;
    call->slot = (const int32_t*)slot.data;
    call->kv = kv.data;
    call->rows = rows;
    call->head_dim = hd;
    call->n_head_kv = nkv;
    call->block_size = bs;
    call->n_blocks = kv.shape[0];
    call->k_pitch = k_pitch;
    call->v_pitch = v_pitch;
    call->w_f32 = w.dtype == RAD_F32;
    call->eps = eps;
    call->wadd = wadd;
    call->theta = theta;
    call->rscale = rscale;
    return RAD_OK;
}

/* ---- the fusion oracle ------------------------------------------------------------------------
 *
 * THREE STEPS, EACH AT ITS OWN GEOMETRY -- and the geometry is where this hook earns its keep:
 * `rmsnorm` runs at M = rows * n_head_kv over a width of head_dim (the norm is per head, not per
 * token), while `rope` and `kv_store` run at M = rows over the fused kv row. Nothing in the fused
 * op's parameter list is that product, which is why step 0 cannot mirror `M` as r4d_fuse's chains
 * usually do.
 *
 * The wiring is the pair's own data flow: the rotation and the store both read the k the norm wrote
 * (operand 0), and the store writes the cache (operand 5) the fused op also writes. No temporary is
 * needed: an INOUT operand of the fused call is a step's output and the next step's input.
 *
 * `present` is ignored by construction -- this op has no optional operand, so every operand the
 * mask could describe is one the call always passes. It is read into a void cast rather than
 * assumed, because a future optional operand would have to change this comment and the compile
 * would not say so.
 *
 * ONE TRAP FOR ANY CONSUMER: step 2 mirrors `M` VERBATIM, and in a declared graph `M` is a
 * RAD_P_RANGE (the band). That is the convention libr4d's own hooks follow and the right answer --
 * collapsing a band to its high bound would hand a banded family a different kernel -- but a caller
 * that copies parameters with a switch over RAD_P_INT / RAD_P_STR / RAD_P_F64 SILENTLY DROPS the
 * range, and the step then sees M = 0 and declines. `tools/rad_kbench.cpp:2673-2680` builds its
 * step geometry exactly that way. A consumer must read a RANGE as its high bound with
 * `rad_param_getdim` (rad_plugin.h:79-86), which is what the ABI provides for this case.
 *
 * ============================ WHAT CAN AND CANNOT EXECUTE THIS ============================
 *
 * THE DECLARATION IS ACCURATE; STOCK rad-kbench CANNOT RUN IT. tools/rad_kbench.cpp:2682 resolves a
 * step with `row_in(r.row->plugin, fs.op, ...)`, i.e. INSIDE THE FUSED ROW'S OWN PLUGIN. The plugin
 * that owns this row also owns `dflash_k_norm_rope_store` and nothing else in the chain: `rmsnorm`,
 * `rope` and `kv_store` live in libr4d and libref. So a stock `--fuse` pass reports
 * "no mine kernel for step 'rmsnorm'" -- a SKIP, not a pass -- and this hook must never be cited as
 * evidence of bit-equality. The chain below is still worth publishing: the graph dump prints it
 * (core/build/rad_graph.cpp:447-455), a dedicated comparator can walk it against libr4d's rows, and
 * the doc says what "replaces" means (docs/PLUGIN.md:150-161).
 *
 * A dedicated comparator must (a) launch this row out of plugin `mine`, (b) launch each step from
 * libr4d on the same device buffers, (c) compare every INOUT/OUT operand BYTE FOR BYTE, and
 * (d) fail closed on a missing step row or a step that refuses -- a skip is not a pass.
 *
 * THE EXPECTED RESULT, STATED SO IT CAN BE FALSIFIED AND NOT ASSUMED -- an expectation, not a
 * finding. Step 0 is expected to agree, for every thread count libr4d picks:
 * `r4d_act_threads(M, 128)` returns 1024, 512 or 256 (libr4d/r4d_common.h:250-254), never below
 * the head width, so each thread folds at most one element and the threads beyond column 127
 * contribute +0.0f -- exact for a finite partial. The remaining fold is the same four wave
 * partials in the same ascending order. Steps 1 and 2 are expected to agree because they are
 * reproductions at identical arguments (powf/sincosf, the same bf16 conversion, the same paged
 * offsets). If any of that fails when the comparator runs, the difference is real and this kernel
 * is wrong -- not close. Until it runs, nothing here is established.
 */
namespace {

struct FuseStep {
    RadFuseStep* s;
    bool ok = true;

    explicit FuseStep(RadFuseStep* out) : s(out) { *s = RadFuseStep{}; }

    FuseStep& op(const char* name) { s->op = name; return *this; }

    /* VERBATIM, KIND AND ALL. `M` is a RANGE in the fused op and re-emitting a collapsed bound
     * would be a different band; `eps`, `wadd`, `theta` and `scale` are RAD_P_F64 and an integer
     * reader would destroy them. Only the ONE step whose extent is a product gets an integer. */
    FuseStep& mirror(const RadParam* p, int n_p, const char* key) {
        const RadParam* f = rad_param_find(p, n_p, key);
        if (!f || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        s->p[s->n_p++] = *f;
        return *this;
    }

    FuseStep& pint(const char* key, long long v) {
        if (s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key = key;
        q.kind = RAD_P_INT;
        q.ival = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    FuseStep& pstr(const char* key, const char* v) {
        if (!v || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key = key;
        q.kind = RAD_P_STR;
        q.sval = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    /* ONE ENTRY PER OPERAND OF THE STEP'S OP, in its schema's order. */
    int wire(int a, int b) {
        if (!ok) return RAD_E_UNSUPPORTED;
        s->from[0] = (int16_t)a;
        s->from[1] = (int16_t)b;
        s->n_from = 2;
        return RAD_OK;
    }
    int wire(int a, int b, int c) {
        if (!ok) return RAD_E_UNSUPPORTED;
        s->from[0] = (int16_t)a;
        s->from[1] = (int16_t)b;
        s->from[2] = (int16_t)c;
        s->n_from = 3;
        return RAD_OK;
    }
    int wire(int a, int b, int c, int d) {
        if (!ok) return RAD_E_UNSUPPORTED;
        s->from[0] = (int16_t)a;
        s->from[1] = (int16_t)b;
        s->from[2] = (int16_t)c;
        s->from[3] = (int16_t)d;
        s->n_from = 4;
        return RAD_OK;
    }
};

}  /* namespace */

int rad_dflash_k_store_fuse(const RadParam* p, int n_p, uint64_t present, int step,
                            RadFuseStep* out) {
    (void)present;
    if (!out) return RAD_E_INVAL;

    const int64_t rows = rad_param_getdim(p, n_p, "M", 0);
    const int64_t nkv  = rad_param_getdim(p, n_p, "n_head_kv", 0);
    const int64_t hd   = rad_param_getdim(p, n_p, "head_dim", 0);
    if (rows <= 0 || nkv <= 0 || hd <= 0 || rows > INT64_MAX / nkv) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* rmsnorm: x, w, y. The norm's row count is the number of (token, head) pairs. */
        FuseStep s(out);
        s.op("rmsnorm")
         .pint("M", rows * nkv)
         .pint("n", hd)
         .mirror(p, n_p, "eps")
         .pstr("dtype", "bf16")
         .mirror(p, n_p, "wadd");
        return s.wire(RAD_DKNS_K, RAD_DKNS_W, RAD_DKNS_K);
    }
    case 1: {
        /* rope: qkv, positions. K-only spelling, as rad_block_dflash2.h declares it: n_head is the
         * kv head count and n_head_kv is 0, so every head present is rotated.
         *
         * M IS EMITTED AS AN INTEGER, resolved from the fused geometry by rad_param_getdim -- a
         * RANGE reads as its high bound, which is the extent the band was resolved for. Nothing in
         * this chain is a banded family (`cRopeRow` / `cKvStore`, libr4d/r4d_rows.cpp:3128-3131,
         * constrain head_dim, mode, block_size and kv_dtype only), so there is no band-selection
         * meaning for a mirrored RANGE to preserve -- and emitting one is how a consumer that
         * copies parameters with a switch over RAD_P_INT/RAD_P_STR/RAD_P_F64 drops it and sees
         * M = 0. r4d_fuse mirrors verbatim precisely where a BANDED GEMM is the step. */
        FuseStep s(out);
        s.op("rope")
         .pint("M", rows)
         .mirror(p, n_p, "head_dim")
         .pint("n_head", nkv)
         .pint("n_head_kv", 0)
         .mirror(p, n_p, "theta")
         .mirror(p, n_p, "scale")
         .pstr("mode", "neox");
        return s.wire(RAD_DKNS_K, RAD_DKNS_POS);
    }
    case 2: {
        /* kv_store: k, v, slot_mapping, kv_cache. The k it reads is the rotation's output, which
         * is the fused call's own `k`; k_scale/v_scale are absent because the cache is bf16. */
        FuseStep s(out);
        s.op("kv_store")
         .pint("M", rows)
         .mirror(p, n_p, "head_dim")
         .mirror(p, n_p, "n_head_kv")
         .mirror(p, n_p, "block_size")
         .mirror(p, n_p, "kv_dtype");
        return s.wire(RAD_DKNS_K, RAD_DKNS_V, RAD_DKNS_SLOT, RAD_DKNS_KV);
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ---- the row ---------------------------------------------------------------------------------- */
/* THE CONSTRAINTS ARE THE SPECIALISATION. Without them this row would be selected for any
 * `dflash_k_norm_rope_store` in a graph and then refuse head_dim != 128 at issue, which turns "the
 * host reference or another row serves this" into a failed step. */
static const RadConstraint kConstraints[] = {
    RAD_CEQ("head_dim", 128),
    RAD_CIN("kv_dtype", "bf16"),
};

RadKernelInfo rad_dflash_k_store_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_dflash_k_norm_rope_store_hd128";
    r.op = kSchema.op;
    r.family = "attn";
    r.computes = kSchema.doc;
    r.shape = "head_dim 128, NeoX full rotary, bf16 k/v/cache, i32 positions and slots; any row "
              "count up to max_tok, any n_head_kv and block_size";
    r.dtypes = "bf16 k inout, bf16 v, f32 or bf16 gain, i32 positions/slots, bf16 paged cache";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kConstraints;
    r.n_constraints = (int)(sizeof(kConstraints) / sizeof(kConstraints[0]));
    r.launch = launch;
    r.opd_shape = rad_dflash_k_store_shape;
    r.replaces = rad_dflash_k_store_fuse;
    return r;
}
