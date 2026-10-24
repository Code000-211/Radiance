// Isolated CV=1, NORM=2, VS=1, NT=1, ANC=1, STG=0, SDT=0 probe.
// The fork shares one device arithmetic body between its legacy signature and a compact
// 19-pointer + 72-byte config signature. Only the latter is exposed by this plugin.
// Engine scratch may be reused with arbitrary contents. The fork emits a counter clear
// before every main dispatch; the last arriving value head still resets its counter.
#ifdef __FAST_MATH__
#error "This experiment must not use fast math: the bit-identity argument needs -ffp-contract=off."
#endif

#include "rad_abi.h"
#include "r4d_emit.h"
#include <climits>

// Hide the included fork's non-CV entry, helpers, and device templates. The only visible
// symbols are the plugin ABI exports after the visibility boundary below.
#pragma GCC visibility push(hidden)

// Only the recurrent (non-CV) entry keeps its private name; the folded entry is generated
// from ru_compact_entry after including the shared arithmetic body.
#define r4d_rad_gdn_recurrent_update qreload_unused_recurrent_entry
#include "gdn_scratch_clear_fork.hip"
#undef r4d_rad_gdn_recurrent_update
R4D_ENTRY(qreload_conv_recurrent_entry, ru_compact_entry)

// The fork calls r4d_gdn_dims to check its compiled-in geometry. libr4d defines it in the chunk
// scan's translation unit, which this probe does not build. These are the three numbers that
// kernel bakes in (KD/VD/BT in r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip), restated here so this
// DSO is self-contained and cannot silently disagree with what it forked. Defined inside the
// hidden region so it matches the declaration r4d.h made there.
extern "C" void r4d_gdn_dims(int* head_k, int* head_v, int* chunk) {
  if (head_k) *head_k = RU_K;
  if (head_v) *head_v = RU_V;
  if (chunk)  *chunk  = 64;
}

namespace qreload {
static const RadConstraint constraints[] = {
    RAD_CEQ("head_k", 128), RAD_CEQ("head_v", 128), RAD_CEQ("conv_width", 4),
    RAD_CGE("q_len", 1), RAD_CLE("q_len", 16),
};

static int64_t scratch(const RadArgs*) {
    // Declare supplies only parameters; the runtime enforces the same bound on N * Hg.
    return (int64_t)kMaxConvCtr * sizeof(unsigned int);
}

}  // namespace qreload

// The plugin ABI exports are the only visible symbols in this DSO.
#pragma GCC visibility pop

// THE SCHEMA IS LIBR4D'S. Declaring it here would make this DSO the schema owner for an op
// libr4d already declares, and the first plugin to declare an op fixes its vocabulary -- a
// re-declaration that drifted by one operand would then be refused against libref for a reason
// that has nothing to do with this probe. Same reasoning as hc-down-probe: the installed libr4d
// owns the op, this row implements it.
extern "C" int rad_kernel_schema_count() { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }

// The DSO's only row. `rad_kernel_at` is the ABI's registration; nothing in mine/, the
// CMakeLists or the frozen tree is touched, and the parent decides whether to register it.
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    static const RadKernelInfo row = [] {
        RadKernelInfo r{};
        r.name    = "gdn_conv_recurrent_update_k128_v128_bf16_qreload";
        r.op      = "gdn_conv_recurrent_update";
        r.family  = "gdn";
        r.computes = "the folded gdn decode: emitted counter clear followed by conv and "
                     "recurrent update with gated norm and folded quantiser; under CV the "
                     "query and key are re-read from LDS instead of held across the update "
                     "(inherited query/key reload arithmetic)";
        r.shape   = "head_k/head_v 128, conv width 4, 1..16 tokens per sequence, "
                    "anchor state, one workgroup per (sequence, value head)";
        r.dtypes  = "bf16 projection/conv/a/b/gate/output, f32 recurrent state, "
                    "int8 quant output and f32 scale; optional bf16 conv bias";
        r.domain  = RAD_DOMAIN_DEVICE;
        // Priority is intra-plugin only. This private row is not registered for serving;
        // the comparator addresses this DSO directly.
        r.priority = 40;
        r.constraints   = qreload::constraints;
        r.n_constraints = (int)(sizeof(qreload::constraints) / sizeof(qreload::constraints[0]));
        r.launch   = qreload_conv_recurrent_entry;
        r.describe = qreload_conv_recurrent_entry_describe;
        r.scratch  = qreload::scratch;
        return r;
    }();
    return i == 0 ? &row : nullptr;
}

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "gdn_scratch_clear_probe", "0.0.0",
        "isolated CV query/key reload with per-dispatch scratch clear; not registered in mine", "gfx1201"
    };
    return &info;
}