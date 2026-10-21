/* plugin/spechub/spechub_plugin.cpp -- the argument packers, the call structs and the exports.
 *
 * The call structs are read by BOTH the device launch (spechub.hip) and the host reference
 * (ref_spechub.cpp), so the two are compared on the same packed arguments rather than on two
 * hand-written readings of the same operands. The arithmetic is in abi/rad_spechub.h.
 *
 * The plugin exports the whole kernel-plugin set: schema count/at, kernel count/at, the ABI version,
 * the info block, and the reference marker that keeps this library out of serving unless asked.
 */
#include "rad_plugin.h"
#include "rad_spechub.h"

#include <climits>

/* Everything the launchers need, packed once. `rad_mtp_couple_propose_launch` and
 * `rad_mtp_couple_verify_launch` are the device entries in spechub.hip.
 *
 * The leading integer parameters are passed to the kernel SEPARATELY rather than through the struct,
 * because a device kernel cannot take a host struct by value and have graph replay rebind the
 * operand addresses inside it. */
int rad_mtp_couple_propose_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_verify_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_propose_args(const RadArgs*, RadMtpCoupleProposeCall*);
int rad_mtp_couple_verify_args(const RadArgs*, RadMtpCoupleVerifyCall*);
int rad_mine_mtp_couple_propose_launch(const RadArgs*, RadStream);
int rad_mine_mtp_couple_verify_launch(const RadArgs*, RadStream);
int rad_ref_mtp_couple_propose(const RadArgs*, RadStream);
int rad_ref_mtp_couple_verify(const RadArgs*, RadStream);
const RadKernelInfo* rad_spechub_row(int i, RadLaunchFn launch);

int rad_mtp_couple_propose_args(const RadArgs* a, RadMtpCoupleProposeCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M, v, nc;
    const int rc = rad_couple_validate(a, /*op=*/0, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadMtpCoupleProposeCall c{};
    c.M = M;
    c.vocab = v;
    c.round = (int)rad_param_geti(a->p, a->n_p, "round", 0);
    c.idx  = (int32_t*)a->t[0].data;
    c.val  = (const float*)a->t[1].data;
    c.card = (const int32_t*)a->t[2].data;
    c.draw = (const uint32_t*)a->t[3].data;
    c.state = (uint32_t*)a->t[4].data;
    c.ld_idx  = a->t[0].stride[0];
    c.ld_val  = a->t[1].stride[0];
    c.ld_draw = a->t[3].stride[0];
    c.st_round = a->t[4].stride[0];
    c.st_row   = a->t[4].stride[1];
    *out = c;
    return RAD_OK;
}

int rad_mtp_couple_verify_args(const RadArgs* a, RadMtpCoupleVerifyCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M, v, nc;
    const int rc = rad_couple_validate(a, /*op=*/1, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadMtpCoupleVerifyCall c{};
    c.M = M;
    c.vocab = v;
    c.n_cand = nc;
    c.idx    = (const int32_t*)a->t[0].data;
    c.val    = (const float*)a->t[1].data;
    c.params = (const uint32_t*)a->t[2].data;
    c.state  = (const uint32_t*)a->t[3].data;
    c.token  = (int32_t*)a->t[4].data;
    for (int i = 0; i < 5; ++i) c.ld[i] = a->t[i].stride[0];
    *out = c;
    return RAD_OK;
}

RadKernelInfo rad_spechub_row(int i, RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = i ? "spechub_mtp_couple_verify" : "spechub_mtp_couple_propose";
    r.op = i ? "mtp_couple_verify" : "mtp_couple_propose";
    r.family = "draft";
    r.computes = i ? "maximal coupling over the surviving target candidates, FP64 P"
                   : "x~Q over the final top8, FP64 T1 softmax";
    r.shape = i ? "M,n_cand,n_vocab" : "M,n_vocab,round,max_rounds<=8";
    r.dtypes = "f32,i32,u32";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.launch = launch;
    r.opd_shape = i ? rad_mtp_couple_verify_shape : rad_mtp_couple_propose_shape;
    return r;
}

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "spechub", "0.1.0",
    "Linear-MTP maximal coupling: a Q draw per round and the coupled target verify", "gfx1201",
};
}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
/* A REFERENCE LIBRARY, because its rows are checked against ref_spechub.cpp and because the flag
 * that enables the coupling is what should put this row into serving -- not a load order. */
extern "C" int rad_plugin_reference(void) { return 1; }

extern "C" int rad_kernel_schema_count(void) {
    return rad_spechub_schema_count();
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return i >= 0 && i < rad_spechub_schema_count() ? rad_spechub_schema_at(i) : nullptr;
}
extern "C" int rad_kernel_count(void) {
    /* One device row each, and the host reference beside them so a card-less machine can still run
     * the ops the flag enables. */
    return 4;
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    switch (i) {
        case 0: return rad_spechub_row(0, rad_mine_mtp_couple_propose_launch);
        case 1: return rad_spechub_row(1, rad_mine_mtp_couple_verify_launch);
        case 2: {
            static const RadKernelInfo r = [] {
                RadKernelInfo x = rad_spechub_row(0, rad_ref_mtp_couple_propose);
                x.name = "ref_mtp_couple_propose";
                x.domain = RAD_DOMAIN_HOST;
                x.priority = 0;
                return x;
            }();
            return &r;
        }
        case 3: {
            static const RadKernelInfo r = [] {
                RadKernelInfo x = rad_spechub_row(1, rad_ref_mtp_couple_verify);
                x.name = "ref_mtp_couple_verify";
                x.domain = RAD_DOMAIN_HOST;
                x.priority = 0;
                return x;
            }();
            return &r;
        }
        default: return nullptr;
    }
}
