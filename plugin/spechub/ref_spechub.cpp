/* plugin/spechub/ref_spechub.cpp -- the two coupling ops on the host.
 *
 * ONE ROW AT A TIME, in the same order and with the same arithmetic as the kernel: these functions
 * call the very same abi/rad_spechub.h helpers the device launch is built from, so an op and its
 * reference cannot drift. What they add is the row loop and the same eligibility gate.
 *
 * This is the oracle rad-kbench checks the device rows against, and the only implementation a
 * machine with no card can run.
 */
#include "spechub.h"

int rad_ref_mtp_couple_propose(const RadArgs* a, RadStream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t row = 0; row < c.M; ++row) rad_mtp_couple_propose_row(c, row);
    return RAD_OK;
}

int rad_ref_mtp_couple_verify(const RadArgs* a, RadStream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t row = 0; row < c.M; ++row) rad_mtp_couple_verify_row(c, row);
    return RAD_OK;
}
