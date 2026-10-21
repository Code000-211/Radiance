#include "spechub.h"
int rad_ref_spechub_propose(const RadArgs* a,RadStream) {
    RadSpechubProposeCall c{};int rc=rad_spechub_propose_args(a,&c);if (rc!=RAD_OK) return rc;
    for (int64_t row=0;row<c.M;++row) rad_spechub_propose_row(c,row);
    return RAD_OK;
}
int rad_ref_spechub_verify(const RadArgs* a,RadStream) {
    RadSpechubVerifyCall c{};int rc=rad_spechub_verify_args(a,&c);if (rc!=RAD_OK) return rc;
    for (int64_t row=0;row<c.M;++row) c.token[row*c.ld[4]]=rad_spechub_verify_row(c,row);
    return RAD_OK;
}
int rad_ref_mtp_couple_propose(const RadArgs* a,RadStream) {
    RadMtpCoupleProposeCall c{};int rc=rad_mtp_couple_propose_args(a,&c);if (rc!=RAD_OK) return rc;
    for (int64_t row=0;row<c.M;++row) rad_mtp_couple_propose_row(c,row);
    return RAD_OK;
}
int rad_ref_mtp_couple_verify(const RadArgs* a,RadStream) {
    RadMtpCoupleVerifyCall c{};int rc=rad_mtp_couple_verify_args(a,&c);if (rc!=RAD_OK) return rc;
    for (int64_t row=0;row<c.M;++row) rad_mtp_couple_verify_row(c,row);
    return RAD_OK;
}
int rad_ref_spechub_count() {return 4;}
const RadKernelInfo* rad_ref_spechub_at(int i) {
    static const RadKernelInfo rows[]={
        [] {auto r=rad_spechub_row(0,rad_ref_spechub_propose);r.name="ref_spechub_propose";r.domain=RAD_DOMAIN_HOST;r.priority=0;return r;}(),
        [] {auto r=rad_spechub_row(1,rad_ref_spechub_verify);r.name="ref_spechub_verify";r.domain=RAD_DOMAIN_HOST;r.priority=0;return r;}(),
        [] {auto r=rad_mtp_couple_row(0,rad_ref_mtp_couple_propose);r.name="ref_mtp_couple_propose";r.domain=RAD_DOMAIN_HOST;r.priority=0;return r;}(),
        [] {auto r=rad_mtp_couple_row(1,rad_ref_mtp_couple_verify);r.name="ref_mtp_couple_verify";r.domain=RAD_DOMAIN_HOST;r.priority=0;return r;}()};
    return i>=0 && i<4 ? &rows[i] : nullptr;
}
