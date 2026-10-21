/* rad_spechub_config.h -- the opt-in switches the coupling reads.
 *
 * Here rather than in core/ because both the architecture plugin and the sampler read them: an
 * arch plugin may not include a core header, and a switch that the arch and the sampler disagree
 * about is exactly the failure these functions exist to prevent. Header-only and total.
 */
#ifndef RAD_SPECHUB_CONFIG_H
#define RAD_SPECHUB_CONFIG_H

#include <cstdlib>
#include <cstring>

namespace rad {

/* RADIANCE_MTP_COUPLED=1: the linear builtin MTP drafter samples x~Q each round and the target
 * sampler applies the maximal coupling (docs/OPS.md, `mtp_couple_propose`/`mtp_couple_verify`).
 * Unset or "0" is the ordinary path. Exclusive with RADIANCE_NO_DRAFT: a deployment that has
 * turned drafting off cannot couple to a draft that is never made. */
inline bool mtp_coupled_enabled() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return value && !std::strcmp(value, "1");
}

/* A typo is OFF, never something else -- and a flag that is on while drafting is off is a
 * contradiction the deployment is told about rather than served through. */
inline bool mtp_coupled_config_valid() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return (!value || !std::strcmp(value, "0") || !std::strcmp(value, "1")) &&
           !(mtp_coupled_enabled() && std::getenv("RADIANCE_NO_DRAFT"));
}

}  // namespace rad
#endif
