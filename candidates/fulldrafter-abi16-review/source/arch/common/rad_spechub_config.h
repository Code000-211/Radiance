#ifndef RAD_SPECHUB_CONFIG_H
#define RAD_SPECHUB_CONFIG_H

#include <cstdlib>
#include <cstring>

namespace rad {
inline bool spechub_enabled() {
    const char* value = std::getenv("RADIANCE_SPECHUB");
    return value && !std::strcmp(value, "1");
}
inline bool spechub_transport() {
    const char* value = std::getenv("RADIANCE_SPECHUB_VERIFY");
    return !value || !std::strcmp(value, "transport");
}
inline bool spechub_config_valid() {
    const char* value = std::getenv("RADIANCE_SPECHUB");
    const char* verify = std::getenv("RADIANCE_SPECHUB_VERIFY");
    return (!value || !std::strcmp(value, "0") || !std::strcmp(value, "1")) &&
           (!verify || (spechub_enabled() &&
                        (!std::strcmp(verify, "match") || !std::strcmp(verify, "transport"))));
}
/* RADIANCE_MTP_COUPLED=1: linear builtin MTP drafts x~Q each round and the target sampler
 * applies the maximal coupling (docs/OPS.md `mtp_couple_*`). Unset or "0" is the ordinary path.
 * Exclusive with SpecHub, which owns the one-position tree experiment. */
inline bool mtp_coupled_enabled() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return value && !std::strcmp(value, "1");
}
inline bool mtp_coupled_config_valid() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return (!value || !std::strcmp(value, "0") || !std::strcmp(value, "1")) &&
           !(mtp_coupled_enabled() && (spechub_enabled() || std::getenv("RADIANCE_NO_DRAFT")));
}
} // namespace rad
#endif
