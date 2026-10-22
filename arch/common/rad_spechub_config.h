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
 * Unset or "0" is the ordinary path.
 *
 * Nothing here enumerates the drafter configurations the coupling cannot serve. Drafting is
 * switched off by asking for no speculative window (`--num-speculative-tokens 0`), which leaves
 * `RadDrafterDecl::kind == RAD_DRAFT_NONE`; a block drafter declares RAD_DRAFT_BLOCK. Both are
 * read from the declaration by the engine (Engine::step_state_init), which is where they are
 * known. An environment variable that no other code reads cannot enforce either. */
inline bool mtp_coupled_enabled() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return value && !std::strcmp(value, "1");
}

/* A value other than unset/0/1 is REFUSED, not read as off: a deployment that asked for coupling
 * and got a typo must not be served the uncoupled path while believing otherwise. The refusal is
 * RAD_E_INVAL out of Sampler::declare (and out of the architecture's own check where it makes one),
 * both before the first request. */
inline bool mtp_coupled_config_valid() {
    const char* value = std::getenv("RADIANCE_MTP_COUPLED");
    return !value || !std::strcmp(value, "0") || !std::strcmp(value, "1");
}

}  // namespace rad
#endif
