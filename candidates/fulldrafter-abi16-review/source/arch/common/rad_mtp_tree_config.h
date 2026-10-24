#ifndef RAD_MTP_TREE_CONFIG_H
#define RAD_MTP_TREE_CONFIG_H

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace rad {
/* Startup-only MTP experiment. cost[n] is the measured cost of ROOT + n
 * proposal rows, n=0..7. There is deliberately no invented calibration.
 * score/cost q is a proxy: full coarse vocabulary logits with local top-8 FP32
 * reranks replaced, normalized globally. pruned instead normalizes merged top8. */
struct MtpTreeConfig {
    int policy = 0; // off=0, score=1, cost=2, pruned=3
    int rounds = 3;
    double costs[8]{};
    const char* error = nullptr;
    bool pruned() const { return policy == 3; }
    static constexpr int width = 8, candidates = 8, depth = 5, budget = 7;
    static constexpr double beta = -0.2;
};

inline bool parse_mtp_tree_config(MtpTreeConfig& out, const char** error = nullptr) {
    MtpTreeConfig c{};
    const char* policy = std::getenv("RADIANCE_MTP_TREE_POLICY");
    const char* rounds = std::getenv("RADIANCE_MTP_TREE_ROUNDS");
    const char* costs = std::getenv("RADIANCE_MTP_TREE_COSTS");
    if (policy && std::strcmp(policy, "off")) {
        if (!std::strcmp(policy, "score")) c.policy = 1;
        else if (!std::strcmp(policy, "cost")) c.policy = 2;
        else if (!std::strcmp(policy, "pruned")) c.policy = 3;
        else c.error = "RADIANCE_MTP_TREE_POLICY must be off, score, cost, or pruned";
    }
    if (rounds) {
        if (!std::strcmp(rounds, "3")) c.rounds = 3;
        else if (!std::strcmp(rounds, "5")) c.rounds = 5;
        else if (!std::strcmp(rounds, "8")) c.rounds = 8;
        else c.error = "RADIANCE_MTP_TREE_ROUNDS must be 3, 5 or 8";
    }
    /* Fixed MTP8/64 tree (8 spine + 56 alternates + root) has no scored/cost/pruned variant. */
    if (c.rounds == 8 && c.policy != 0)
        c.error = "RADIANCE_MTP_TREE_ROUNDS=8 requires RADIANCE_MTP_TREE_POLICY=off";
    if (c.pruned() && c.rounds != MtpTreeConfig::depth)
        c.error = "pruned policy requires RADIANCE_MTP_TREE_ROUNDS=5";
    if (c.policy == 2) {
        if (!costs || !*costs) c.error = "cost policy requires RADIANCE_MTP_TREE_COSTS";
        else {
            const char* p = costs;
            for (int n = 0; n < 8; ++n) {
                char* end = nullptr;
                errno = 0;
                const double value = std::strtod(p, &end);
                if (end == p || errno == ERANGE || !std::isfinite(value) || value <= 0 ||
                    (n < 7 ? *end != ',' : *end != '\0')) {
                    c.error = "RADIANCE_MTP_TREE_COSTS requires exactly 8 finite positive costs (root + n rows, n=0..7)";
                    break;
                }
                c.costs[n] = value;
                p = n < 7 ? end + 1 : end;
            }
        }
    } else if (costs) {
        c.error = "RADIANCE_MTP_TREE_COSTS is only valid with cost policy";
    }
    out = c;
    if (error) *error = c.error;
    return c.error == nullptr;
}

inline const MtpTreeConfig& mtp_tree_config() {
    static const MtpTreeConfig config = [] {
        MtpTreeConfig c{};
        parse_mtp_tree_config(c);
        return c;
    }();
    return config;
}
} // namespace rad
#endif
