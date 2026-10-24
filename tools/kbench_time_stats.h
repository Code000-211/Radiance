/* kbench_time_stats.h -- order statistics over per-launch times: the median a measurement reports
 * and its distribution-free interval. Nothing device-side, no state.
 *
 * Split out of kbench_core.h so a harness that is not rad-kbench reports the same statistic from the
 * same code: candidates/gdn-query-reload's comparator times one dispatch between two events of the
 * device layer and reports what this computes. tests/kbench_test.cpp checks the arithmetic on the
 * host, so the comparator's evaluation path has a CPU test that does not need a card.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rad {
namespace kb {

/* Order statistics of per-launch times in microseconds, one per timed batch. */
struct TimeStats {
    int    n = 0;
    double median = -1, min = -1, max = -1;
    double p90 = -1;          /* nearest rank: the ceil(0.9 n)-th smallest */
    /* A distribution-free confidence interval for the median: the order statistics of ranks
     * floor(n/2 - 0.98 sqrt(n)) and ceil(n/2 + 1 + 0.98 sqrt(n)), clipped to the sample. Up to
     * ten samples that is the whole range, which covers the median with probability
     * 1 - 2^(1-n): 95% from six samples on, less below (94% at five, 50% at two). From eleven on,
     * one outlier on either side leaves it unchanged. None (-1) for fewer than two samples. */
    double ci_lo = -1, ci_hi = -1;
    double ci_pct = -1;       /* half the interval's width, as a percent of the median */
};

inline TimeStats time_stats(std::vector<double> v) {
    TimeStats s;
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }),
            v.end());
    const int64_t n = (int64_t)v.size();
    s.n = (int)n;
    if (n == 0) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    s.median = n % 2 ? v[(size_t)(n / 2)] : 0.5 * (v[(size_t)(n / 2 - 1)] + v[(size_t)(n / 2)]);
    /* ceil(0.9 n) in integers: 0.9 * 20 is 18.000000000000004 in doubles. */
    s.p90 = v[(size_t)((9 * n + 9) / 10 - 1)];
    if (n < 2) return s;
    const double h = 0.98 * std::sqrt((double)n);
    const int64_t lo = std::max<int64_t>(1, (int64_t)std::floor((double)n / 2.0 - h));
    const int64_t hi = std::min<int64_t>(n, (int64_t)std::ceil((double)n / 2.0 + 1.0 + h));
    s.ci_lo = v[(size_t)(lo - 1)];
    s.ci_hi = v[(size_t)(hi - 1)];
    if (s.median > 0) s.ci_pct = 100.0 * (s.ci_hi - s.ci_lo) / 2.0 / s.median;
    return s;
}

}  /* namespace kb */
}  /* namespace rad */
