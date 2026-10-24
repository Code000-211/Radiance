/* A test harness small enough to read in one sitting. No framework, no Python, no dependency.
 *   TEST(name) { CHECK(cond); CHECK_EQ(a, b); }
 * main() runs every registered test and returns non-zero on the first failure count.
 *
 * A BINARY THAT CHECKED NOTHING IS NOT A BINARY THAT PASSED, and that is what the exit code says.
 * Several files here guard cases on a model file, a vocabulary or a backend, and print a "  SKIP "
 * line and return when it is absent. ctest's SKIP_REGULAR_EXPRESSION cannot carry that signal:
 * the property is evaluated BEFORE the return code, so a binary that prints one skip line and
 * then FAILS is reported Skipped and counted toward "100% tests passed".
 *
 * So the skip signal is the exit code and it is derived rather than declared: every CHECK
 * increments a counter, and main returns 77 (ctest's SKIP_RETURN_CODE) only when nothing failed
 * AND nothing was checked at all. A file where some cases ran and others skipped passes on the
 * ones that ran, which is the honest answer, and a failure in any of them is a failure.
 */
#pragma once
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

namespace radtest {

struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& cases() { static std::vector<Case> v; return v; }
inline int& failures() { static int n = 0; return n; }
/* How many assertions actually ran. The difference between "everything passed" and "nothing was
 * asked" is not visible in the failure count, and it is the whole of what a skip means. */
inline int& checks() { static int n = 0; return n; }
inline const char*& current() { static const char* s = ""; return s; }

struct Reg { Reg(const char* n, std::function<void()> f) { cases().push_back({n, f}); } };

inline void fail(const char* file, int line, const std::string& what) {
    ++failures();
    fprintf(stderr, "  FAIL %s\n    %s:%d: %s\n", current(), file, line, what.c_str());
}

template <class A, class B>
inline std::string show(const A& a, const B& b) {
    char buf[512];
    if constexpr (std::is_convertible_v<A, std::string> && std::is_convertible_v<B, std::string>)
        snprintf(buf, sizeof buf, "\"%s\" vs \"%s\"",
                 std::string(a).c_str(), std::string(b).c_str());
    else if constexpr (std::is_floating_point_v<A> || std::is_floating_point_v<B>)
        snprintf(buf, sizeof buf, "%g vs %g", (double)a, (double)b);
    else
        snprintf(buf, sizeof buf, "%lld vs %lld", (long long)a, (long long)b);
    return buf;
}

}  /* namespace radtest */

#define TEST(name)                                                                    \
    static void radtest_##name();                                                     \
    static ::radtest::Reg radtest_reg_##name(#name, radtest_##name);                  \
    static void radtest_##name()

#define CHECK(cond)                                                                   \
    do { ++::radtest::checks();                                                       \
         if (!(cond)) ::radtest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); } while (0)

#define CHECK_EQ(a, b)                                                                \
    do { ++::radtest::checks();                                                       \
         auto _a = (a); auto _b = (b);                                                \
         if (!(_a == _b)) ::radtest::fail(__FILE__, __LINE__,                          \
             std::string("CHECK_EQ(" #a ", " #b "): ") + ::radtest::show(_a, _b));    \
    } while (0)

/* A CHECK WHOSE FAILURE MAKES THE REST OF THE CASE MEANINGLESS -- the size of a container the
 * next line indexes, a pointer the next line dereferences. It reports exactly like CHECK and
 * then leaves the case, because the alternative is a segfault: the binary dies mid-suite, every
 * case after it never runs, and a run that should have reported one failure reports one crash
 * and no detail at all. Use CHECK where the case can keep going and learn more. */
#define REQUIRE(cond)                                                                 \
    do { ++::radtest::checks();                                                       \
         if (!(cond)) { ::radtest::fail(__FILE__, __LINE__, "REQUIRE(" #cond ")");    \
                        return; } } while (0)

#define REQUIRE_EQ(a, b)                                                              \
    do { ++::radtest::checks();                                                       \
         auto _a = (a); auto _b = (b);                                                \
         if (!(_a == _b)) { ::radtest::fail(__FILE__, __LINE__,                        \
             std::string("REQUIRE_EQ(" #a ", " #b "): ") + ::radtest::show(_a, _b));  \
             return; } } while (0)

#define CHECK_NEAR(a, b, tol)                                                         \
    do { ++::radtest::checks();                                                       \
         double _a = (double)(a), _b = (double)(b);                                   \
         if (!(std::fabs(_a - _b) <= (tol))) ::radtest::fail(__FILE__, __LINE__,       \
             std::string("CHECK_NEAR(" #a ", " #b "): ") + ::radtest::show(_a, _b));  \
    } while (0)

#define CHECK_OK(expr)                                                                \
    do { ++::radtest::checks();                                                       \
         int _s = (expr);                                                             \
         if (_s < 0) ::radtest::fail(__FILE__, __LINE__,                               \
             std::string("CHECK_OK(" #expr "): ") + rad_strerror(_s)); } while (0)

#define REQUIRE_OK(expr)                                                              \
    do { ++::radtest::checks();                                                       \
         int _s = (expr);                                                             \
         if (_s < 0) { ::radtest::fail(__FILE__, __LINE__,                            \
             std::string("REQUIRE_OK(" #expr "): ") + rad_strerror(_s));              \
             return; } } while (0)

/* 77 is ctest's SKIP_RETURN_CODE, set in tests/CMakeLists.txt. Returned ONLY when nothing failed
 * and nothing was checked -- a file whose every case guarded itself out. */
#define RAD_TEST_MAIN()                                                               \
    int main() {                                                                      \
        for (auto& c : ::radtest::cases()) {                                          \
            ::radtest::current() = c.name;                                            \
            int before = ::radtest::failures();                                       \
            c.fn();                                                                   \
            if (::radtest::failures() == before) fprintf(stderr, "  ok   %s\n", c.name); \
        }                                                                             \
        if (::radtest::failures())                                                    \
            fprintf(stderr, "%d failure(s)\n", ::radtest::failures());                 \
        if (::radtest::failures()) return 1;                                          \
        if (::radtest::checks() == 0) {                                               \
            fprintf(stderr, "nothing was checked: every case in this binary guarded "  \
                            "itself out\n");                                          \
            return 77;                                                                \
        }                                                                             \
        fprintf(stderr, "%d check(s) over %zu case(s)\n", ::radtest::checks(),         \
                ::radtest::cases().size());                                            \
        return 0;                                                                     \
    }
