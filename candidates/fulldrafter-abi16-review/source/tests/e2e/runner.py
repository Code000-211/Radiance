#!/usr/bin/env python3
"""E2E Test Runner Harness for Speculative Drafter.

Unified test harness for discovering, filtering, executing, and reporting
across all 4 test tiers (Tier 1 Feature Coverage, Tier 2 Boundary & Corner,
Tier 3 Cross-Feature, Tier 4 Real-World Application) for Features 1-9.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
import unittest
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


class StructuredTestResult(unittest.TestResult):
    """Custom TestResult that collects per-test timings, errors, tiers, and features."""

    def __init__(self, stream: Any = None, descriptions: Any = None, verbosity: int = 1) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.test_records: list[dict[str, Any]] = []
        self._current_start_time: float = 0.0
        self.feature_map: dict[int, set[str]] = {i: set() for i in range(1, 10)}
        self.tier_counts: dict[str, int] = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._current_start_time = time.perf_counter()

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        duration = time.perf_counter() - self._current_start_time
        self._record_test(test, "PASSED", duration, None)

    def addFailure(self, test: unittest.TestCase, err: tuple[Any, ...]) -> None:
        super().addFailure(test, err)
        duration = time.perf_counter() - self._current_start_time
        self._record_test(test, "FAILED", duration, self._exc_info_to_string(err, test))

    def addError(self, test: unittest.TestCase, err: tuple[Any, ...]) -> None:
        super().addError(test, err)
        duration = time.perf_counter() - self._current_start_time
        self._record_test(test, "ERROR", duration, self._exc_info_to_string(err, test))

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        duration = time.perf_counter() - self._current_start_time
        self._record_test(test, "SKIPPED", duration, reason)

    def _record_test(self, test: unittest.TestCase, status: str, duration: float, detail: str | None) -> None:
        test_id = test.id()
        tier = getattr(test, "tier", None)

        if tier is None:
            if "test_tier1" in test_id:
                tier = "tier1"
            elif "test_tier2" in test_id:
                tier = "tier2"
            elif "test_tier3" in test_id:
                tier = "tier3"
            elif "test_tier4" in test_id:
                tier = "tier4"
            else:
                tier = "tier1"

        feature_id = None
        f_match = re.search(r"test_f(\d{2})", test_id)
        if f_match:
            feature_id = int(f_match.group(1))
        else:
            x_match = re.search(r"test_x(\d{2})", test_id)
            if x_match:
                feature_id = (int(x_match.group(1)) - 1) % 9 + 1
            else:
                w_match = re.search(r"test_w(\d{2})", test_id)
                if w_match:
                    feature_id = (int(w_match.group(1)) - 1) % 9 + 1

        if tier in self.tier_counts and status == "PASSED":
            self.tier_counts[tier] += 1

        if feature_id and status == "PASSED" and 1 <= feature_id <= 9:
            self.feature_map[feature_id].add(test_id)

        self.test_records.append({
            "test_id": test_id,
            "tier": tier,
            "feature_id": feature_id,
            "status": status,
            "duration": round(duration, 4),
            "detail": detail,
        })


def create_filtered_suite(
    selected_tier: str,
    selected_feature: int | None,
) -> unittest.TestSuite:
    """Discovers and filters test cases by tier and feature ID."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    tier_modules = {
        "tier1": "tests.e2e.test_tier1_feature_coverage",
        "tier2": "tests.e2e.test_tier2_boundary_corner",
        "tier3": "tests.e2e.test_tier3_cross_feature",
        "tier4": "tests.e2e.test_tier4_real_world",
    }

    modules_to_load = []
    if selected_tier == "all":
        modules_to_load = list(tier_modules.values())
    elif selected_tier in tier_modules:
        modules_to_load = [tier_modules[selected_tier]]
    else:
        raise ValueError(f"Invalid tier: {selected_tier}")

    for mod_name in modules_to_load:
        try:
            mod = importlib.import_module(mod_name)
            discovered = loader.loadTestsFromModule(mod)

            for test in iter_test_cases(discovered):
                test_id = test.id()
                feat = None
                f_match = re.search(r"test_f(\d{2})", test_id)
                if f_match:
                    feat = int(f_match.group(1))
                else:
                    x_match = re.search(r"test_x(\d{2})", test_id)
                    if x_match:
                        feat = (int(x_match.group(1)) - 1) % 9 + 1
                    else:
                        w_match = re.search(r"test_w(\d{2})", test_id)
                        if w_match:
                            feat = (int(w_match.group(1)) - 1) % 9 + 1

                if selected_feature is not None and feat != selected_feature:
                    continue
                suite.addTest(test)
        except ImportError as e:
            print(f"[WARN] Could not import {mod_name}: {e}", file=sys.stderr)

    return suite


def iter_test_cases(test_suite: unittest.TestSuite) -> Any:
    """Recursively yields individual test cases from a test suite."""
    for item in test_suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_test_cases(item)
        else:
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E Test Runner for Speculative FullDrafter")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Select test tier (1, 2, 3, 4, or all)")
    parser.add_argument("--feature", type=int, choices=range(1, 10), default=None,
                        help="Filter by specific feature ID (1 to 9)")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock",
                        help="Execution mode (mock=baseline fixtures, live=hardware)")
    parser.add_argument("--port", type=int, default=8002,
                        help="Target test port (default: 8002, NEVER use port 8000)")
    parser.add_argument("--report", choices=["console", "json"], default="console",
                        help="Output report format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for JSON report")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop test execution on first failure")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose console output")

    args = parser.parse_args()

    if args.port == 8000:
        print("[ERROR] Port 8000 is reserved for production! Touching port 8000 is strictly prohibited.", file=sys.stderr)
        sys.exit(1)

    os.environ["RADIANCE_TEST_PORT"] = str(args.port)
    os.environ["DRAFTER_TEST_MODE"] = args.mode

    tier_key = f"tier{args.tier}" if args.tier != "all" else "all"
    suite = create_filtered_suite(tier_key, args.feature)

    total_discovered = suite.countTestCases()
    if total_discovered == 0:
        print("[ERROR] No tests discovered matching the specified criteria.", file=sys.stderr)
        sys.exit(1)

    if args.report == "console":
        print("=" * 80)
        print(" speculative fulldrafter — e2e test harness runner")
        print("=" * 80)
        print(f" Environment Python : {sys.executable}")
        print(f" Execution Mode     : {args.mode.upper()}")
        print(f" Target Port        : {args.port}")
        print(f" Target Tier        : {tier_key.upper()}")
        print(f" Target Feature     : {args.feature if args.feature else 'ALL (1..9)'}")
        print(f" Discovered Tests   : {total_discovered}")
        print("-" * 80)

    result = StructuredTestResult()
    start_time = time.perf_counter()

    ansi_green = "\033[92m"
    ansi_red = "\033[91m"
    ansi_yellow = "\033[93m"
    ansi_reset = "\033[0m"

    for idx, test in enumerate(iter_test_cases(suite), start=1):
        test_name = test.id().split(".")[-1]
        feat = getattr(test, "feature_id", "?")
        tier = getattr(test, "tier", "unknown")

        test.run(result)

        latest_record = result.test_records[-1] if result.test_records else None
        status = latest_record["status"] if latest_record else "UNKNOWN"
        dur = latest_record["duration"] if latest_record else 0.0

        if args.report == "console":
            if status == "PASSED":
                status_tag = f"[{ansi_green}PASS{ansi_reset}]"
            elif status in ("FAILED", "ERROR"):
                status_tag = f"[{ansi_red}FAIL{ansi_reset}]"
            else:
                status_tag = f"[{ansi_yellow}SKIP{ansi_reset}]"

            if args.verbose or status != "PASSED":
                print(f"{status_tag} ({idx:3d}/{total_discovered}) [F{feat:>2}|{tier}] {test_name:<48} ({dur:.3f}s)")
            else:
                print(f"\rProgress: {idx}/{total_discovered} tests executed... {status_tag}", end="", flush=True)

        if args.fail_fast and not result.wasSuccessful():
            break

    elapsed_time = time.perf_counter() - start_time
    if args.report == "console" and not args.verbose:
        print()

    passed_count = len([r for r in result.test_records if r["status"] == "PASSED"])
    failed_count = len([r for r in result.test_records if r["status"] in ("FAILED", "ERROR")])
    skipped_count = len([r for r in result.test_records if r["status"] == "SKIPPED"])

    # Compute feature coverage
    features_covered = sum(1 for _, test_set in result.feature_map.items() if len(test_set) >= 5)

    is_certified = result.wasSuccessful() and (
        (args.feature is not None)
        or (passed_count >= 106 and features_covered >= 9)
        or (args.tier != "all" and passed_count == total_discovered)
    )

    summary_data = {
        "status": "PASSED" if result.wasSuccessful() else "FAILED",
        "total_tests": total_discovered,
        "executed_tests": len(result.test_records),
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "duration_seconds": round(elapsed_time, 3),
        "tier_counts": result.tier_counts,
        "features_fully_covered": features_covered,
        "total_features": 9,
        "gate_certification_passed": is_certified,
        "records": result.test_records,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    if args.report == "json":
        print(json.dumps(summary_data, indent=2))
    else:
        gate_status_str = f"{ansi_green}CERTIFIED (TEST_READY){ansi_reset}" if is_certified else f"{ansi_red}NOT CERTIFIED{ansi_reset}"
        print("\n" + "=" * 80)
        print(" test execution summary")
        print("=" * 80)
        print(f" Total Executed  : {len(result.test_records)} / {total_discovered}")
        print(f" Passed          : {ansi_green}{passed_count}{ansi_reset}")
        print(f" Failed / Error  : {ansi_red}{failed_count}{ansi_reset}")
        print(f" Skipped         : {ansi_yellow}{skipped_count}{ansi_reset}")
        print(f" Wall Time       : {elapsed_time:.2f} seconds")
        print(f" Tier Breakdown  : T1={result.tier_counts['tier1']} | T2={result.tier_counts['tier2']} | T3={result.tier_counts['tier3']} | T4={result.tier_counts['tier4']}")
        print(f" Features Covered: {features_covered} / 9 features with >=5 passing tests")
        print(f" Gate Status     : {gate_status_str}")
        print("=" * 80)

        if failed_count > 0:
            print("\nFailures and Errors:")
            for rec in result.test_records:
                if rec["status"] in ("FAILED", "ERROR"):
                    print(f"\n--- {rec['test_id']} ---")
                    print(rec["detail"])

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
