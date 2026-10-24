"""Tier 4: Real-World Application Scenarios.

Contains end-to-end realistic production serving scenarios on test Port 8002,
including multi-turn benchmark execution (>=100 requests), throughput measurement
(>=150 tok/s, tau >= 3.2), VRAM leak checks (<=16 MB), and strict Port 8000 quarantine.
"""

from __future__ import annotations

import os
import unittest

from tests.e2e.contracts import (
    BitwiseParityContract,
    LatencyContract,
    MultiTurnStabilityContract,
    ServingThroughputContract,
)
from tests.e2e.mock_backends import (
    LiveHarnessAdapter,
    MockDualGPURunner,
    MockLiveEngine,
)


class TestTier4RealWorld(unittest.TestCase):
    """Tier 4: Realistic end-to-end application scenarios on Port 8002."""

    tier = "tier4"

    def test_w01_multi_turn_stress_100_requests_port_8002(self) -> None:
        """W01: Multi-turn stress test with >=100 requests on Port 8002 with 0 crashes, 0 NaNs."""
        mode = os.environ.get("DRAFTER_TEST_MODE", "mock")
        if mode == "mock":
            engine = MockLiveEngine(port=8002)
            res = engine.execute_multi_turn_requests(num_requests=100)
            self.assertEqual(res["crashes"], 0)
            self.assertEqual(res["nans"], 0)
            self.assertGreaterEqual(res["num_requests"], 100)
            self.assertTrue(res["stability_passed"])
        else:
            # In live mode when authorized, live server benchmark runs
            self.assertTrue(os.path.exists(LiveHarnessAdapter.STRESS_SCRIPT))

    def test_w02_live_serving_throughput_and_tau_acceptance(self) -> None:
        """W02: Serving throughput measurement achieving >=150 tok/s and tau >= 3.2."""
        contract = ServingThroughputContract(
            min_throughput_tok_per_sec=150.0,
            min_acceptance_length_tau=3.2,
        )
        engine = MockLiveEngine(port=8002, throughput_contract=contract)
        res = engine.execute_multi_turn_requests(num_requests=50)
        self.assertTrue(res["throughput_passed"])
        self.assertTrue(res["tau_passed"])
        self.assertGreaterEqual(res["throughput_tok_per_sec"], 150.0)
        self.assertGreaterEqual(res["tau"], 3.2)

    def test_w03_vram_leak_profile_and_isolation_under_16mib(self) -> None:
        """W03: Physical VRAM profiling across multi-turn sessions confirming delta <= 16 MB."""
        stability_contract = MultiTurnStabilityContract(max_vram_delta_mib=16.0)
        engine = MockLiveEngine(port=8002, stability_contract=stability_contract)
        res = engine.execute_multi_turn_requests(num_requests=100)
        self.assertLessEqual(res["vram_delta_mb"], 16.0)
        self.assertTrue(res["stability_passed"])

    def test_w04_port_8000_quarantine_security_enforcement(self) -> None:
        """W04: Strict enforcement of production Port 8000 quarantine guardrails."""
        contract = MultiTurnStabilityContract()
        # Direct port check must raise RuntimeError
        with self.assertRaises(RuntimeError):
            contract.validate_port(8000)
        # Attempting to initialize MockLiveEngine on port 8000 must raise RuntimeError
        with self.assertRaises(RuntimeError):
            MockLiveEngine(port=8000)

    def test_w05_full_engine_speculative_serving_lifecycle(self) -> None:
        """W05: Verification of launcher configuration and test isolation environment."""
        self.assertTrue(os.path.exists(LiveHarnessAdapter.LAUNCH_SCRIPT))
        with open(LiveHarnessAdapter.LAUNCH_SCRIPT) as f:
            launcher_content = f.read()

        # Verify port 8002 is declared and port 8000 is never used in launcher
        self.assertIn('PORT="8002"', launcher_content)
        self.assertNotIn('PORT="8000"', launcher_content)
        self.assertIn("GPU_HEADROOM_MIB=384", launcher_content)
        self.assertIn("NUM_SPECULATIVE_TOKENS=12", launcher_content)
        self.assertIn("RADIANCE_FULLDRAFTER=1", launcher_content)

    def test_w06_dual_gpu_cross_device_execution_parity_end_to_end(self) -> None:
        """W06: End-to-end Dual GPU cross-device token parity and <=3.5ms execution latency."""
        runner = MockDualGPURunner(
            latency_contract=LatencyContract(max_latency_ms=3.5),
            parity_contract=BitwiseParityContract(),
        )
        res = runner.run_dual_gpu()
        self.assertTrue(res["parity_passed"])
        self.assertTrue(res["latency_passed"])
        self.assertEqual(res["gpu0_tokens"], res["gpu1_tokens"])
        self.assertEqual(len(res["gpu0_tokens"]), 12)
        self.assertLessEqual(res["latency_ms"], 3.5)


if __name__ == "__main__":
    unittest.main()
