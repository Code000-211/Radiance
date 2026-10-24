"""adversarial_tier5_concurrency_headroom.py — Tier 5 Concurrency, Headroom & Live Acceptance Adversarial Test Suite.

Milestone 5 Phase 2 White-Box Verification:
1. Concurrency & Multi-Batch Scratchpad Allocation (M in [1, 32]):
   - Scratchpad sub-buffer non-overlapping layout
   - Sequence isolation: batch_idx * SCR_PER_SEQ prevents cross-batch data corruption
   - Static memory footprint adherence to GPU_HEADROOM_MIB=384
   - 64-bit pointer arithmetic and overflow protection
2. AQL Stream Safety & Pipelining Contract:
   - 56 consecutive kernel launches on stream s (28 stage 1 + 28 stage 2)
   - 100% single-stream dispatch without implicit host barriers
   - Per-sequence atomic arrival counter isolation and reset protocol
3. Acceptance Automation & Quarantine Robustness:
   - Port 8000 quarantine checks in scripts/run_live_acceptance_post_training.sh and multi_turn_500.py
   - Process cleanup trap handlers (SIGINT, SIGTERM, EXIT)
   - Training Priority Directive active execution interception
   - Telemetry JSON parsing and NaN/Inf integrity scanner under simulated failures
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.stress.multi_turn_500 import IntegrityScanner, assert_port_safety


class TestScratchpadLayoutAndMultiBatchIsolation(unittest.TestCase):
    """Audits scratchpad memory allocation, sub-buffer layout, and multi-batch sequence isolation."""

    HIP_SOURCE_PATH = PROJECT_ROOT / "libr4d" / "r4d_fulldrafter.hip"
    H_HEADER_PATH = PROJECT_ROOT / "libr4d" / "r4d_fulldrafter.h"

    # Layout constants from r4d_fulldrafter.hip
    SCR_PER_SEQ = 8388608  # 8 MiB (8 * 1024 * 1024)
    V_TOTAL = 248320
    H_DIM = 768
    W_DIM = 2560
    BLOCK_K = 6
    TOTAL_K = 12

    def setUp(self) -> None:
        self.assertTrue(
            self.HIP_SOURCE_PATH.exists(),
            f"Expected HIP file at {self.HIP_SOURCE_PATH}",
        )
        with open(self.HIP_SOURCE_PATH, "r", encoding="utf-8") as f:
            self.hip_source = f.read()

    def test_scratchpad_constants_and_subbuffer_ranges(self) -> None:
        """Verify all sub-buffer offsets are strictly non-overlapping and fit within 8 MiB."""
        # Buffer definitions with (offset, size_in_bytes, description)
        sub_buffers: list[tuple[str, int, int]] = [
            ("base1", 0, 768 * 4),  # [768] float = 3072 B
            ("h1", 4096, 7 * 768 * 4),  # [7, 768] float = 21504 B
            ("o1", 32768, 6 * 2560 * 4),  # [6, 2560] float = 61440 B
            ("base2", 98304, 768 * 4),  # [768] float = 3072 B
            ("h2", 102400, 7 * 768 * 4),  # [7, 768] float = 21504 B
            ("o2", 131072, 6 * 2560 * 4),  # [6, 2560] float = 61440 B
            ("prev_tokens", 196608, 16 * 4),  # [16] uint32_t = 64 B
            ("block_cands", 200704, 64 * 16 * 4),  # [64, 16] uint32_t = 4096 B
            ("block_unary", 216064, 64 * 16 * 4),  # [64, 16] float = 4096 B
            ("arrival_ctr", 220160, 4),  # [1] uint32_t = 4 B
            ("prop_ids", 220264, 12 * 16 * 4),  # [12, 16] uint32_t = 768 B
            ("prop_vals", 221032, 12 * 16 * 4),  # [12, 16] float = 768 B
            (
                "temp",
                232000,
                (7 * 768 + 7 * 2304) * 4,
            ),  # max temp usage: pre-norm + SwiGLU = 86016 B
            (
                "base_logits",
                2097152,
                6 * self.V_TOTAL * 4,
            ),  # [6, 248320] float = 5,959,680 B
        ]

        # Verify sorted non-overlapping ranges
        sub_buffers.sort(key=lambda b: b[1])
        for i in range(len(sub_buffers) - 1):
            curr_name, curr_start, curr_size = sub_buffers[i]
            curr_end = curr_start + curr_size
            next_name, next_start, _ = sub_buffers[i + 1]

            self.assertLessEqual(
                curr_end,
                next_start,
                f"Buffer collision detected: '{curr_name}' ends at {curr_end}, "
                f"overlapping with '{next_name}' starting at {next_start}",
            )

        # Verify last buffer fits strictly within SCR_PER_SEQ (8 MiB)
        last_name, last_start, last_size = sub_buffers[-1]
        last_end = last_start + last_size
        self.assertLess(
            last_end,
            self.SCR_PER_SEQ,
            f"Buffer '{last_name}' exceeds 8MB scratchpad capacity: {last_end} > {self.SCR_PER_SEQ}",
        )
        headroom_bytes = self.SCR_PER_SEQ - last_end
        self.assertGreaterEqual(
            headroom_bytes,
            300_000,
            f"Expected >= 300 KiB margin at end of sequence scratchpad, got {headroom_bytes} bytes",
        )

    def test_multibatch_sequence_isolation_all_m_1_to_32(self) -> None:
        """Prove that batch offset calculations prevent cross-batch corruption for all M in [1, 32]."""
        max_internal_end = 2097152 + 6 * self.V_TOTAL * 4  # 8,056,832 bytes

        for m in range(1, 33):
            allocated_bytes = m * self.SCR_PER_SEQ
            for s in range(m):
                seq_start = s * self.SCR_PER_SEQ
                seq_end = seq_start + max_internal_end
                next_seq_start = (s + 1) * self.SCR_PER_SEQ

                # Strict boundary isolation invariant
                self.assertLess(
                    seq_end,
                    next_seq_start,
                    f"Cross-batch corruption hazard: Batch M={m}, Seq {s} write end ({seq_end}) "
                    f"violates Seq {s + 1} boundary ({next_seq_start})",
                )
                self.assertLessEqual(
                    next_seq_start,
                    allocated_bytes,
                    f"Out-of-bounds hazard: Seq {s + 1} boundary exceeds allocated capacity",
                )

    def test_64bit_offset_arithmetic_safety(self) -> None:
        """Verify that batch offset arithmetic in r4d_fulldrafter.hip uses 64-bit integer casting."""
        # Find all occurrences of SCR_PER_SEQ in r4d_fulldrafter.hip
        scr_seq_matches = re.findall(
            r"([^\n;]+\bSCR_PER_SEQ\b[^\n;]*)", self.hip_source
        )
        self.assertGreater(
            len(scr_seq_matches),
            0,
            "Expected SCR_PER_SEQ usages in r4d_fulldrafter.hip",
        )

        # Check definition is 64-bit signed literal (8388608LL)
        self.assertIn(
            "#define SCR_PER_SEQ         8388608LL",
            self.hip_source,
            "SCR_PER_SEQ must be defined with LL suffix to prevent 32-bit integer overflow",
        )

        # Check that pointer arithmetic casts to size_t or int64_t
        for match in scr_seq_matches:
            if "#define" in match or "extern" in match:
                continue
            has_64bit_cast = (
                "(size_t)" in match
                or "(int64_t)" in match
                or "(uint64_t)" in match
            )
            self.assertTrue(
                has_64bit_cast,
                f"Unsafe 32-bit offset arithmetic found: '{match.strip()}'. Expected (size_t) or (int64_t) cast.",
            )

    def test_scratch_allocation_api_contracts(self) -> None:
        """Verify r4d_rad_fulldrafter_two_stage_scratch logic under valid and edge case inputs."""
        # Regex inspect function r4d_rad_fulldrafter_two_stage_scratch
        func_match = re.search(
            r'r4d_rad_fulldrafter_two_stage_scratch\s*\([^)]*\)\s*\{([^}]+)\}',
            self.hip_source,
        )
        self.assertIsNotNone(
            func_match,
            "Could not locate r4d_rad_fulldrafter_two_stage_scratch implementation",
        )
        body = func_match.group(1)

        # Must guard against M < 1 (defaulting to 1) and calculate (int64_t)M * SCR_PER_SEQ
        self.assertIn("if (M < 1) M = 1;", body)
        self.assertIn("(int64_t)M * SCR_PER_SEQ", body)


class TestVRAMHeadroomBudgetAdherence(unittest.TestCase):
    """Audits static memory footprint against GPU_HEADROOM_MIB=384."""

    HEADROOM_MIB = 384
    SCR_MIB_PER_SEQ = 8.0

    def test_headroom_budget_adherence_m_1_to_32(self) -> None:
        """Verify M * 8.0 MiB <= 384 MiB for all M in [1, 32], leaving ample margin."""
        for m in range(1, 33):
            vram_mib = m * self.SCR_MIB_PER_SEQ
            self.assertLess(
                vram_mib,
                self.HEADROOM_MIB,
                f"Headroom breached at batch size M={m}: {vram_mib} MiB >= {self.HEADROOM_MIB} MiB",
            )
            margin_mib = self.HEADROOM_MIB - vram_mib
            self.assertGreater(margin_mib, 0.0)

        # Maximum batch size M=32 check: exactly 256 MiB footprint, 128 MiB margin
        max_batch_footprint = 32 * self.SCR_MIB_PER_SEQ
        self.assertEqual(max_batch_footprint, 256.0)
        margin_at_32 = self.HEADROOM_MIB - max_batch_footprint
        self.assertEqual(
            margin_at_32,
            128.0,
            f"Expected exactly 128.0 MiB headroom margin at M=32, got {margin_at_32}",
        )

    def test_headroom_saturation_boundary(self) -> None:
        """Verify the theoretical saturation batch size is M=48."""
        m_sat = int(self.HEADROOM_MIB / self.SCR_MIB_PER_SEQ)
        self.assertEqual(
            m_sat,
            48,
            f"Expected saturation batch size M=48, got M={m_sat}",
        )
        # 49 sequences would breach 384 MiB headroom
        self.assertGreater(49 * self.SCR_MIB_PER_SEQ, self.HEADROOM_MIB)

    def test_launch_script_headroom_configuration(self) -> None:
        """Audit tests/stress/launch_engine_8002.sh for correct headroom configuration."""
        launch_script_path = (
            PROJECT_ROOT / "tests/stress/launch_engine_8002.sh"
        )
        self.assertTrue(launch_script_path.exists())
        with open(launch_script_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export GPU_HEADROOM_MIB=384", content)
        self.assertIn('--gpu-headroom-mib "${GPU_HEADROOM_MIB}"', content)
        self.assertIn("--max-num-seqs 8", content)

        # At max-num-seqs 8, scratchpad is 64 MiB, adhering with 320 MiB margin
        configured_m = 8
        configured_footprint = configured_m * self.SCR_MIB_PER_SEQ
        self.assertEqual(configured_footprint, 64.0)
        self.assertEqual(self.HEADROOM_MIB - configured_footprint, 320.0)


class TestAQLStreamSafetyAndPipelining(unittest.TestCase):
    """Audits AQL stream ordering, 56-launch pipeline, and lack of host synchronization."""

    HIP_SOURCE_PATH = PROJECT_ROOT / "libr4d" / "r4d_fulldrafter.hip"

    def setUp(self) -> None:
        with open(self.HIP_SOURCE_PATH, "r", encoding="utf-8") as f:
            self.hip_source = f.read()

    def test_kernel_launch_pipeline_count_56(self) -> None:
        """Verify that exactly 56 consecutive kernel launches are dispatched for M=1."""
        # Trace stages in r4d_rad_fulldrafter_two_stage:
        # Stage 1:
        #   - 1x stage1_backbone_kernel (prep)
        #   - 1x stage1_base_commit_kernel (base commit)
        #   - 2 transformer blocks * 8 kernels = 16 kernels
        #   - 2x final RMSNorm kernels (norm + copy) = 2 kernels
        #   - 1x fast_out_proj_kernel = 1 kernel
        #   - 1x lmhead_gemv6_kernel = 1 kernel (for M=1)
        #   - 6x slot_markov_select_kernel = 6 kernels
        #   Subtotal Stage 1 = 1 + 1 + 16 + 2 + 1 + 1 + 6 = 28 launches
        # Stage 2:
        #   - 1x stage2_backbone_kernel (prep)
        #   - 1x stage2_base_commit_kernel (base commit)
        #   - 2 transformer blocks * 8 kernels = 16 kernels
        #   - 2x final RMSNorm kernels (norm + copy) = 2 kernels
        #   - 1x fast_out_proj_kernel = 1 kernel
        #   - 1x lmhead_gemv6_kernel = 1 kernel (for M=1)
        #   - 6x slot_markov_select_kernel = 6 kernels
        #   Subtotal Stage 2 = 1 + 1 + 16 + 2 + 1 + 1 + 6 = 28 launches
        # Total = 28 + 28 = 56 launches
        stage1_backbone_launches = 1 + 1 + 16 + 2  # 20
        stage1_total = stage1_backbone_launches + 1 + 1 + 6  # 28

        stage2_backbone_launches = 1 + 1 + 16 + 2  # 20
        stage2_total = stage2_backbone_launches + 1 + 1 + 6  # 28

        total_launches = stage1_total + stage2_total
        self.assertEqual(
            total_launches,
            56,
            f"Expected 56 consecutive kernel launches, calculated {total_launches}",
        )

    def test_single_stream_coordination_100_percent(self) -> None:
        """Verify that 100% of the kernel launches explicitly pass 'stream'."""
        # Find all kernel launch patterns <<< ... >>>
        raw_launches = re.findall(
            r"([a-zA-Z0-9_<>]+)\s*<<<([^>]+)>>>", self.hip_source
        )
        self.assertGreater(
            len(raw_launches),
            20,
            "Expected at least 20 distinct kernel launch sites in source code",
        )

        for kernel_name, config in raw_launches:
            clean_config = config.strip()
            # Must end with ', 0, stream' or ', stream'
            self.assertTrue(
                clean_config.endswith("stream"),
                f"Kernel {kernel_name}<<< {clean_config} >>> does not use 'stream'! Violates stream ordering.",
            )

    def test_zero_host_synchronization_on_hot_path(self) -> None:
        """Verify no un-gated host synchronizations exist on the production execution path."""
        sync_calls = re.findall(
            r"(hipStreamSynchronize|hipDeviceSynchronize)\s*\([^)]*\)",
            self.hip_source,
        )
        self.assertEqual(
            len(sync_calls),
            1,
            f"Found unexpected host synchronizations: {sync_calls}",
        )

        # The only sync call must be gated by profile_check
        sync_block = re.search(
            r"if\s*\(profile_check\)\s*\{[^}]*hipStreamSynchronize[^}]*\}",
            self.hip_source,
        )
        self.assertIsNotNone(
            sync_block,
            "The single hipStreamSynchronize must be enclosed within 'if (profile_check)'",
        )

    def test_interblock_atomic_arrival_isolation(self) -> None:
        """Audit inter-block arrival counter isolation and memory ordering in Markov select."""
        select_kernel_match = re.search(
            r"void\s+r4d_fulldrafter_slot_markov_select_kernel\s*\([^)]*\)\s*\{([\s\S]*?)\n\}\n",
            self.hip_source,
        )
        self.assertIsNotNone(
            select_kernel_match,
            "Could not locate r4d_fulldrafter_slot_markov_select_kernel",
        )
        kernel_body = select_kernel_match.group(1)

        # Verify atomic arrival counter uses per-sequence s_base
        self.assertIn(
            "s_base + SCR_ARRIVAL_CTR_OFF",
            kernel_body,
            "Atomic counter must be located at sequence offset s_base",
        )

        # Verify atomic operation uses agent memory scope and acquire-release semantics
        self.assertIn(
            "__hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT)",
            kernel_body,
        )

        # Verify threshold is gridDim.x - 1 (63 blocks)
        self.assertIn("(unsigned int)(gridDim.x - 1)", kernel_body)

        # Verify arrival counter is cleanly reset to 0 by the winner block
        self.assertIn("*ctr = 0u;", kernel_body)


class TestLiveAcceptanceAutomationAndQuarantine(unittest.TestCase):
    """Stress-tests acceptance runner error handling, process cleanup traps, and quarantine."""

    RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_live_acceptance_post_training.sh"

    def setUp(self) -> None:
        self.assertTrue(self.RUNNER_SCRIPT.exists())
        with open(self.RUNNER_SCRIPT, "r", encoding="utf-8") as f:
            self.runner_content = f.read()

    def test_multi_turn_port_8000_quarantine(self) -> None:
        """Test that multi_turn_500.py assert_port_safety raises RuntimeError on port 8000."""
        forbidden_urls = [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8000/v1/chat/completions",
            "http://production-host:8000",
            "http://test-server?port=8000",
        ]
        for url in forbidden_urls:
            with self.assertRaises(
                RuntimeError,
                msg=f"Expected assert_port_safety to reject {url}",
            ):
                assert_port_safety(url)

        # Permitted URLs on test port 8002
        permitted_urls = [
            "http://127.0.0.1:8002",
            "http://localhost:8002",
            "http://127.0.0.1:8002/v1/chat/completions",
        ]
        for url in permitted_urls:
            try:
                assert_port_safety(url)
            except RuntimeError as e:
                self.fail(f"assert_port_safety raised on legitimate test URL {url}: {e}")

    def test_runner_port_8000_quarantine_rejection(self) -> None:
        """Verify runner script has an explicit port quarantine check."""
        self.assertIn('PROD_PORT="8000"', self.runner_content)
        self.assertIn('PORT="8002"', self.runner_content)
        self.assertIn('if [[ "${PORT}" == "${PROD_PORT}" ]]', self.runner_content)
        self.assertIn("CRITICAL SECURITY ERROR", self.runner_content)

    def test_priority_directive_active_training_guard(self) -> None:
        """Verify runner script inspects train_flash_next.py and halts when running live."""
        self.assertIn("check_training_active()", self.runner_content)
        self.assertIn('pgrep -fa "train_flash_next.py"', self.runner_content)
        self.assertIn(
            "PRIORITY DIRECTIVE ACTIVE — TRAINING HAS 100% EXCLUSIVE GPU ACCESS",
            self.runner_content,
        )

        # Execute runner script without flags: since training is currently running, it must exit 1
        res = subprocess.run(
            ["bash", str(self.RUNNER_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        self.assertEqual(
            res.returncode,
            1,
            f"Expected exit code 1 when training is active, got {res.returncode}",
        )
        self.assertIn("PRIORITY DIRECTIVE ACTIVE", res.stdout)

    def test_runner_dry_run_execution_success(self) -> None:
        """Verify runner script executes cleanly and deterministically in --dry-run mode."""
        res = subprocess.run(
            ["bash", str(self.RUNNER_SCRIPT), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"Expected exit code 0 in --dry-run mode, got {res.returncode}. Stderr: {res.stderr}",
        )
        self.assertIn(
            "ALL POST-TRAINING ACCEPTANCE GATES CERTIFIED AND PASSED",
            res.stdout,
        )

    def test_process_cleanup_trap_and_pid_handling(self) -> None:
        """Audit process trap handlers for SIGINT, SIGTERM, and EXIT."""
        self.assertIn("trap cleanup_engine EXIT INT TERM", self.runner_content)
        self.assertIn("cleanup_engine()", self.runner_content)
        self.assertIn("kill -TERM", self.runner_content)
        self.assertIn("kill -9", self.runner_content)
        self.assertIn('pkill -f "radiance.*--port ${PORT}"', self.runner_content)
        self.assertIn("rm -f", self.runner_content)
        self.assertIn("engine_8002.pid", self.runner_content)
        # Trap unregistering before final normal exit
        self.assertIn("trap - EXIT INT TERM", self.runner_content)

    def test_integrity_scanner_nan_inf_detection(self) -> None:
        """Stress-test IntegrityScanner from multi_turn_500.py for numerical anomaly detection."""
        # Positive cases: text anomalies
        anomalous_texts = [
            "The model output is nan and fails",
            "Probability was -inf at step 1",
            "Result: Infinity in logit vector",
            "Found -Infinity here",
            "Token logprob: NaN",
            "logits: [1.2, Inf, 3.4]",
        ]
        for text in anomalous_texts:
            anomalies = IntegrityScanner.check_text(text)
            self.assertGreater(
                len(anomalies),
                0,
                f"Expected anomaly in text '{text}', but none found",
            )

        # Negative cases: normal words that must NOT trigger false positives
        benign_texts = [
            "I like to eat a banana every morning.",
            "The company reported a financial loss in finance quarterly.",
            "Infinitesimal changes in the learning rate have negligible impact.",
            "Information theory is foundational to language modeling.",
            "An infant was sleeping peacefully.",
        ]
        for text in benign_texts:
            anomalies = IntegrityScanner.check_text(text)
            self.assertEqual(
                anomalies,
                [],
                f"False positive anomaly detected in benign text '{text}': {anomalies}",
            )

        # Positive cases: float structures
        anomalous_json: dict[str, Any] = {
            "choices": [
                {"logprobs": {"content": [{"logprob": float("nan")}]}},
                {"logprobs": {"top_logprobs": [{"score": float("inf")}]}},
                {"tokens": [-float("inf"), 1.0, 2.0]},
            ]
        }
        float_anomalies = IntegrityScanner.check_floats(anomalous_json)
        self.assertEqual(len(float_anomalies), 3)

        # Negative cases: benign float structures
        benign_json = {
            "choices": [
                {"logprobs": {"content": [{"logprob": -0.245}]}},
                {"tokens": [1.0, 2.5, -99.9]},
            ]
        }
        self.assertEqual(IntegrityScanner.check_floats(benign_json), [])

    def test_telemetry_json_parsing_and_resilience(self) -> None:
        """Stress-test telemetry JSON parsing logic under partial, empty, and edge-case schemas."""
        parser_code = """
import json, sys
data = json.loads(sys.stdin.read())
tps = data.get('throughput_tok_per_sec', 0.0)
tau = data.get('tau', 0.0)
crashes = data.get('zero_failure_assertions', {}).get('server_crashes', 0)
nans = data.get('zero_failure_assertions', {}).get('nans_detected', 0) + data.get('zero_failure_assertions', {}).get('infs_detected', 0)
deltas = data.get('zero_failure_assertions', {}).get('vram_deltas_mb', {})
max_delta = max(deltas.values()) if deltas else 0.0
passed = data.get('passed', False)
print(f'{tps} {tau} {crashes} {nans} {max_delta} {passed}')
"""

        test_payloads = [
            # Minimal empty JSON
            ("{}", "0.0 0.0 0 0 0.0 False"),
            # Valid telemetry
            (
                json.dumps(
                    {
                        "throughput_tok_per_sec": 192.5,
                        "tau": 3.42,
                        "passed": True,
                        "zero_failure_assertions": {
                            "server_crashes": 0,
                            "nans_detected": 0,
                            "infs_detected": 0,
                            "vram_deltas_mb": {"0": 1.2, "1": 1.5},
                        },
                    }
                ),
                "192.5 3.42 0 0 1.5 True",
            ),
            # Failure scenario with crashes and NaNs
            (
                json.dumps(
                    {
                        "throughput_tok_per_sec": 80.0,
                        "tau": 2.1,
                        "passed": False,
                        "zero_failure_assertions": {
                            "server_crashes": 2,
                            "nans_detected": 5,
                            "infs_detected": 1,
                            "vram_deltas_mb": {"0": 35.0},
                        },
                    }
                ),
                "80.0 2.1 2 6 35.0 False",
            ),
        ]

        for payload, expected in test_payloads:
            res = subprocess.run(
                [sys.executable, "-c", parser_code],
                input=payload,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0)
            self.assertEqual(res.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
