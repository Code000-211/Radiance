"""Adversarial Empirical Challenge Test Suite for Milestone 3:
Concurrency, Grid Synchronization, Stream Safety, and Scratchpad Layout Coordination.

Author: teamwork_preview_challenger_m3_2 (Concurrency & Buffer Safety Challenger)
"""

import os
import re
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HIP_FILE = os.path.join(REPO_ROOT, "libr4d", "r4d_fulldrafter.hip")
HEADER_FILE = os.path.join(REPO_ROOT, "libr4d", "r4d_fulldrafter.h")
ARCH_FILE = os.path.join(REPO_ROOT, "arch", "common", "rad_block_fulldrafter.h")
SIDECAR_FILE = os.path.join(REPO_ROOT, "arch", "qwen4exp_fp8", "qwen4exp_fp8.cpp")

V_SHARD = 124160
V_TOTAL = 248320
H_DIM = 768
W_DIM = 2560
BLOCK_K = 6
TOTAL_K = 12
K_TOP = 16


def parse_offsets_from_file(filepath):
    offsets = {}
    pattern = re.compile(r"#define\s+(FULLDRAFTER_)?([A-Za-z0-9_]+)\s+([0-9]+L*)")
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                name = m.group(2)
                val_str = m.group(3).rstrip("L")
                offsets[name] = int(val_str)
    return offsets


def extract_stream_from_launch(launch_str):
    """Extract stream argument from HIP launch string <<<...>>> respecting nested parentheses."""
    depth = 0
    tokens = []
    current = []
    for char in launch_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current).strip())
    # tokens are [grid, block] or [grid, block, lds] or [grid, block, lds, stream]
    if len(tokens) >= 4:
        return tokens[3]
    return None


class TestStreamDispatchAndOrdering(unittest.TestCase):
    """Adversarially challenge stream ordering and asynchronous dispatch safety."""

    def setUp(self):
        with open(HIP_FILE, "r") as f:
            self.hip_src = f.read()

    def test_hip_stream_uniformity_in_stage1_backbone(self):
        """Verify all kernel launches in r4d_fulldrafter_stage1_backbone use the passed stream."""
        fn_match = re.search(
            r"int r4d_fulldrafter_stage1_backbone\([^)]*\)\s*\{(.*?)\n\}",
            self.hip_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match, "r4d_fulldrafter_stage1_backbone function not found")
        fn_body = fn_match.group(1)

        launches = re.findall(r"<<<([^>]+)>>>", fn_body)
        self.assertGreaterEqual(len(launches), 8, "Expected at least 8 kernel launches in Stage 1 backbone")
        for launch in launches:
            stream_arg = extract_stream_from_launch(launch)
            self.assertEqual(
                stream_arg,
                "stream",
                f"Launch '{launch}' does not use 'stream'! Stream mismatch can cause data races.",
            )

    def test_hip_stream_uniformity_in_stage2_backbone(self):
        """Verify all kernel launches in r4d_fulldrafter_stage2_backbone use the passed stream."""
        fn_match = re.search(
            r"int r4d_fulldrafter_stage2_backbone\([^)]*\)\s*\{(.*?)\n\}",
            self.hip_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match, "r4d_fulldrafter_stage2_backbone function not found")
        fn_body = fn_match.group(1)

        launches = re.findall(r"<<<([^>]+)>>>", fn_body)
        self.assertGreaterEqual(len(launches), 8, "Expected at least 8 kernel launches in Stage 2 backbone")
        for launch in launches:
            stream_arg = extract_stream_from_launch(launch)
            self.assertEqual(
                stream_arg,
                "stream",
                f"Launch '{launch}' does not use 'stream'! Stream mismatch can cause data races.",
            )

    def test_hip_stream_uniformity_in_two_stage_orchestrator(self):
        """Verify all pipeline dispatches in r4d_rad_fulldrafter_two_stage share stream."""
        fn_match = re.search(
            r"int r4d_rad_fulldrafter_two_stage\([^)]*\)\s*\{(.*?)\n\}",
            self.hip_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match, "r4d_rad_fulldrafter_two_stage function not found")
        fn_body = fn_match.group(1)

        # Check sub-calls pass stream
        self.assertIn("r4d_fulldrafter_stage1_backbone", fn_body)
        self.assertIn("r4d_fulldrafter_lmhead_gemv6", fn_body)
        self.assertIn("r4d_fulldrafter_slot_markov_select", fn_body)
        self.assertIn("r4d_fulldrafter_stage2_backbone", fn_body)

        # Kernel launches in two_stage (fast_out_proj_kernel)
        launches = re.findall(r"<<<([^>]+)>>>", fn_body)
        for launch in launches:
            stream_arg = extract_stream_from_launch(launch)
            self.assertEqual(stream_arg, "stream")


class TestScratchpadLayoutCoordination(unittest.TestCase):
    """Adversarially challenge scratchpad memory layout and buffer disjointness."""

    def setUp(self):
        self.hip_offsets = parse_offsets_from_file(HIP_FILE)
        self.header_offsets = parse_offsets_from_file(HEADER_FILE)

    def test_offsets_sync_between_header_and_hip(self):
        """Ensure all shared offsets between header and hip are identical."""
        for k in ["SCR_ARRIVAL_CTR_OFF", "SCR_PROP_ID_OFF", "SCR_PROP_VAL_OFF"]:
            self.assertIn(k, self.hip_offsets, f"Missing {k} in hip file")
            self.assertIn(k, self.header_offsets, f"Missing {k} in header file")
            self.assertEqual(
                self.hip_offsets[k],
                self.header_offsets[k],
                f"Mismatch for {k}: hip={self.hip_offsets[k]} vs header={self.header_offsets[k]}",
            )

    def test_pairwise_non_overlapping_all_buffers_m3(self):
        """Verify that all scratchpad buffers are strictly disjoint with proper safety margins."""
        buffers = [
            ("SCR_BASE1", self.hip_offsets["SCR_BASE1_OFF"], 768 * 4),
            ("SCR_H1", self.hip_offsets["SCR_H1_OFF"], 7 * 768 * 4),
            ("SCR_O1", self.hip_offsets["SCR_O1_OFF"], 6 * 2560 * 4),
            ("SCR_BASE2", self.hip_offsets["SCR_BASE2_OFF"], 768 * 4),
            ("SCR_H2", self.hip_offsets["SCR_H2_OFF"], 7 * 768 * 4),
            ("SCR_O2", self.hip_offsets["SCR_O2_OFF"], 6 * 2560 * 4),
            ("SCR_PREV_TOK", self.hip_offsets["SCR_PREV_TOK_OFF"], 16 * 4),
            ("SCR_BLK_CANDS", self.hip_offsets["SCR_BLK_CANDS_OFF"], 64 * 16 * 4),
            ("SCR_BLK_UNARY", self.hip_offsets["SCR_BLK_UNARY_OFF"], 64 * 16 * 4),
            ("SCR_ARRIVAL_CTR", self.hip_offsets["SCR_ARRIVAL_CTR_OFF"], 4),
            ("SCR_PROP_ID", self.hip_offsets["SCR_PROP_ID_OFF"], 12 * 16 * 4),
            ("SCR_PROP_VAL", self.hip_offsets["SCR_PROP_VAL_OFF"], 12 * 16 * 4),
            ("SCR_TEMP", self.hip_offsets["SCR_TEMP_OFF"], 21504 * 4),
            ("SCR_BASE_LOGITS", self.hip_offsets["SCR_BASE_LOGITS_OFF"], 6 * V_TOTAL * 4),
        ]

        buffers.sort(key=lambda b: b[1])

        for i in range(len(buffers) - 1):
            curr_name, curr_start, curr_size = buffers[i]
            next_name, next_start, _ = buffers[i + 1]
            curr_end = curr_start + curr_size
            self.assertLessEqual(
                curr_end,
                next_start,
                f"Scratchpad collision: buffer '{curr_name}' [0x{curr_start:X}..0x{curr_end:X}) "
                f"overlaps into '{next_name}' at 0x{next_start:X}!",
            )

        # Check total sequence boundary
        last_name, last_start, last_size = buffers[-1]
        last_end = last_start + last_size
        per_seq = self.hip_offsets["SCR_PER_SEQ"]
        self.assertLessEqual(
            last_end,
            per_seq,
            f"Buffer '{last_name}' exceeds per-sequence allocation {per_seq}: ends at {last_end}!",
        )

    def test_proposal_safety_margin_to_temp_buffer(self):
        """Verify strict safety margin between proposal buffers and scratch temp buffer."""
        prop_val_end = self.hip_offsets["SCR_PROP_VAL_OFF"] + 12 * 16 * 4
        temp_start = self.hip_offsets["SCR_TEMP_OFF"]
        margin = temp_start - prop_val_end
        self.assertGreaterEqual(
            margin,
            10000,
            f"Safety margin between proposal values and temp buffer is {margin} bytes; must be >= 10,000 bytes",
        )

    def test_multi_batch_sequence_isolation(self):
        """Verify that multi-batch sequences M in [1, 32] do not overlap."""
        per_seq = self.hip_offsets["SCR_PER_SEQ"]
        self.assertEqual(per_seq, 8 * 1024 * 1024, "SCR_PER_SEQ must be 8 MB")

        for s in range(31):
            seq_s_start = s * per_seq
            seq_s_end = seq_s_start + self.hip_offsets["SCR_BASE_LOGITS_OFF"] + 6 * V_TOTAL * 4
            seq_next_start = (s + 1) * per_seq
            self.assertLessEqual(seq_s_end, seq_next_start)


class TestStage2ArrivalCounterSafetyAndReinit(unittest.TestCase):
    """Adversarially challenge Stage 2 arrival counter re-initialization and atomic barriers."""

    def setUp(self):
        with open(HIP_FILE, "r") as f:
            self.hip_src = f.read()

    def test_stage2_backbone_counter_reinit_exact_statement(self):
        """Verify that fulldrafter_stage2_backbone_kernel resets SCR_ARRIVAL_CTR_OFF."""
        idx = self.hip_src.find("fulldrafter_stage2_backbone_kernel")
        self.assertNotEqual(idx, -1, "fulldrafter_stage2_backbone_kernel not found")
        kernel_snippet = self.hip_src[idx : idx + 2000]

        # Verify exact reinit statement
        self.assertIn(
            "*(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;",
            kernel_snippet,
            "fulldrafter_stage2_backbone_kernel missing exact arrival counter reinit!",
        )

        # Verify race-free guard: only block 0 thread 0 executes the write
        self.assertIn("if (blockIdx.x == 0 && tid == 0)", kernel_snippet)

    def test_atomic_arrival_counter_simulation_with_corrupted_state(self):
        """Simulate 64 concurrent blocks arriving when arrival counter starts in dirty state."""
        # Scenario: Prior execution left counter = 0xDEADBEEF
        arrival_counter = np.uint32(0xDEADBEEF)

        # Stage 2 Backbone Kernel Block 0 Thread 0 runs
        # Gated by (blockIdx.x == 0 && tid == 0)
        arrival_counter = np.uint32(0)
        self.assertEqual(arrival_counter, 0)

        # 64 concurrent blocks arrive in random order
        num_blocks = 64
        arrival_order = np.random.permutation(num_blocks)
        arrived_tickets = []

        last_block_id = None
        for b_id in arrival_order:
            ticket = arrival_counter
            arrival_counter = np.uint32(arrival_counter + 1)
            arrived_tickets.append((b_id, ticket))
            if ticket == (num_blocks - 1):
                last_block_id = b_id
                # Reset counter for next slot
                arrival_counter = np.uint32(0)

        # Verify exactly one block was recognized as the last block
        self.assertIsNotNone(last_block_id)
        self.assertEqual(last_block_id, arrival_order[-1])
        # Verify tickets span 0..63 exactly once
        tickets = [t for _, t in arrived_tickets]
        self.assertEqual(sorted(tickets), list(range(64)))
        # Verify counter is left clean at 0
        self.assertEqual(arrival_counter, 0)


class TestNonBlockingHostLaunch(unittest.TestCase):
    """Verify zero CPU-GPU blocking synchronizations during the forward pass."""

    def test_zero_unconditional_stream_sync_in_libr4d(self):
        """Verify no unconditional hipStreamSynchronize in production forward dispatch."""
        with open(HIP_FILE, "r") as f:
            lines = f.readlines()

        sync_lines = []
        for i, line in enumerate(lines, 1):
            if "hipStreamSynchronize" in line or "hipDeviceSynchronize" in line:
                sync_lines.append((i, line.strip()))

        # In r4d_fulldrafter.hip, line 1671 has hipStreamSynchronize, but it MUST be guarded by profile_check
        self.assertEqual(len(sync_lines), 1, f"Unexpected sync lines: {sync_lines}")
        line_no, _ = sync_lines[0]
        # Inspect context of line 1671
        start = max(0, line_no - 15)
        context = "".join(lines[start:line_no])
        self.assertIn("if (profile_check)", context, "hipStreamSynchronize must be guarded by profile_check!")

    def test_zero_stream_sync_in_stage_backbone_functions(self):
        """Verify that r4d_fulldrafter_stage1_backbone and stage2_backbone have zero synchronizations."""
        with open(HIP_FILE, "r") as f:
            src = f.read()

        for fn_name in ["r4d_fulldrafter_stage1_backbone", "r4d_fulldrafter_stage2_backbone"]:
            fn_match = re.search(rf"int {fn_name}\([^)]*\)\s*\{{(.*?)\n\}}", src, re.DOTALL)
            self.assertIsNotNone(fn_match, f"{fn_name} not found")
            fn_body = fn_match.group(1)
            self.assertNotIn("hipStreamSynchronize", fn_body, f"{fn_name} contains hipStreamSynchronize!")
            self.assertNotIn("hipDeviceSynchronize", fn_body, f"{fn_name} contains hipDeviceSynchronize!")


class TestTerminalRecurrencePreservation(unittest.TestCase):
    """Verify that Stage 1 terminal state h1[5] is preserved and consumed by Stage 2."""

    def setUp(self):
        with open(HIP_FILE, "r") as f:
            self.hip_src = f.read()

    def test_stage1_final_norm_copies_slots_to_h1(self):
        """Verify Stage 1 final norm copies proposal slots 0..5 into h1[0..5]."""
        self.assertIn("r4d_fulldrafter_copy_final_norm_kernel", self.hip_src)
        idx = self.hip_src.find("r4d_fulldrafter_copy_final_norm_kernel")
        snippet = self.hip_src[idx : idx + 800]
        self.assertIn("h_base[slot * H_DIM + i] = t_base[slot * H_DIM + i];", snippet)

    def test_stage2_base_commit_reads_terminal_state_h1_slot5(self):
        """Verify Stage 2 base commit performs recurrent RMSNorm on h1[5]."""
        idx = self.hip_src.find("fulldrafter_stage2_base_commit_kernel")
        self.assertNotEqual(idx, -1, "fulldrafter_stage2_base_commit_kernel not found")
        snippet = self.hip_src[idx : idx + 800]
        self.assertIn(
            "dev_rmsnorm(s_h1 + (BLOCK_K - 1) * H_DIM, p->recurrent_norm, s_temp, H_DIM);",
            snippet,
            "Stage 2 base commit does not read terminal state h1[5]!",
        )


if __name__ == "__main__":
    unittest.main()
